#pragma once

#include "InternalTypes.h"
#include "ParticleMath.cuh"
#include "CurlNoise.cuh"
#include "Collisions.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace vparticles {

__global__ void simulateKernel(
    ParticlePool pool,
    DeviceFrameState* state,
    uint8_t* aliveFlags,
    uint32_t* freeIndices,
    const GraphParams* params,
    uint32_t recipeCount,
    uint32_t tileCount)
{
    const float simTime = params->simTime;
    const float dt = params->dt;
    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];

        // Recipe lookup: when recipeCount <= 1, avoid reading pool.systemId from DRAM (0 bytes overhead!)
        const uint32_t recipeId = (recipeCount > 1)
            ? (pool.systemId[particleIndex] % kMaxRecipes)
            : 0u;
        const EffectRecipe& recipe = cRecipes[recipeId];

        const float age = pool.age[particleIndex] + dt;
        const float lifetime = pool.lifetime[particleIndex] * recipe.lifetimeScale;
        if (age >= lifetime) {
            pool.age[particleIndex] = age;
            aliveFlags[compactIndex] = 0;
            const uint32_t freeSlot = atomicAdd(&state->freeCount, 1u);
            freeIndices[freeSlot] = particleIndex;
            atomicAdd(&state->deadCount, 1u);
            continue;
        }

        float4 velocity = pool.vel[particleIndex];
        float4 position = pool.pos[particleIndex];
        const float dragFactor = fmaxf(0.0f, 1.0f - recipe.drag * dt);

        // Forces: gravity + wind + drag
        velocity.x = (velocity.x + (recipe.gravity.x + recipe.wind.x) * dt) * dragFactor;
        velocity.y = (velocity.y + (recipe.gravity.y + recipe.wind.y) * dt) * dragFactor;
        velocity.z = (velocity.z + (recipe.gravity.z + recipe.wind.z) * dt) * dragFactor;

        // Fused turbulence module (divergence-free curl noise)
        if (recipe.turbulence.strength > 0.0f) {
            const float3 turb = computeCurlNoise(
                make_float3(position.x, position.y, position.z),
                simTime * recipe.turbulence.speed,
                recipe.turbulence.frequency,
                recipe.turbulence.strength);
            velocity.x += turb.x * dt;
            velocity.y += turb.y * dt;
            velocity.z += turb.z * dt;
        }

        // Integration
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;

        // Fused analytical collision modules
        if (recipe.plane.enabled) {
            resolvePlaneCollision(position, velocity, recipe.plane);
        }
        if (recipe.sphere.enabled) {
            resolveSphereCollision(position, velocity, recipe.sphere);
        }
        if (recipe.box.enabled) {
            resolveBoxCollision(position, velocity, recipe.box);
        }

        // Fused color & size curves
        float4 color = pool.color[particleIndex];
        const float normalizedAge = fminf(1.0f, fmaxf(0.0f, age / lifetime));

        if (recipe.curves.enabled) {
            // Size-over-life
            if (normalizedAge < 0.5f) {
                const float t = normalizedAge * 2.0f;
                position.w = recipe.curves.startSize + (recipe.curves.peakSize - recipe.curves.startSize) * t;
            } else {
                const float t = (normalizedAge - 0.5f) * 2.0f;
                position.w = recipe.curves.peakSize + (recipe.curves.endSize - recipe.curves.peakSize) * t;
            }

            // Color-over-life
            const float t = normalizedAge;
            color.x = recipe.curves.startColor.r * (1.0f - t) + recipe.curves.endColor.r * t;
            color.y = recipe.curves.startColor.g * (1.0f - t) + recipe.curves.endColor.g * t;
            color.z = recipe.curves.startColor.b * (1.0f - t) + recipe.curves.endColor.b * t;
            color.w = recipe.curves.startColor.a * (1.0f - t) + recipe.curves.endColor.a * t;
        } else {
            color.w = fmaxf(0.0f, 1.0f - normalizedAge);
        }

        pool.vel[particleIndex] = velocity;
        pool.pos[particleIndex] = position;
        pool.color[particleIndex] = color;
        pool.age[particleIndex] = age;
        aliveFlags[compactIndex] = 1;
    }
}

__global__ void simulatePackedKernel(
    PackedPool pool,
    DeviceFrameState* state,
    uint8_t* aliveFlags,
    uint32_t* freeIndices,
    const GraphParams* params,
    float posScale,
    float invPosScale,
    uint32_t recipeCount,
    uint32_t tileCount)
{
    const float simTime = params->simTime;
    const float dt = params->dt;
    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];

        // Recipe lookup: when recipeCount <= 1, avoid reading pool.systemId from DRAM (0 bytes overhead!)
        const uint32_t recipeId = (recipeCount > 1)
            ? (pool.systemId[particleIndex] % kMaxRecipes)
            : 0u;
        const EffectRecipe& recipe = cRecipes[recipeId];

        // ── DECODE: packed DRAM → FP32 registers ─────────────────────
        const float lifetime = __half2float(pool.lifetime[particleIndex]) * recipe.lifetimeScale;
        const uint16_t packedAge = pool.age[particleIndex];
        const float normalizedAge = static_cast<float>(packedAge) * (1.0f / 65535.0f);
        const float age = normalizedAge * lifetime + dt;

        if (age >= lifetime) {
            pool.age[particleIndex] = 65535u;
            aliveFlags[compactIndex] = 0;
            const uint32_t freeSlot = atomicAdd(&state->freeCount, 1u);
            freeIndices[freeSlot] = particleIndex;
            atomicAdd(&state->deadCount, 1u);
            continue;
        }

        const uint16_t tileId = (tileCount > 1) ? pool.tileId[particleIndex] : 0u;
        const float originX = cTiles[tileId].originX;
        const float originY = cTiles[tileId].originY;
        const float originZ = cTiles[tileId].originZ;

        float4 velocity = make_float4(
            __half2float(pool.velX[particleIndex]),
            __half2float(pool.velY[particleIndex]),
            __half2float(pool.velZ[particleIndex]),
            0.0f);
        float4 position = make_float4(
            decodePosition(pool.posX[particleIndex], originX, invPosScale),
            decodePosition(pool.posY[particleIndex], originY, invPosScale),
            decodePosition(pool.posZ[particleIndex], originZ, invPosScale),
            __half2float(pool.size[particleIndex]));

        const float dragFactor = fmaxf(0.0f, 1.0f - recipe.drag * dt);

        // Forces: gravity + wind + drag
        velocity.x = (velocity.x + (recipe.gravity.x + recipe.wind.x) * dt) * dragFactor;
        velocity.y = (velocity.y + (recipe.gravity.y + recipe.wind.y) * dt) * dragFactor;
        velocity.z = (velocity.z + (recipe.gravity.z + recipe.wind.z) * dt) * dragFactor;

        // Fused turbulence module (divergence-free curl noise)
        if (recipe.turbulence.strength > 0.0f) {
            const float3 turb = computeCurlNoise(
                make_float3(position.x, position.y, position.z),
                simTime * recipe.turbulence.speed,
                recipe.turbulence.frequency,
                recipe.turbulence.strength);
            velocity.x += turb.x * dt;
            velocity.y += turb.y * dt;
            velocity.z += turb.z * dt;
        }

        // Integration
        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;

        // Fused analytical collision modules
        if (recipe.plane.enabled) {
            resolvePlaneCollision(position, velocity, recipe.plane);
        }
        if (recipe.sphere.enabled) {
            resolveSphereCollision(position, velocity, recipe.sphere);
        }
        if (recipe.box.enabled) {
            resolveBoxCollision(position, velocity, recipe.box);
        }

        // Fused color & size curves
        float4 color = decodeColorRGBA8(pool.colorPacked[particleIndex]);
        const float newNormalizedAge = fminf(1.0f, fmaxf(0.0f, age / lifetime));

        if (recipe.curves.enabled) {
            // Size-over-life
            if (newNormalizedAge < 0.5f) {
                const float t = newNormalizedAge * 2.0f;
                position.w = recipe.curves.startSize + (recipe.curves.peakSize - recipe.curves.startSize) * t;
            } else {
                const float t = (newNormalizedAge - 0.5f) * 2.0f;
                position.w = recipe.curves.peakSize + (recipe.curves.endSize - recipe.curves.peakSize) * t;
            }

            // Color-over-life
            const float t = newNormalizedAge;
            color.x = recipe.curves.startColor.r * (1.0f - t) + recipe.curves.endColor.r * t;
            color.y = recipe.curves.startColor.g * (1.0f - t) + recipe.curves.endColor.g * t;
            color.z = recipe.curves.startColor.b * (1.0f - t) + recipe.curves.endColor.b * t;
            color.w = recipe.curves.startColor.a * (1.0f - t) + recipe.curves.endColor.a * t;
        } else {
            color.w = fmaxf(0.0f, 1.0f - newNormalizedAge);
        }

        // ── ENCODE: FP32 registers → packed DRAM ─────────────────────
        pool.posX[particleIndex] = encodePosition(position.x, originX, posScale);
        pool.posY[particleIndex] = encodePosition(position.y, originY, posScale);
        pool.posZ[particleIndex] = encodePosition(position.z, originZ, posScale);
        pool.velX[particleIndex] = __float2half(velocity.x);
        pool.velY[particleIndex] = __float2half(velocity.y);
        pool.velZ[particleIndex] = __float2half(velocity.z);
        pool.age[particleIndex] = static_cast<uint16_t>(
            fminf(65535.0f, __float2uint_rn(age / lifetime * 65535.0f)));
        pool.colorPacked[particleIndex] = encodeColorRGBA8(color.x, color.y, color.z, color.w);
        pool.size[particleIndex] = __float2half(position.w);
        aliveFlags[compactIndex] = 1;
    }
}

} // namespace vparticles
