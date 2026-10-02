#include "VParticles/ParticleSystem.h"

#include "VParticles/CudaCheck.h"

#include <cub/cub.cuh>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vparticles {
namespace {

__constant__ TileDesc cTiles[kMaxTiles];
__constant__ EffectRecipe cRecipes[kMaxRecipes];

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kTargetBlocksPerMultiprocessor = 8;
constexpr size_t kTelemetryRingSize = 8;
constexpr size_t kSpawnUploadRingSize = 8;

template <typename T>
void cudaAlloc(T*& pointer, size_t count)
{
    VP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
}

uint32_t blockCount(uint32_t itemCount)
{
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(itemCount) + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

void allocatePool(ParticlePool& pool, uint32_t capacity)
{
    cudaAlloc(pool.pos, capacity);
    cudaAlloc(pool.vel, capacity);
    cudaAlloc(pool.color, capacity);
    cudaAlloc(pool.age, capacity);
    cudaAlloc(pool.lifetime, capacity);
    cudaAlloc(pool.systemId, capacity);
    cudaAlloc(pool.tileId, capacity);
    pool.activeIndices = nullptr;
    pool.capacity = capacity;
    pool.aliveCount = 0;
}

void releasePool(ParticlePool& pool)
{
    cudaFree(pool.pos);
    cudaFree(pool.vel);
    cudaFree(pool.color);
    cudaFree(pool.age);
    cudaFree(pool.lifetime);
    cudaFree(pool.systemId);
    cudaFree(pool.tileId);
    pool = {};
}

void allocatePackedPool(PackedPool& pool, uint32_t capacity)
{
    cudaAlloc(pool.posX, capacity);
    cudaAlloc(pool.posY, capacity);
    cudaAlloc(pool.posZ, capacity);
    cudaAlloc(pool.velX, capacity);
    cudaAlloc(pool.velY, capacity);
    cudaAlloc(pool.velZ, capacity);
    cudaAlloc(pool.age, capacity);
    cudaAlloc(pool.lifetime, capacity);
    cudaAlloc(pool.colorPacked, capacity);
    cudaAlloc(pool.size, capacity);
    cudaAlloc(pool.systemId, capacity);
    cudaAlloc(pool.tileId, capacity);
    pool.capacity = capacity;
}

void releasePackedPool(PackedPool& pool)
{
    cudaFree(pool.posX);
    cudaFree(pool.posY);
    cudaFree(pool.posZ);
    cudaFree(pool.velX);
    cudaFree(pool.velY);
    cudaFree(pool.velZ);
    cudaFree(pool.age);
    cudaFree(pool.lifetime);
    cudaFree(pool.colorPacked);
    cudaFree(pool.size);
    cudaFree(pool.systemId);
    cudaFree(pool.tileId);
    pool = {};
}

__device__ uint32_t mixBits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

__device__ float random01(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    const uint32_t mixed = mixBits(seed ^ (particle * 0x9E3779B9u) ^ (frame * 0x85EBCA6Bu) ^ lane);
    return static_cast<float>(mixed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ float randomSigned(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    return random01(seed, particle, frame, lane) * 2.0f - 1.0f;
}

// This state is intentionally device-resident. Kernels read the current active
// count and active-list pointer directly, so CPU submission never needs an
// exact lifecycle readback to choose the next frame's work.
struct DeviceFrameState {
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t activeCount = 0;
    uint32_t freeCount = 0;
    uint32_t deadCount = 0;
    uint32_t compactedCount = 0;
    uint32_t spawnStart = 0;
    uint32_t spawnCount = 0;
    uint32_t freeStart = 0;
    uint32_t requestedSpawn = 0;
    uint32_t frameIndex = 0;
    uint32_t activeBufferIndex = 0;
};

struct DeviceTelemetry {
    uint32_t aliveCount = 0;
    uint32_t requestedSpawn = 0;
    uint32_t spawned = 0;
    uint32_t dropped = 0;
    uint32_t deadCount = 0;
    uint32_t frameIndex = 0;
    uint32_t activeBufferIndex = 0;
    uint32_t tileAliveCounts[kMaxTiles] = {};
};

// Per-frame values are supplied through device memory so a captured graph can
// be replayed without mutating kernel-node arguments.
struct GraphParams {
    uint32_t requestedSpawn = 0;
    uint32_t frameIndex = 0;
    float simTime = 0.0f;
    float dt = 0.0f;
};

__global__ void initializeFreeListKernel(uint32_t* freeIndices, uint32_t capacity)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < capacity) {
        freeIndices[index] = index;
    }
}

__global__ void beginFrameFromParamsKernel(
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    const GraphParams* params)
{
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            state->deadCount = 0;
            state->compactedCount = 0;
            state->spawnStart = 0;
            state->spawnCount = 0;
            state->freeStart = 0;
            state->requestedSpawn = params->requestedSpawn;
            state->frameIndex = params->frameIndex;
        }
        if (threadIdx.x < kMaxTiles) {
            telemetry->tileAliveCounts[threadIdx.x] = 0;
        }
    }
}

struct SpawnCommand {
    EmitterDesc emitter = {};
    uint32_t systemId = 0;
    uint32_t requestStart = 0;
    uint32_t count = 0;
};

__device__ uint32_t findSpawnCommand(
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t spawnOffset)
{
    uint32_t first = 0;
    uint32_t last = commandCount;
    while (first + 1 < last) {
        const uint32_t middle = first + (last - first) / 2;
        if (commands[middle].requestStart <= spawnOffset) {
            first = middle;
        } else {
            last = middle;
        }
    }

    return first;
}

// ── Procedural Divergence-Free Curl Noise ─────────────────────────────
// Analytically divergence-free velocity field:
// v(x) = sum_i (k_i x a_i) * cos(k_i . x + omega * t + phi_i)
// Because k_i . (k_i x a_i) == 0, div(v) = 0 exactly.
__device__ float3 computeCurlNoise(
    float3 pos,
    float time,
    float freq,
    float strength)
{
    if (strength <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float3 p = make_float3(pos.x * freq, pos.y * freq, pos.z * freq);

    // 4 orthogonal/quasi-orthogonal wave directions and basis vectors
    // Wave 1: k1 = (1, 1, 0), a1 = (0, 0, 1) -> w1 = k1 x a1 = (1, -1, 0)
    const float phi1 = p.x + p.y + time;
    const float c1 = cosf(phi1);
    float vx = c1;
    float vy = -c1;
    float vz = 0.0f;

    // Wave 2: k2 = (0, 1, 1), a2 = (1, 0, 0) -> w2 = (0, 1, -1)
    const float phi2 = p.y + p.z + time * 1.3f + 1.57f;
    const float c2 = cosf(phi2);
    vy += c2;
    vz -= c2;

    // Wave 3: k3 = (1, 0, 1), a3 = (0, 1, 0) -> w3 = (-1, 0, 1)
    const float phi3 = p.x + p.z + time * 0.7f + 3.14f;
    const float c3 = cosf(phi3);
    vx -= c3;
    vz += c3;

    // Wave 4: k4 = (1, -1, 1), a4 = (1, 1, 0) -> w4 = (-1, 1, 2)
    const float phi4 = (p.x - p.y + p.z) * 0.707f + time * 1.1f + 0.78f;
    const float c4 = cosf(phi4) * 0.5f;
    vx -= c4;
    vy += c4;
    vz += c4 * 2.0f;

    return make_float3(vx * strength, vy * strength, vz * strength);
}

// ── Analytical Collision Handlers ─────────────────────────────────────
__device__ void resolvePlaneCollision(
    float4& pos,
    float4& vel,
    const CollisionPlane& plane)
{
    const float radius = pos.w;
    const float dotP = (pos.x - plane.point.x) * plane.normal.x +
                       (pos.y - plane.point.y) * plane.normal.y +
                       (pos.z - plane.point.z) * plane.normal.z;

    if (dotP < radius) {
        // Penetration correction
        const float penetration = radius - dotP;
        pos.x += plane.normal.x * penetration;
        pos.y += plane.normal.y * penetration;
        pos.z += plane.normal.z * penetration;

        // Normal velocity component
        const float vDotN = vel.x * plane.normal.x +
                            vel.y * plane.normal.y +
                            vel.z * plane.normal.z;
        if (vDotN < 0.0f) {
            // Moving into the plane: reflect normal component, damp tangential with friction
            const float vNormX = plane.normal.x * vDotN;
            const float vNormY = plane.normal.y * vDotN;
            const float vNormZ = plane.normal.z * vDotN;

            const float vTanX = vel.x - vNormX;
            const float vTanY = vel.y - vNormY;
            const float vTanZ = vel.z - vNormZ;

            const float frictionFactor = fmaxf(0.0f, 1.0f - plane.friction);
            vel.x = vTanX * frictionFactor - vNormX * plane.bounce;
            vel.y = vTanY * frictionFactor - vNormY * plane.bounce;
            vel.z = vTanZ * frictionFactor - vNormZ * plane.bounce;
        }
    }
}

__device__ void resolveSphereCollision(
    float4& pos,
    float4& vel,
    const CollisionSphere& sphere)
{
    const float radius = pos.w;
    const float dx = pos.x - sphere.center.x;
    const float dy = pos.y - sphere.center.y;
    const float dz = pos.z - sphere.center.z;
    const float distSqr = dx * dx + dy * dy + dz * dz;
    const float dist = sqrtf(fmaxf(1.0e-7f, distSqr));

    if (!sphere.invert) {
        // Exterior obstacle sphere: particles bounce off the outside
        const float targetDist = sphere.radius + radius;
        if (dist < targetDist) {
            const float invDist = 1.0f / dist;
            const float nx = dx * invDist;
            const float ny = dy * invDist;
            const float nz = dz * invDist;

            pos.x = sphere.center.x + nx * targetDist;
            pos.y = sphere.center.y + ny * targetDist;
            pos.z = sphere.center.z + nz * targetDist;

            const float vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
            if (vDotN < 0.0f) {
                const float vNormX = nx * vDotN;
                const float vNormY = ny * vDotN;
                const float vNormZ = nz * vDotN;

                const float frictionFactor = fmaxf(0.0f, 1.0f - sphere.friction);
                vel.x = (vel.x - vNormX) * frictionFactor - vNormX * sphere.bounce;
                vel.y = (vel.y - vNormY) * frictionFactor - vNormY * sphere.bounce;
                vel.z = (vel.z - vNormZ) * frictionFactor - vNormZ * sphere.bounce;
            }
        }
    } else {
        // Inverted containment sphere: particles are trapped inside
        const float targetDist = fmaxf(0.001f, sphere.radius - radius);
        if (dist > targetDist) {
            const float invDist = 1.0f / dist;
            // Normal points inwards towards center
            const float nx = -dx * invDist;
            const float ny = -dy * invDist;
            const float nz = -dz * invDist;

            pos.x = sphere.center.x - nx * targetDist;
            pos.y = sphere.center.y - ny * targetDist;
            pos.z = sphere.center.z - nz * targetDist;

            const float vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
            if (vDotN < 0.0f) {
                const float vNormX = nx * vDotN;
                const float vNormY = ny * vDotN;
                const float vNormZ = nz * vDotN;

                const float frictionFactor = fmaxf(0.0f, 1.0f - sphere.friction);
                vel.x = (vel.x - vNormX) * frictionFactor - vNormX * sphere.bounce;
                vel.y = (vel.y - vNormY) * frictionFactor - vNormY * sphere.bounce;
                vel.z = (vel.z - vNormZ) * frictionFactor - vNormZ * sphere.bounce;
            }
        }
    }
}

__device__ void resolveBoxCollision(
    float4& pos,
    float4& vel,
    const CollisionBox& box)
{
    const float radius = pos.w;
    const float frictionFactor = fmaxf(0.0f, 1.0f - box.friction);

    // X axis boundaries
    if (pos.x < box.minBounds.x + radius) {
        pos.x = box.minBounds.x + radius;
        if (vel.x < 0.0f) {
            vel.x = -vel.x * box.bounce;
            vel.y *= frictionFactor;
            vel.z *= frictionFactor;
        }
    } else if (pos.x > box.maxBounds.x - radius) {
        pos.x = box.maxBounds.x - radius;
        if (vel.x > 0.0f) {
            vel.x = -vel.x * box.bounce;
            vel.y *= frictionFactor;
            vel.z *= frictionFactor;
        }
    }

    // Y axis boundaries
    if (pos.y < box.minBounds.y + radius) {
        pos.y = box.minBounds.y + radius;
        if (vel.y < 0.0f) {
            vel.y = -vel.y * box.bounce;
            vel.x *= frictionFactor;
            vel.z *= frictionFactor;
        }
    } else if (pos.y > box.maxBounds.y - radius) {
        pos.y = box.maxBounds.y - radius;
        if (vel.y > 0.0f) {
            vel.y = -vel.y * box.bounce;
            vel.x *= frictionFactor;
            vel.z *= frictionFactor;
        }
    }

    // Z axis boundaries
    if (pos.z < box.minBounds.z + radius) {
        pos.z = box.minBounds.z + radius;
        if (vel.z < 0.0f) {
            vel.z = -vel.z * box.bounce;
            vel.x *= frictionFactor;
            vel.y *= frictionFactor;
        }
    } else if (pos.z > box.maxBounds.z - radius) {
        pos.z = box.maxBounds.z - radius;
        if (vel.z > 0.0f) {
            vel.z = -vel.z * box.bounce;
            vel.x *= frictionFactor;
            vel.y *= frictionFactor;
        }
    }
}

// ── Packed Format Encode/Decode Helpers ───────────────────────────────

__device__ __forceinline__ int16_t encodePosition(float worldPos, float tileOrigin, float scale)
{
    float local = (worldPos - tileOrigin) * scale;
    local = fminf(32767.0f, fmaxf(-32768.0f, local));
    return static_cast<int16_t>(__float2int_rn(local));
}

__device__ __forceinline__ float decodePosition(int16_t packed, float tileOrigin, float invScale)
{
    return tileOrigin + static_cast<float>(packed) * invScale;
}

__device__ __forceinline__ uint32_t encodeColorRGBA8(float r, float g, float b, float a)
{
    uint32_t ri = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, r)) * 255.0f));
    uint32_t gi = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, g)) * 255.0f));
    uint32_t bi = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, b)) * 255.0f));
    uint32_t ai = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, a)) * 255.0f));
    return ri | (gi << 8) | (bi << 16) | (ai << 24);
}

__device__ __forceinline__ float4 decodeColorRGBA8(uint32_t packed)
{
    float r = static_cast<float>(packed & 0xFFu) * (1.0f / 255.0f);
    float g = static_cast<float>((packed >> 8) & 0xFFu) * (1.0f / 255.0f);
    float b = static_cast<float>((packed >> 16) & 0xFFu) * (1.0f / 255.0f);
    float a = static_cast<float>((packed >> 24) & 0xFFu) * (1.0f / 255.0f);
    return make_float4(r, g, b, a);
}

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

// CUB's block scan lets each tile reserve one contiguous output range. This
// avoids a CPU readback and avoids one global atomic per surviving particle.
__global__ void compactActiveIndicesKernel(
    DeviceFrameState* state,
    const uint8_t* aliveFlags)
{
    if (state->deadCount == 0) {
        return;
    }

    using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ uint32_t outputStart;

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    uint32_t* compactedIndices = state->scratchActiveIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t tileStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x;
         tileStart < count;
         tileStart += stride) {
        const uint64_t activeIndex = tileStart + threadIdx.x;
        const bool inRange = activeIndex < count;
        const uint32_t isAlive = inRange ? static_cast<uint32_t>(aliveFlags[activeIndex]) : 0u;
        uint32_t localOffset = 0;
        uint32_t selectedInTile = 0;
        BlockScan(scanStorage).ExclusiveSum(isAlive, localOffset, selectedInTile);

        if (threadIdx.x == 0) {
            outputStart = selectedInTile == 0
                ? 0u
                : atomicAdd(&state->compactedCount, selectedInTile);
        }
        __syncthreads();

        if (isAlive != 0) {
            compactedIndices[outputStart + localOffset] = activeIndices[activeIndex];
        }
        __syncthreads();
    }
}

__global__ void finalizeCompactionKernel(DeviceFrameState* state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && state->deadCount != 0) {
        uint32_t* oldActiveIndices = state->activeIndices;
        state->activeIndices = state->scratchActiveIndices;
        state->scratchActiveIndices = oldActiveIndices;
        state->activeCount = state->compactedCount;
        state->activeBufferIndex ^= 1u;
    }
}

__global__ void prepareSpawnKernel(DeviceFrameState* state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t availableSlots = state->freeCount;
        const uint32_t spawned = min(state->requestedSpawn, availableSlots);

        state->spawnStart = state->activeCount;
        state->spawnCount = spawned;
        state->freeStart = availableSlots;
        state->freeCount = availableSlots - spawned;
        state->activeCount += spawned;
    }
}

__global__ void spawnBatchKernel(
    ParticlePool pool,
    const DeviceFrameState* state,
    const uint32_t* freeIndices,
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t seed)
{
    const uint32_t totalSpawn = state->spawnCount;
    const uint32_t firstActiveIndex = state->spawnStart;
    const uint32_t firstFreeSlot = state->freeStart;
    const uint32_t frameIndex = state->frameIndex;
    uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t spawnOffset = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         spawnOffset < totalSpawn;
         spawnOffset += stride) {
        const uint32_t compactOffset = static_cast<uint32_t>(spawnOffset);
        const uint32_t activeIndex = firstActiveIndex + compactOffset;
        const uint32_t commandIndex = findSpawnCommand(commands, commandCount, compactOffset);
        const SpawnCommand command = commands[commandIndex];
        const EmitterDesc emitter = command.emitter;
        const uint32_t particleIndex = freeIndices[firstFreeSlot - 1u - compactOffset];
        activeIndices[activeIndex] = particleIndex;

        const uint32_t randomIndex = activeIndex;
        const float vx = emitter.velocity.x + emitter.velocityVariance.x * randomSigned(seed, randomIndex, frameIndex, 1u);
        const float vy = emitter.velocity.y + emitter.velocityVariance.y * randomSigned(seed, randomIndex, frameIndex, 2u);
        const float vz = emitter.velocity.z + emitter.velocityVariance.z * randomSigned(seed, randomIndex, frameIndex, 3u);
        const float lifetimeJitter = emitter.lifetimeVariance * randomSigned(seed, randomIndex, frameIndex, 4u);
        const float particleLifetime = fmaxf(0.001f, emitter.lifetime * (1.0f + lifetimeJitter));
        const float sizeJitter = emitter.sizeVariance * randomSigned(seed, randomIndex, frameIndex, 5u);
        const float particleSize = fmaxf(0.001f, emitter.size * (1.0f + sizeJitter));

        pool.pos[particleIndex] = make_float4(emitter.position.x, emitter.position.y, emitter.position.z, particleSize);
        pool.vel[particleIndex] = make_float4(vx, vy, vz, 0.0f);
        pool.color[particleIndex] = make_float4(emitter.color.r, emitter.color.g, emitter.color.b, emitter.color.a);
        pool.age[particleIndex] = 0.0f;
        pool.lifetime[particleIndex] = particleLifetime;
        pool.systemId[particleIndex] = emitter.recipeId;
        pool.tileId[particleIndex] = emitter.tileId;
    }
}

__global__ void spawnBatchPackedKernel(
    PackedPool pool,
    const DeviceFrameState* state,
    const uint32_t* freeIndices,
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t seed,
    float posScale)
{
    const uint32_t totalSpawn = state->spawnCount;
    const uint32_t firstActiveIndex = state->spawnStart;
    const uint32_t firstFreeSlot = state->freeStart;
    const uint32_t frameIndex = state->frameIndex;
    uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t spawnOffset = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         spawnOffset < totalSpawn;
         spawnOffset += stride) {
        const uint32_t compactOffset = static_cast<uint32_t>(spawnOffset);
        const uint32_t activeIndex = firstActiveIndex + compactOffset;
        const uint32_t commandIndex = findSpawnCommand(commands, commandCount, compactOffset);
        const SpawnCommand command = commands[commandIndex];
        const EmitterDesc emitter = command.emitter;
        const uint32_t particleIndex = freeIndices[firstFreeSlot - 1u - compactOffset];
        activeIndices[activeIndex] = particleIndex;

        const uint32_t randomIndex = activeIndex;
        const float vx = emitter.velocity.x + emitter.velocityVariance.x * randomSigned(seed, randomIndex, frameIndex, 1u);
        const float vy = emitter.velocity.y + emitter.velocityVariance.y * randomSigned(seed, randomIndex, frameIndex, 2u);
        const float vz = emitter.velocity.z + emitter.velocityVariance.z * randomSigned(seed, randomIndex, frameIndex, 3u);
        const float lifetimeJitter = emitter.lifetimeVariance * randomSigned(seed, randomIndex, frameIndex, 4u);
        const float particleLifetime = fmaxf(0.001f, emitter.lifetime * (1.0f + lifetimeJitter));
        const float sizeJitter = emitter.sizeVariance * randomSigned(seed, randomIndex, frameIndex, 5u);
        const float particleSize = fmaxf(0.001f, emitter.size * (1.0f + sizeJitter));

        const uint16_t tileId = emitter.tileId;
        pool.tileId[particleIndex] = tileId;
        const float originX = cTiles[tileId].originX;
        const float originY = cTiles[tileId].originY;
        const float originZ = cTiles[tileId].originZ;

        pool.posX[particleIndex] = encodePosition(emitter.position.x, originX, posScale);
        pool.posY[particleIndex] = encodePosition(emitter.position.y, originY, posScale);
        pool.posZ[particleIndex] = encodePosition(emitter.position.z, originZ, posScale);
        pool.velX[particleIndex] = __float2half(vx);
        pool.velY[particleIndex] = __float2half(vy);
        pool.velZ[particleIndex] = __float2half(vz);
        pool.age[particleIndex] = 0;
        pool.lifetime[particleIndex] = __float2half(particleLifetime);
        pool.colorPacked[particleIndex] = encodeColorRGBA8(emitter.color.r, emitter.color.g, emitter.color.b, emitter.color.a);
        pool.size[particleIndex] = __float2half(particleSize);
        pool.systemId[particleIndex] = emitter.recipeId;
    }
}

__global__ void migrateTilesPackedKernel(
    PackedPool pool,
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount,
    float posScale,
    float invPosScale)
{
    __shared__ uint32_t sTileCounts[kMaxTiles];
    if (threadIdx.x < kMaxTiles) {
        sTileCounts[threadIdx.x] = 0;
    }
    __syncthreads();

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];
        const uint16_t currentTile = pool.tileId[particleIndex];

        const float currentOriginX = cTiles[currentTile].originX;
        const float currentOriginY = cTiles[currentTile].originY;
        const float currentOriginZ = cTiles[currentTile].originZ;

        const float worldX = decodePosition(pool.posX[particleIndex], currentOriginX, invPosScale);
        const float worldY = decodePosition(pool.posY[particleIndex], currentOriginY, invPosScale);
        const float worldZ = decodePosition(pool.posZ[particleIndex], currentOriginZ, invPosScale);

        uint16_t bestTile = currentTile;
        float bestDistSq = (worldX - currentOriginX) * (worldX - currentOriginX) +
                           (worldY - currentOriginY) * (worldY - currentOriginY) +
                           (worldZ - currentOriginZ) * (worldZ - currentOriginZ);

        for (uint32_t t = 0; t < tileCount; ++t) {
            if (t == currentTile) continue;
            const float dx = worldX - cTiles[t].originX;
            const float dy = worldY - cTiles[t].originY;
            const float dz = worldZ - cTiles[t].originZ;
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestTile = static_cast<uint16_t>(t);
            }
        }

        if (bestTile != currentTile) {
            pool.tileId[particleIndex] = bestTile;
            pool.posX[particleIndex] = encodePosition(worldX, cTiles[bestTile].originX, posScale);
            pool.posY[particleIndex] = encodePosition(worldY, cTiles[bestTile].originY, posScale);
            pool.posZ[particleIndex] = encodePosition(worldZ, cTiles[bestTile].originZ, posScale);
        }

        atomicAdd(&sTileCounts[bestTile], 1u);
    }
    __syncthreads();

    if (threadIdx.x < tileCount) {
        const uint32_t countInTile = sTileCounts[threadIdx.x];
        if (countInTile > 0) {
            atomicAdd(&telemetry->tileAliveCounts[threadIdx.x], countInTile);
        }
    }
}

__global__ void migrateTilesKernel(
    ParticlePool pool,
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount)
{
    __shared__ uint32_t sTileCounts[kMaxTiles];
    if (threadIdx.x < kMaxTiles) {
        sTileCounts[threadIdx.x] = 0;
    }
    __syncthreads();

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];
        const uint16_t currentTile = pool.tileId[particleIndex];

        const float worldX = pool.pos[particleIndex].x;
        const float worldY = pool.pos[particleIndex].y;
        const float worldZ = pool.pos[particleIndex].z;

        const float currentOriginX = cTiles[currentTile].originX;
        const float currentOriginY = cTiles[currentTile].originY;
        const float currentOriginZ = cTiles[currentTile].originZ;

        uint16_t bestTile = currentTile;
        float bestDistSq = (worldX - currentOriginX) * (worldX - currentOriginX) +
                           (worldY - currentOriginY) * (worldY - currentOriginY) +
                           (worldZ - currentOriginZ) * (worldZ - currentOriginZ);

        for (uint32_t t = 0; t < tileCount; ++t) {
            if (t == currentTile) continue;
            const float dx = worldX - cTiles[t].originX;
            const float dy = worldY - cTiles[t].originY;
            const float dz = worldZ - cTiles[t].originZ;
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestTile = static_cast<uint16_t>(t);
            }
        }

        if (bestTile != currentTile) {
            pool.tileId[particleIndex] = bestTile;
        }

        atomicAdd(&sTileCounts[bestTile], 1u);
    }
    __syncthreads();

    if (threadIdx.x < tileCount) {
        const uint32_t countInTile = sTileCounts[threadIdx.x];
        if (countInTile > 0) {
            atomicAdd(&telemetry->tileAliveCounts[threadIdx.x], countInTile);
        }
    }
}

__global__ void snapshotTelemetryKernel(
    const DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        telemetry->aliveCount = state->activeCount;
        telemetry->requestedSpawn = state->requestedSpawn;
        telemetry->spawned = state->spawnCount;
        telemetry->dropped = state->requestedSpawn - state->spawnCount;
        telemetry->deadCount = state->deadCount;
        telemetry->frameIndex = state->frameIndex;
        telemetry->activeBufferIndex = state->activeBufferIndex;
        if (tileCount <= 1) {
            telemetry->tileAliveCounts[0] = state->activeCount;
        }
    }
}

struct EmitterState {
    EmitterDesc desc = {};
    float spawnCarry = 0.0f;
    uint32_t pendingBurst = 0;
};

} // namespace

struct ParticleSystem::Impl {
    struct TelemetrySlot {
        DeviceTelemetry* hostTelemetry = nullptr;
        cudaEvent_t frameStart = nullptr;
        cudaEvent_t afterSimulate = nullptr;
        cudaEvent_t afterCompact = nullptr;
        cudaEvent_t afterSpawn = nullptr;
        cudaEvent_t ready = nullptr;
        uint64_t spawnUpperBoundThroughFrame = 0;
        uint32_t frameIndex = 0;
        bool inFlight = false;
        bool graphActive = false;
    };

    struct SpawnUploadSlot {
        SpawnCommand* hostCommands = nullptr;
        uint32_t capacity = 0;
        cudaEvent_t ready = nullptr;
        bool inFlight = false;
    };

    struct GraphKey {
        uint32_t simulationBlocks = 0;
        uint32_t spawnBlocks = 0;
        uint32_t migrateBlocks = 0;
        uint32_t commandCount = 0;
        uint32_t recipeCount = 0;
        uint32_t tileCount = 0;
        bool valid = false;

        bool matches(const GraphKey& other) const
        {
            return valid && other.valid &&
                simulationBlocks == other.simulationBlocks &&
                spawnBlocks == other.spawnBlocks &&
                migrateBlocks == other.migrateBlocks &&
                commandCount == other.commandCount &&
                recipeCount == other.recipeCount &&
                tileCount == other.tileCount;
        }
    };

    explicit Impl(uint32_t capacity, StorageMode mode, bool enableCudaGraphs)
    {
        if (capacity == 0) {
            throw std::invalid_argument("ParticleSystem capacity must be greater than zero");
        }

        mode_ = mode;
        if (mode == StorageMode::Packed) {
            allocatePackedPool(packedPool_, capacity);
            // FP32 pool still needed for capacity tracking and telemetry.
            pool.capacity = capacity;
            pool.aliveCount = 0;
        } else {
            allocatePool(pool, capacity);
        }
        cudaAlloc(activeIndices, capacity);
        cudaAlloc(scratchActiveIndices, capacity);
        cudaAlloc(freeIndices, capacity);
        cudaAlloc(aliveFlags, capacity);
        cudaAlloc(deviceState, 1);
        cudaAlloc(deviceTelemetryScratch, 1);
        cudaAlloc(deviceGraphParams, 1);
        pool.activeIndices = activeIndices;

        VP_CUDA_CHECK(cudaStreamCreate(&stream));
        createTelemetrySlots();
        createSpawnUploadSlots();
        VP_CUDA_CHECK(cudaEventCreate(&latestFrameStart));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterSimulate));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterCompact));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterSpawn));

        int device = 0;
        cudaDeviceProp properties = {};
        VP_CUDA_CHECK(cudaGetDevice(&device));
        VP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        maxDispatchBlocks = std::max(
            1u,
            static_cast<uint32_t>(properties.multiProcessorCount) *
                kTargetBlocksPerMultiprocessor);

        if (mode != StorageMode::Packed) {
            gpuPool.pos = pool.pos;
            gpuPool.vel = pool.vel;
            gpuPool.color = pool.color;
            gpuPool.age = pool.age;
            gpuPool.lifetime = pool.lifetime;
            gpuPool.systemId = pool.systemId;
            gpuPool.tileId = pool.tileId;
            gpuPool.activeIndices = reinterpret_cast<uint32_t* const*>(
                reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeIndices));
            gpuPool.aliveCount = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeCount));
        }
        gpuPool.capacity = capacity;

        settings.tileCount = 1;
        settings.recipeCount = 1;
        settings.recipes[0].gravity = settings.gravity;
        settings.recipes[0].wind = settings.wind;
        settings.recipes[0].drag = settings.drag;
        settings.recipes[0].turbulence = settings.turbulence;
        settings.recipes[0].plane = settings.plane;
        settings.recipes[0].sphere = settings.sphere;
        settings.recipes[0].box = settings.box;
        settings.recipes[0].curves = settings.curves;
        VP_CUDA_CHECK(cudaMemcpyToSymbol(cTiles, settings.tiles, sizeof(TileDesc) * kMaxTiles));
        VP_CUDA_CHECK(cudaMemcpyToSymbol(cRecipes, settings.recipes, sizeof(EffectRecipe) * kMaxRecipes));

        stats.capacity = capacity;
        graphsEnabled = enableCudaGraphs;
        initializeFreeList();
    }

    ~Impl()
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }

        destroyTelemetrySlots();
        destroySpawnUploadSlots();
        if (graphExec != nullptr) {
            cudaGraphExecDestroy(graphExec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        cudaEventDestroy(latestAfterSpawn);
        cudaEventDestroy(latestAfterCompact);
        cudaEventDestroy(latestAfterSimulate);
        cudaEventDestroy(latestFrameStart);

        cudaFree(deviceTelemetryScratch);
        cudaFree(deviceGraphParams);
        cudaFree(deviceState);
        cudaFree(spawnCommands);
        cudaFree(aliveFlags);
        cudaFree(freeIndices);
        cudaFree(scratchActiveIndices);
        cudaFree(activeIndices);
        releasePool(pool);
        releasePackedPool(packedPool_);

        cudaStreamDestroy(stream);
    }

    void createTelemetrySlots()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            VP_CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&slot.hostTelemetry),
                sizeof(DeviceTelemetry),
                cudaHostAllocPortable));
            VP_CUDA_CHECK(cudaEventCreate(&slot.frameStart));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterSimulate));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterCompact));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterSpawn));
            VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        }
    }

    void destroyTelemetrySlots()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            cudaEventDestroy(slot.ready);
            cudaEventDestroy(slot.afterSpawn);
            cudaEventDestroy(slot.afterCompact);
            cudaEventDestroy(slot.afterSimulate);
            cudaEventDestroy(slot.frameStart);
            cudaFreeHost(slot.hostTelemetry);
            slot = {};
        }
    }

    void createSpawnUploadSlots()
    {
        spawnUploadSlots.resize(kSpawnUploadRingSize);
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        }
    }

    void destroySpawnUploadSlots()
    {
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (slot.ready != nullptr) {
                cudaEventDestroy(slot.ready);
            }
            if (slot.hostCommands != nullptr) {
                cudaFreeHost(slot.hostCommands);
            }
            slot = {};
        }
        spawnUploadSlots.clear();
    }

    void initializeFreeList()
    {
        DeviceFrameState initialState = {};
        initialState.activeIndices = activeIndices;
        initialState.scratchActiveIndices = scratchActiveIndices;
        initialState.activeCount = 0;
        initialState.freeCount = pool.capacity;
        initialState.activeBufferIndex = 0;

        VP_CUDA_CHECK(cudaMemcpyAsync(
            deviceState,
            &initialState,
            sizeof(initialState),
            cudaMemcpyHostToDevice,
            stream));
        initializeFreeListKernel<<<blockCount(pool.capacity), kThreadsPerBlock, 0, stream>>>(
            freeIndices,
            pool.capacity);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaStreamSynchronize(stream));

        pool.aliveCount = 0;
        pool.activeIndices = activeIndices;
        activeWorkUpperBound = 0;
        submittedSpawnUpperBoundTotal = 0;
        nextTelemetrySlot = 0;
        hasSubmittedFrame = false;
        for (TelemetrySlot& slot : telemetrySlots) {
            slot.inFlight = false;
        }
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            slot.inFlight = false;
        }
    }

    void invalidateGraph()
    {
        if (graphExec != nullptr) {
            VP_CUDA_CHECK(cudaGraphExecDestroy(graphExec));
            graphExec = nullptr;
        }
        if (graph != nullptr) {
            VP_CUDA_CHECK(cudaGraphDestroy(graph));
            graph = nullptr;
        }
        graphSpawnMemcpyNode = nullptr;
        graphSpawnMemcpyParams = {};
        currentGraphKey = {};
    }

    void rebuildGraph(const GraphKey& key, const SpawnUploadSlot* uploadSlot)
    {
        invalidateGraph();

        VP_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        beginFrameFromParamsKernel<<<1, 256, 0, stream>>>(
            deviceState,
            deviceTelemetryScratch,
            deviceGraphParams);
        VP_CUDA_CHECK(cudaGetLastError());

        if (mode_ == StorageMode::Packed) {
            const float posScale = settings.posQuantizationScale;
            const float invPosScale = 1.0f / posScale;
            simulatePackedKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
                packedPool_, deviceState, aliveFlags, freeIndices,
                deviceGraphParams, posScale, invPosScale,
                key.recipeCount, key.tileCount);
        } else {
            simulateKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
                pool, deviceState, aliveFlags, freeIndices,
                deviceGraphParams, key.recipeCount, key.tileCount);
        }
        VP_CUDA_CHECK(cudaGetLastError());

        compactActiveIndicesKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
            deviceState, aliveFlags);
        VP_CUDA_CHECK(cudaGetLastError());
        finalizeCompactionKernel<<<1, 1, 0, stream>>>(deviceState);
        VP_CUDA_CHECK(cudaGetLastError());

        if (key.commandCount != 0) {
            VP_CUDA_CHECK(cudaMemcpyAsync(
                spawnCommands,
                uploadSlot->hostCommands,
                sizeof(SpawnCommand) * key.commandCount,
                cudaMemcpyHostToDevice,
                stream));
        }

        prepareSpawnKernel<<<1, 1, 0, stream>>>(deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        if (key.commandCount != 0) {
            if (mode_ == StorageMode::Packed) {
                const float posScale = settings.posQuantizationScale;
                spawnBatchPackedKernel<<<key.spawnBlocks, kThreadsPerBlock, 0, stream>>>(
                    packedPool_, deviceState, freeIndices, spawnCommands,
                    key.commandCount, settings.seed, posScale);
            } else {
                spawnBatchKernel<<<key.spawnBlocks, kThreadsPerBlock, 0, stream>>>(
                    pool, deviceState, freeIndices, spawnCommands,
                    key.commandCount, settings.seed);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }

        if (settings.tileCount > 1) {
            if (mode_ == StorageMode::Packed) {
                const float posScale = settings.posQuantizationScale;
                const float invPosScale = 1.0f / posScale;
                migrateTilesPackedKernel<<<key.migrateBlocks, kThreadsPerBlock, 0, stream>>>(
                    packedPool_, deviceState, deviceTelemetryScratch, settings.tileCount,
                    posScale, invPosScale);
            } else {
                migrateTilesKernel<<<key.migrateBlocks, kThreadsPerBlock, 0, stream>>>(
                    pool, deviceState, deviceTelemetryScratch, settings.tileCount);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }

        VP_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        VP_CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph));

        if (key.commandCount != 0) {
            size_t nodeCount = 0;
            VP_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodeCount));
            std::vector<cudaGraphNode_t> nodes(nodeCount);
            VP_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeKernel;
                VP_CUDA_CHECK(cudaGraphNodeGetType(node, &nodeType));
                if (nodeType == cudaGraphNodeTypeMemcpy) {
                    graphSpawnMemcpyNode = node;
                    VP_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &graphSpawnMemcpyParams));
                    break;
                }
            }
            if (graphSpawnMemcpyNode == nullptr) {
                throw std::runtime_error("CUDA Graph capture did not produce the spawn upload node");
            }
        }

        currentGraphKey = key;
        ++graphRebuildCount;
    }

    void setGraphSpawnUploadSource(const SpawnUploadSlot& uploadSlot)
    {
        if (graphSpawnMemcpyNode == nullptr) {
            return;
        }
        cudaMemcpy3DParms params = graphSpawnMemcpyParams;
        params.srcPtr.ptr = uploadSlot.hostCommands;
        VP_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams(
            graphExec, graphSpawnMemcpyNode, &params));
    }

    uint32_t dispatchBlockCount(uint32_t expectedWorkItems) const
    {
        const uint32_t requestedBlocks = std::max(1u, blockCount(expectedWorkItems));
        return std::min(requestedBlocks, maxDispatchBlocks);
    }

    void ensureSpawnCommandStorage(uint32_t commandCount)
    {
        if (commandCount <= spawnCommandCapacity) {
            return;
        }

        VP_CUDA_CHECK(cudaFree(spawnCommands));
        spawnCommands = nullptr;
        cudaAlloc(spawnCommands, commandCount);
        spawnCommandCapacity = commandCount;
    }

    void pollSpawnUploadSlots()
    {
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (!slot.inFlight) {
                continue;
            }

            const cudaError_t queryResult = cudaEventQuery(slot.ready);
            if (queryResult == cudaErrorNotReady) {
                continue;
            }
            VP_CUDA_CHECK(queryResult);
            slot.inFlight = false;
        }
    }

    void ensureSpawnUploadCapacity(SpawnUploadSlot& slot, uint32_t commandCount)
    {
        if (commandCount <= slot.capacity) {
            return;
        }

        if (slot.hostCommands != nullptr) {
            VP_CUDA_CHECK(cudaFreeHost(slot.hostCommands));
        }
        slot.hostCommands = nullptr;
        VP_CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.hostCommands),
            sizeof(SpawnCommand) * commandCount,
            cudaHostAllocPortable));
        slot.capacity = commandCount;
    }

    SpawnUploadSlot& acquireSpawnUploadSlot(uint32_t commandCount)
    {
        pollSpawnUploadSlots();
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (!slot.inFlight) {
                ensureSpawnUploadCapacity(slot, commandCount);
                return slot;
            }
        }

        SpawnUploadSlot slot = {};
        VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        spawnUploadSlots.push_back(slot);
        ensureSpawnUploadCapacity(spawnUploadSlots.back(), commandCount);
        return spawnUploadSlots.back();
    }

    TelemetrySlot* acquireTelemetrySlot()
    {
        for (size_t offset = 0; offset < telemetrySlots.size(); ++offset) {
            const size_t index = (nextTelemetrySlot + offset) % telemetrySlots.size();
            TelemetrySlot& slot = telemetrySlots[index];
            if (!slot.inFlight) {
                nextTelemetrySlot = (index + 1) % telemetrySlots.size();
                return &slot;
            }
        }

        return nullptr;
    }

    void publishTelemetry(
        const DeviceTelemetry& telemetry,
        uint32_t latencyFrames,
        float spawnMs,
        float simulateMs,
        float compactMs,
        float totalMs,
        bool graphActive)
    {
        SimulationStats nextStats = {};
        nextStats.frameIndex = telemetry.frameIndex;
        nextStats.telemetryLatencyFrames = latencyFrames;
        nextStats.valid = true;
        nextStats.graphActive = graphActive;
        nextStats.graphRebuildCount = graphRebuildCount;
        nextStats.capacity = pool.capacity;
        nextStats.aliveCount = telemetry.aliveCount;
        nextStats.requestedSpawn = telemetry.requestedSpawn;
        nextStats.spawned = telemetry.spawned;
        nextStats.dropped = telemetry.dropped;
        nextStats.deadCount = telemetry.deadCount;
        nextStats.spawnMs = spawnMs;
        nextStats.simulateMs = simulateMs;
        nextStats.compactMs = compactMs;
        nextStats.totalMs = totalMs;

        stats = nextStats;
        pool.aliveCount = telemetry.aliveCount;
        pool.activeIndices = telemetry.activeBufferIndex == 0
            ? activeIndices
            : scratchActiveIndices;

        tileStats.resize(settings.tileCount);
        for (uint32_t t = 0; t < settings.tileCount; ++t) {
            tileStats[t].tileId = t;
            tileStats[t].aliveCount = telemetry.tileAliveCounts[t];
            tileStats[t].origin = {
                settings.tiles[t].originX,
                settings.tiles[t].originY,
                settings.tiles[t].originZ
            };
        }
    }

    void tightenActiveWorkBound(const TelemetrySlot& slot, const DeviceTelemetry& telemetry)
    {
        if (submittedSpawnUpperBoundTotal < slot.spawnUpperBoundThroughFrame) {
            return;
        }

        const uint64_t potentialSpawnsSinceSample =
            submittedSpawnUpperBoundTotal - slot.spawnUpperBoundThroughFrame;
        const uint64_t refreshedBound = std::min<uint64_t>(
            pool.capacity,
            static_cast<uint64_t>(telemetry.aliveCount) + potentialSpawnsSinceSample);
        activeWorkUpperBound = std::min(
            activeWorkUpperBound,
            static_cast<uint32_t>(refreshedBound));
    }

    void pollTelemetry()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            if (!slot.inFlight) {
                continue;
            }

            const cudaError_t queryResult = cudaEventQuery(slot.ready);
            if (queryResult == cudaErrorNotReady) {
                continue;
            }
            VP_CUDA_CHECK(queryResult);

            const DeviceTelemetry telemetry = *slot.hostTelemetry;
            float spawnMs = 0.0f;
            float simulateMs = 0.0f;
            float compactMs = 0.0f;
            float totalMs = 0.0f;
            VP_CUDA_CHECK(cudaEventElapsedTime(&spawnMs, slot.afterCompact, slot.afterSpawn));
            VP_CUDA_CHECK(cudaEventElapsedTime(&simulateMs, slot.frameStart, slot.afterSimulate));
            VP_CUDA_CHECK(cudaEventElapsedTime(&compactMs, slot.afterSimulate, slot.afterCompact));
            VP_CUDA_CHECK(cudaEventElapsedTime(&totalMs, slot.frameStart, slot.afterSpawn));

            if (slot.graphActive) {
                // A graph launch has no externally-addressable internal timing
                // events. Publish its complete pipeline duration as simulateMs.
                simulateMs = totalMs;
                compactMs = 0.0f;
                spawnMs = 0.0f;
            }

            const uint64_t submittedFrames = frameIndex;
            const uint32_t latencyFrames = submittedFrames > slot.frameIndex
                ? static_cast<uint32_t>(std::min<uint64_t>(
                      submittedFrames - slot.frameIndex - 1u,
                      std::numeric_limits<uint32_t>::max()))
                : 0u;
            publishTelemetry(
                telemetry, latencyFrames, spawnMs, simulateMs, compactMs, totalMs,
                slot.graphActive);
            tightenActiveWorkBound(slot, telemetry);
            slot.inFlight = false;
        }
    }

    void queueTelemetry(TelemetrySlot& slot)
    {
        snapshotTelemetryKernel<<<1, 1, 0, stream>>>(deviceState, deviceTelemetryScratch, settings.tileCount);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaMemcpyAsync(
            slot.hostTelemetry,
            deviceTelemetryScratch,
            sizeof(DeviceTelemetry),
            cudaMemcpyDeviceToHost,
            stream));
        VP_CUDA_CHECK(cudaEventRecord(slot.ready, stream));
        slot.frameIndex = frameIndex;
        slot.spawnUpperBoundThroughFrame = submittedSpawnUpperBoundTotal;
        slot.inFlight = true;
    }

    void advanceActiveWorkUpperBound(uint32_t potentialSpawnCount)
    {
        activeWorkUpperBound = static_cast<uint32_t>(std::min<uint64_t>(
            pool.capacity,
            static_cast<uint64_t>(activeWorkUpperBound) + potentialSpawnCount));

        const uint64_t maxValue = std::numeric_limits<uint64_t>::max();
        submittedSpawnUpperBoundTotal = maxValue - submittedSpawnUpperBoundTotal < potentialSpawnCount
            ? maxValue
            : submittedSpawnUpperBoundTotal + potentialSpawnCount;
    }

    ParticlePool pool = {};
    PackedPool packedPool_ = {};
    StorageMode mode_ = StorageMode::FP32;
    GpuParticlePool gpuPool = {};
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t* freeIndices = nullptr;
    uint8_t* aliveFlags = nullptr;
    SpawnCommand* spawnCommands = nullptr;
    uint32_t spawnCommandCapacity = 0;
    DeviceFrameState* deviceState = nullptr;
    DeviceTelemetry* deviceTelemetryScratch = nullptr;
    GraphParams* deviceGraphParams = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaGraphNode_t graphSpawnMemcpyNode = nullptr;
    cudaMemcpy3DParms graphSpawnMemcpyParams = {};
    GraphKey currentGraphKey = {};
    cudaEvent_t latestFrameStart = nullptr;
    cudaEvent_t latestAfterSimulate = nullptr;
    cudaEvent_t latestAfterCompact = nullptr;
    cudaEvent_t latestAfterSpawn = nullptr;
    std::array<TelemetrySlot, kTelemetryRingSize> telemetrySlots = {};
    std::vector<SpawnUploadSlot> spawnUploadSlots;
    size_t nextTelemetrySlot = 0;
    uint32_t maxDispatchBlocks = 1;
    uint32_t activeWorkUpperBound = 0;
    uint64_t submittedSpawnUpperBoundTotal = 0;
    SimulationSettings settings = {};
    SimulationStats stats = {};
    std::vector<TileStats> tileStats;
    std::vector<EmitterState> emitters;
    std::vector<SpawnCommand> hostSpawnCommands;
    uint32_t frameIndex = 0;
    float simulationTime = 0.0f;
    bool hasSubmittedFrame = false;
    bool graphsEnabled = true;
    bool lastFrameUsedGraph = false;
    uint32_t graphRebuildCount = 0;
};

ParticleSystem::ParticleSystem(uint32_t capacity, StorageMode mode, bool enableCudaGraphs)
    : impl_(new Impl(capacity, mode, enableCudaGraphs))
{
}

ParticleSystem::~ParticleSystem()
{
    delete impl_;
}

uint32_t ParticleSystem::addEmitter(const EmitterDesc& desc)
{
    impl_->emitters.push_back(EmitterState{desc});
    try {
        impl_->ensureSpawnCommandStorage(static_cast<uint32_t>(impl_->emitters.size()));
    } catch (...) {
        impl_->emitters.pop_back();
        throw;
    }
    impl_->currentGraphKey.valid = false;
    return static_cast<uint32_t>(impl_->emitters.size() - 1);
}

void ParticleSystem::clearEmitters()
{
    impl_->emitters.clear();
    impl_->currentGraphKey.valid = false;
}

void ParticleSystem::setEmitter(uint32_t index, const EmitterDesc& desc)
{
    if (index >= impl_->emitters.size()) {
        throw std::out_of_range("Emitter index out of range");
    }
    impl_->emitters[index].desc = desc;
    impl_->currentGraphKey.valid = false;
}

uint32_t ParticleSystem::emitterCount() const
{
    return static_cast<uint32_t>(impl_->emitters.size());
}

void ParticleSystem::queueBurst(uint32_t systemId, uint32_t count)
{
    if (systemId >= impl_->emitters.size()) {
        return;
    }

    EmitterState& emitter = impl_->emitters[systemId];
    emitter.pendingBurst = static_cast<uint32_t>(std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(),
        static_cast<uint64_t>(emitter.pendingBurst) + count));
}

void ParticleSystem::update(float dt)
{
    impl_->pollTelemetry();
    dt = std::max(0.0f, dt);

    uint64_t requestedSpawnTotal = 0;
    uint32_t commandCoveredSpawn = 0;
    impl_->hostSpawnCommands.clear();
    impl_->hostSpawnCommands.reserve(impl_->emitters.size());

    for (uint32_t systemId = 0; systemId < impl_->emitters.size(); ++systemId) {
        EmitterState& emitter = impl_->emitters[systemId];
        const double exactSpawn = static_cast<double>(emitter.spawnCarry) +
            static_cast<double>(std::max(0.0f, emitter.desc.spawnRate)) * dt;
        const double wholeSpawn = std::floor(exactSpawn);
        const uint32_t continuousSpawn = static_cast<uint32_t>(std::min<double>(
            wholeSpawn,
            static_cast<double>(std::numeric_limits<uint32_t>::max())));
        emitter.spawnCarry = static_cast<float>(exactSpawn - wholeSpawn);

        const uint32_t emitterRequest = static_cast<uint32_t>(std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(),
            static_cast<uint64_t>(continuousSpawn) + emitter.pendingBurst));
        emitter.pendingBurst = 0;
        requestedSpawnTotal = std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(),
            requestedSpawnTotal + emitterRequest);

        const uint32_t commandCount = std::min(
            emitterRequest,
            impl_->pool.capacity - commandCoveredSpawn);
        if (commandCount == 0) {
            continue;
        }

        impl_->hostSpawnCommands.push_back(SpawnCommand{
            emitter.desc,
            systemId,
            commandCoveredSpawn,
            commandCount});
        commandCoveredSpawn += commandCount;
    }

    const uint32_t requestedSpawn = static_cast<uint32_t>(requestedSpawnTotal);
    const uint32_t simulationBlocks = impl_->dispatchBlockCount(impl_->activeWorkUpperBound);
    const uint32_t spawnBlocks = impl_->dispatchBlockCount(commandCoveredSpawn);
    const uint32_t migrateBlocks = impl_->dispatchBlockCount(
        impl_->activeWorkUpperBound + commandCoveredSpawn);
    const uint32_t commandCount = static_cast<uint32_t>(impl_->hostSpawnCommands.size());
    const GraphParams params = {
        requestedSpawn,
        impl_->frameIndex,
        impl_->simulationTime + dt,
        dt};

    // This small copy replaces variable scalar kernel arguments for both
    // submission modes. It is intentionally recorded before timing starts.
    VP_CUDA_CHECK(cudaMemcpyAsync(
        impl_->deviceGraphParams,
        &params,
        sizeof(params),
        cudaMemcpyHostToDevice,
        impl_->stream));

    bool graphFrame = false;
    Impl::SpawnUploadSlot* uploadSlot = nullptr;
    if (impl_->graphsEnabled) {
        if (commandCount != 0) {
            uploadSlot = &impl_->acquireSpawnUploadSlot(commandCount);
            std::copy(
                impl_->hostSpawnCommands.begin(),
                impl_->hostSpawnCommands.end(),
                uploadSlot->hostCommands);
        }

        const Impl::GraphKey graphKey = {
            simulationBlocks,
            spawnBlocks,
            migrateBlocks,
            commandCount,
            impl_->settings.recipeCount,
            impl_->settings.tileCount,
            true};
        if (!impl_->currentGraphKey.matches(graphKey)) {
            impl_->rebuildGraph(graphKey, uploadSlot);
        }
        if (uploadSlot != nullptr) {
            // The captured memcpy stays inside the graph. Its source rotates
            // through the pinned ring, avoiding host writes to an in-flight
            // source buffer while retaining a single graph executable.
            impl_->setGraphSpawnUploadSource(*uploadSlot);
        }
        graphFrame = true;
    }

    Impl::TelemetrySlot* telemetrySlot = impl_->acquireTelemetrySlot();

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestFrameStart, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->frameStart, impl_->stream));
        telemetrySlot->graphActive = graphFrame;
    }

    if (graphFrame) {
        VP_CUDA_CHECK(cudaGraphLaunch(impl_->graphExec, impl_->stream));
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSimulate, impl_->stream));
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterCompact, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSimulate, impl_->stream));
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterCompact, impl_->stream));
        }
    } else {
        beginFrameFromParamsKernel<<<1, 256, 0, impl_->stream>>>(
            impl_->deviceState, impl_->deviceTelemetryScratch, impl_->deviceGraphParams);
        VP_CUDA_CHECK(cudaGetLastError());
        if (impl_->mode_ == StorageMode::Packed) {
            const float posScale = impl_->settings.posQuantizationScale;
            simulatePackedKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                impl_->packedPool_, impl_->deviceState, impl_->aliveFlags, impl_->freeIndices,
                impl_->deviceGraphParams, posScale, 1.0f / posScale,
                impl_->settings.recipeCount, impl_->settings.tileCount);
        } else {
            simulateKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                impl_->pool, impl_->deviceState, impl_->aliveFlags, impl_->freeIndices,
                impl_->deviceGraphParams,
                impl_->settings.recipeCount, impl_->settings.tileCount);
        }
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSimulate, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSimulate, impl_->stream));
        }

        compactActiveIndicesKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
            impl_->deviceState, impl_->aliveFlags);
        VP_CUDA_CHECK(cudaGetLastError());
        finalizeCompactionKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterCompact, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterCompact, impl_->stream));
        }

        if (commandCount != 0) {
            Impl::SpawnUploadSlot& eagerUploadSlot = impl_->acquireSpawnUploadSlot(commandCount);
            std::copy(impl_->hostSpawnCommands.begin(), impl_->hostSpawnCommands.end(), eagerUploadSlot.hostCommands);
            VP_CUDA_CHECK(cudaMemcpyAsync(
                impl_->spawnCommands, eagerUploadSlot.hostCommands,
                sizeof(SpawnCommand) * commandCount, cudaMemcpyHostToDevice, impl_->stream));
            VP_CUDA_CHECK(cudaEventRecord(eagerUploadSlot.ready, impl_->stream));
            eagerUploadSlot.inFlight = true;
        }
        prepareSpawnKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        if (commandCount != 0) {
            if (impl_->mode_ == StorageMode::Packed) {
                const float posScale = impl_->settings.posQuantizationScale;
                spawnBatchPackedKernel<<<spawnBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->packedPool_, impl_->deviceState, impl_->freeIndices, impl_->spawnCommands,
                    commandCount, impl_->settings.seed, posScale);
            } else {
                spawnBatchKernel<<<spawnBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->pool, impl_->deviceState, impl_->freeIndices, impl_->spawnCommands,
                    commandCount, impl_->settings.seed);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }
        if (impl_->settings.tileCount > 1) {
            if (impl_->mode_ == StorageMode::Packed) {
                const float posScale = impl_->settings.posQuantizationScale;
                migrateTilesPackedKernel<<<migrateBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->packedPool_, impl_->deviceState, impl_->deviceTelemetryScratch,
                    impl_->settings.tileCount, posScale, 1.0f / posScale);
            } else {
                migrateTilesKernel<<<migrateBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->pool, impl_->deviceState, impl_->deviceTelemetryScratch,
                    impl_->settings.tileCount);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSpawn, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSpawn, impl_->stream));
    }
    if (graphFrame && uploadSlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(uploadSlot->ready, impl_->stream));
        uploadSlot->inFlight = true;
    }

    impl_->advanceActiveWorkUpperBound(commandCoveredSpawn);
    if (telemetrySlot != nullptr) {
        impl_->queueTelemetry(*telemetrySlot);
    }

    impl_->simulationTime += dt;
    impl_->lastFrameUsedGraph = graphFrame;
    ++impl_->frameIndex;
    impl_->hasSubmittedFrame = true;
}

void ParticleSystem::reset()
{
    impl_->initializeFreeList();
    impl_->stats = {};
    impl_->stats.capacity = impl_->pool.capacity;
    impl_->frameIndex = 0;
    impl_->simulationTime = 0.0f;
    for (EmitterState& emitter : impl_->emitters) {
        emitter.spawnCarry = 0.0f;
        emitter.pendingBurst = 0;
    }
}

void ParticleSystem::setSettings(const SimulationSettings& settings)
{
    impl_->settings = settings;
    if (impl_->settings.recipeCount == 0) {
        impl_->settings.recipeCount = 1;
    }
    // Sync legacy/convenience module fields into recipe 0 only when in single-recipe mode
    if (impl_->settings.recipeCount <= 1) {
        impl_->settings.recipes[0].gravity = impl_->settings.gravity;
        impl_->settings.recipes[0].wind = impl_->settings.wind;
        impl_->settings.recipes[0].drag = impl_->settings.drag;
        impl_->settings.recipes[0].turbulence = impl_->settings.turbulence;
        impl_->settings.recipes[0].plane = impl_->settings.plane;
        impl_->settings.recipes[0].sphere = impl_->settings.sphere;
        impl_->settings.recipes[0].box = impl_->settings.box;
        impl_->settings.recipes[0].curves = impl_->settings.curves;
    } else {
        // Multi-recipe mode: sync recipe 0 into legacy fields for convenience readers
        impl_->settings.gravity = impl_->settings.recipes[0].gravity;
        impl_->settings.wind = impl_->settings.recipes[0].wind;
        impl_->settings.drag = impl_->settings.recipes[0].drag;
        impl_->settings.turbulence = impl_->settings.recipes[0].turbulence;
        impl_->settings.plane = impl_->settings.recipes[0].plane;
        impl_->settings.sphere = impl_->settings.recipes[0].sphere;
        impl_->settings.box = impl_->settings.recipes[0].box;
        impl_->settings.curves = impl_->settings.recipes[0].curves;
    }

    if (impl_->settings.tileCount <= 1) {
        impl_->settings.tileCount = 1;
        if (settings.tiles[0].originX == 0.0f && settings.tiles[0].originY == 0.0f && settings.tiles[0].originZ == 0.0f) {
            impl_->settings.tiles[0].originX = settings.tileOrigin.x;
            impl_->settings.tiles[0].originY = settings.tileOrigin.y;
            impl_->settings.tiles[0].originZ = settings.tileOrigin.z;
        } else {
            impl_->settings.tileOrigin = {
                settings.tiles[0].originX,
                settings.tiles[0].originY,
                settings.tiles[0].originZ
            };
        }
    }
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cTiles,
        impl_->settings.tiles,
        sizeof(TileDesc) * kMaxTiles));
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
    impl_->currentGraphKey.valid = false;
}

const SimulationSettings& ParticleSystem::settings() const
{
    return impl_->settings;
}

StorageMode ParticleSystem::storageMode() const
{
    return impl_->mode_;
}

uint32_t ParticleSystem::addRecipe(const EffectRecipe& recipe)
{
    if (impl_->settings.recipeCount >= kMaxRecipes) {
        throw std::runtime_error("Maximum recipe count reached (kMaxRecipes = 64)");
    }
    const uint32_t id = impl_->settings.recipeCount++;
    impl_->settings.recipes[id] = recipe;
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
    impl_->currentGraphKey.valid = false;
    return id;
}

void ParticleSystem::setRecipe(uint32_t recipeId, const EffectRecipe& recipe)
{
    if (recipeId >= kMaxRecipes) {
        throw std::out_of_range("Recipe ID out of range");
    }
    impl_->settings.recipes[recipeId] = recipe;
    if (recipeId >= impl_->settings.recipeCount) {
        impl_->settings.recipeCount = recipeId + 1;
        impl_->currentGraphKey.valid = false;
    }
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
}

const EffectRecipe& ParticleSystem::recipe(uint32_t recipeId) const
{
    if (recipeId >= kMaxRecipes) {
        throw std::out_of_range("Recipe ID out of range");
    }
    return impl_->settings.recipes[recipeId];
}

uint32_t ParticleSystem::recipeCount() const
{
    return impl_->settings.recipeCount;
}

void ParticleSystem::synchronize()
{
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    impl_->pollTelemetry();
    impl_->pollSpawnUploadSlots();
    if (!impl_->hasSubmittedFrame) {
        return;
    }

    DeviceTelemetry telemetry = {};
    snapshotTelemetryKernel<<<1, 1, 0, impl_->stream>>>(
        impl_->deviceState,
        impl_->deviceTelemetryScratch,
        impl_->settings.tileCount);
    VP_CUDA_CHECK(cudaGetLastError());
    VP_CUDA_CHECK(cudaMemcpyAsync(
        &telemetry,
        impl_->deviceTelemetryScratch,
        sizeof(telemetry),
        cudaMemcpyDeviceToHost,
        impl_->stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

    float spawnMs = 0.0f;
    float simulateMs = 0.0f;
    float compactMs = 0.0f;
    float totalMs = 0.0f;
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &spawnMs,
        impl_->latestAfterCompact,
        impl_->latestAfterSpawn));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &simulateMs,
        impl_->latestFrameStart,
        impl_->latestAfterSimulate));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &compactMs,
        impl_->latestAfterSimulate,
        impl_->latestAfterCompact));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &totalMs,
        impl_->latestFrameStart,
        impl_->latestAfterSpawn));
    if (impl_->lastFrameUsedGraph) {
        simulateMs = totalMs;
        compactMs = 0.0f;
        spawnMs = 0.0f;
    }
    impl_->publishTelemetry(
        telemetry, 0, spawnMs, simulateMs, compactMs, totalMs,
        impl_->lastFrameUsedGraph);
    impl_->activeWorkUpperBound = telemetry.aliveCount;
}

const ParticlePool& ParticleSystem::buffers() const
{
    return impl_->pool;
}

const GpuParticlePool& ParticleSystem::gpuBuffers() const
{
    return impl_->gpuPool;
}

const SimulationStats& ParticleSystem::stats() const
{
    return impl_->stats;
}

const std::vector<TileStats>& ParticleSystem::tileStats() const
{
    return impl_->tileStats;
}

} // namespace vparticles
