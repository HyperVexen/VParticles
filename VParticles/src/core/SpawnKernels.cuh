#pragma once

#include "InternalTypes.h"
#include "ParticleMath.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace vparticles {

__global__ void initializeFreeListKernel(uint32_t* freeIndices, uint32_t capacity)
{
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < capacity) {
        freeIndices[index] = static_cast<uint32_t>(index);
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

} // namespace vparticles
