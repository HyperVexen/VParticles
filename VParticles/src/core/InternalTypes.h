#pragma once

#include "VParticles/ParticlePool.h"
#include "VParticles/SimulationTypes.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vparticles {

// ── Constant Memory Declarations ──────────────────────────────────────
extern __constant__ TileDesc cTiles[kMaxTiles];
extern __constant__ EffectRecipe cRecipes[kMaxRecipes];

// ── Execution Constants ───────────────────────────────────────────────
constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kTargetBlocksPerMultiprocessor = 8;
constexpr size_t kTelemetryRingSize = 8;
constexpr size_t kSpawnUploadRingSize = 8;

// ── Allocation Helpers ────────────────────────────────────────────────
template <typename T>
inline void cudaAlloc(T*& pointer, size_t count)
{
    VP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
}

inline uint32_t blockCount(uint32_t itemCount)
{
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(itemCount) + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

inline void allocatePool(ParticlePool& pool, uint32_t capacity)
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

inline void releasePool(ParticlePool& pool)
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

inline void allocatePackedPool(PackedPool& pool, uint32_t capacity)
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

inline void releasePackedPool(PackedPool& pool)
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

// ── Device State Structures ───────────────────────────────────────────
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

struct GraphParams {
    uint32_t requestedSpawn = 0;
    uint32_t frameIndex = 0;
    float simTime = 0.0f;
    float dt = 0.0f;
};

struct SpawnCommand {
    EmitterDesc emitter = {};
    uint32_t systemId = 0;
    uint32_t requestStart = 0;
    uint32_t count = 0;
};

__device__ inline uint32_t findSpawnCommand(
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

struct EmitterState {
    EmitterDesc desc = {};
    float spawnCarry = 0.0f;
    uint32_t pendingBurst = 0;
};

} // namespace vparticles
