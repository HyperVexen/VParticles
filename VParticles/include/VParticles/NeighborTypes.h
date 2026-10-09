#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace vparticles {

constexpr uint32_t kMaxNeighborsPerGrain = 32;

struct GridParams {
    float3 origin = {0.0f, 0.0f, 0.0f};
    float cellSize = 0.024f;        // 2 * maxRadius + skin
    float invCellSize = 41.6667f;
    int3 gridDim = {1, 1, 1};       // (dimX, dimY, dimZ)
    uint32_t numCells = 1;
    float skin = 0.002f;            // skin margin for rebuild trigger
    float radius = 0.01f;
    float searchRadius = 0.022f;    // 2 * maxRadius + skin
    float searchRadiusSq = 0.022f * 0.022f;
    uint32_t grainCount = 0;
    bool enableColorKey = false;
};

struct NeighborData {
    uint32_t* list = nullptr;             // capacity * 32
    uint32_t* counts = nullptr;           // capacity * 4 bytes
    float4* referencePos = nullptr;       // capacity * 16 bytes
    uint32_t* cellStart = nullptr;        // numCells * 4 bytes
    uint32_t* cellEnd = nullptr;          // numCells * 4 bytes
    uint32_t* colorOffsets = nullptr;     // 9 elements (offsets for 8 colors: 0..7, plus total)
    uint32_t hostColorOffsets[9] = {};    // 9 elements host-cached for zero-sync Gauss-Seidel dispatch
    uint32_t* colorIndices = nullptr;     // capacity * 4 bytes
    uint32_t* d_overflowCounter = nullptr;// 4 bytes device counter
    uint32_t* d_rebuildNeeded = nullptr;  // 4 bytes device flag
    uint32_t maxNeighborsPerGrain = kMaxNeighborsPerGrain;
    uint32_t capacity = 0;
    uint32_t aliveCount = 0;
};

struct NeighborStats {
    uint64_t totalRebuilds = 0;
    uint64_t displacementChecks = 0;
    uint32_t lastOverflowCount = 0;
    float lastRebuildMs = 0.0f;
    float lastKeygenMs = 0.0f;
    float lastSortMs = 0.0f;
    float lastBoundsMs = 0.0f;
    float lastReorderMs = 0.0f;
    float lastVerletBuildMs = 0.0f;
    float lastDispCheckMs = 0.0f;
    float amortizedRebuildMs = 0.0f;
};

} // namespace vparticles
