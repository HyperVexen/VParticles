#pragma once

#include "VParticles/NeighborTypes.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace vparticles {

constexpr int kNeighborThreadsPerBlock = 256;

__device__ inline float3 makeNeighborFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dotNeighbor3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 subNeighbor3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

// -----------------------------------------------------------------------------
// 1. Key Generation Kernel
// Computes 3D integer cell coordinates and linear 32-bit spatial cell key
// -----------------------------------------------------------------------------
static __global__ void generateKeysKernel(
    const float4* __restrict__ pos,
    uint32_t* __restrict__ keys,
    uint32_t* __restrict__ indices,
    GridParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    const float4 p = pos[i];
    const int cx = min(max(static_cast<int>(floorf((p.x - params.origin.x) * params.invCellSize)), 0), params.gridDim.x - 1);
    const int cy = min(max(static_cast<int>(floorf((p.y - params.origin.y) * params.invCellSize)), 0), params.gridDim.y - 1);
    const int cz = min(max(static_cast<int>(floorf((p.z - params.origin.z) * params.invCellSize)), 0), params.gridDim.z - 1);

    const uint32_t cellIdx = cx + cy * params.gridDim.x + cz * params.gridDim.x * params.gridDim.y;

    if (params.enableColorKey) {
        const uint32_t color = (cx & 1) | ((cy & 1) << 1) | ((cz & 1) << 2);
        keys[i] = (color << 28) | (cellIdx & 0x0FFFFFFF);
    } else {
        keys[i] = cellIdx;
    }

    indices[i] = i;
}

// -----------------------------------------------------------------------------
// 2. Cell Bounds Kernel
// Identifies contiguous cell intervals [cellStart, cellEnd) in sorted keys
// -----------------------------------------------------------------------------
static __global__ void findCellBoundsKernel(
    const uint32_t* __restrict__ sortedKeys,
    uint32_t* __restrict__ cellStart,
    uint32_t* __restrict__ cellEnd,
    uint32_t grainCount,
    bool enableColorKey)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const uint32_t keyMask = enableColorKey ? 0x0FFFFFFF : 0xFFFFFFFF;
    const uint32_t currentCell = sortedKeys[i] & keyMask;

    if (i == 0) {
        cellStart[currentCell] = 0;
    } else {
        const uint32_t prevCell = sortedKeys[i - 1] & keyMask;
        if (currentCell != prevCell) {
            cellEnd[prevCell] = i;
            cellStart[currentCell] = i;
        }
    }

    if (i == grainCount - 1) {
        cellEnd[currentCell] = grainCount;
    }
}

// -----------------------------------------------------------------------------
// 3. Dense SoA Reorder Kernel
// Permutes grain attributes into spatially coherent order
// -----------------------------------------------------------------------------
static __global__ void reorderGranularPoolKernel(
    const float4* __restrict__ inPos,
    const float4* __restrict__ inVel,
    const float4* __restrict__ inPred,
    const uint32_t* __restrict__ inTint,
    const uint16_t* __restrict__ inMaterialId,
    const uint16_t* __restrict__ inFlags,
    const uint32_t* __restrict__ inGrainId,
    const uint32_t* __restrict__ sortedIndices,
    float4* __restrict__ outPos,
    float4* __restrict__ outVel,
    float4* __restrict__ outPred,
    uint32_t* __restrict__ outTint,
    uint16_t* __restrict__ outMaterialId,
    uint16_t* __restrict__ outFlags,
    uint32_t* __restrict__ outGrainId,
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const uint32_t origIdx = sortedIndices[i];
    outPos[i] = inPos[origIdx];
    outVel[i] = inVel[origIdx];
    outPred[i] = inPred[origIdx];
    outTint[i] = inTint[origIdx];
    outMaterialId[i] = inMaterialId[origIdx];
    outFlags[i] = inFlags[origIdx];
    outGrainId[i] = inGrainId[origIdx];
}

// -----------------------------------------------------------------------------
// 4. Verlet Neighbor List Build Kernel
// Queries 27 neighboring cells and records up to 32 neighbors per grain
// -----------------------------------------------------------------------------
static __global__ void buildVerletListKernel(
    const float4* __restrict__ sortedPos,
    const uint32_t* __restrict__ cellStart,
    const uint32_t* __restrict__ cellEnd,
    uint32_t* __restrict__ neighborList,
    uint32_t* __restrict__ neighborCount,
    uint32_t* __restrict__ overflowCounter,
    GridParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    const float4 p4 = sortedPos[i];
    const float3 p = makeNeighborFloat3(p4);
    const float ri = p4.w > 0.0f ? p4.w : params.radius;

    const int cx = min(max(static_cast<int>(floorf((p.x - params.origin.x) * params.invCellSize)), 0), params.gridDim.x - 1);
    const int cy = min(max(static_cast<int>(floorf((p.y - params.origin.y) * params.invCellSize)), 0), params.gridDim.y - 1);
    const int cz = min(max(static_cast<int>(floorf((p.z - params.origin.z) * params.invCellSize)), 0), params.gridDim.z - 1);

    uint32_t count = 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    #pragma unroll 1
    for (int dz = -1; dz <= 1; ++dz) {
        const int ncz = cz + dz;
        if (ncz < 0 || ncz >= params.gridDim.z) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            const int ncy = cy + dy;
            if (ncy < 0 || ncy >= params.gridDim.y) continue;

            for (int dx = -1; dx <= 1; ++dx) {
                const int ncx = cx + dx;
                if (ncx < 0 || ncx >= params.gridDim.x) continue;

                const uint32_t ncell = ncx + ncy * params.gridDim.x + ncz * params.gridDim.x * params.gridDim.y;
                const uint32_t start = cellStart[ncell];
                if (start == 0xFFFFFFFF) continue;
                const uint32_t end = cellEnd[ncell];

                for (uint32_t j = start; j < end; ++j) {
                    if (j == i) continue;

                    const float4 otherP4 = sortedPos[j];
                    const float3 otherP = makeNeighborFloat3(otherP4);
                    const float rj = otherP4.w > 0.0f ? otherP4.w : params.radius;

                    const float3 diff = subNeighbor3(otherP, p);
                    const float distSq = dotNeighbor3(diff, diff);
                    const float maxDist = ri + rj + params.skin;

                    if (distSq < maxDist * maxDist) {
                        if (count < kMaxNeighborsPerGrain) {
                            neighborList[baseOffset + count] = j;
                        }
                        count++;
                    }
                }
            }
        }
    }

    if (count > kMaxNeighborsPerGrain) {
        atomicAdd(overflowCounter, count - kMaxNeighborsPerGrain);
    }
    neighborCount[i] = min(count, kMaxNeighborsPerGrain);
}

// -----------------------------------------------------------------------------
// 5. Displacement Check Kernel
// Flags d_rebuildNeeded if any grain has moved by more than skin / 2
// -----------------------------------------------------------------------------
static __global__ void checkDisplacementKernel(
    const float4* __restrict__ currentPos,
    const float4* __restrict__ referencePos,
    uint32_t* __restrict__ d_rebuildNeeded,
    float halfSkinSq,
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const float3 p = makeNeighborFloat3(currentPos[i]);
    const float3 ref = makeNeighborFloat3(referencePos[i]);
    const float3 diff = subNeighbor3(p, ref);
    const float distSq = dotNeighbor3(diff, diff);

    if (distSq > halfSkinSq) {
        *d_rebuildNeeded = 1;
    }
}

// -----------------------------------------------------------------------------
// 6. Generate 8-Color Keys Kernel
// Computes 3-bit color key (0..7) for each grain from its cell coordinate
// -----------------------------------------------------------------------------
static __global__ void generateColorKeysKernel(
    const float4* __restrict__ pos,
    uint32_t* __restrict__ colorKeys,
    uint32_t* __restrict__ colorIndices,
    GridParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    const float4 p = pos[i];
    const int cx = min(max(static_cast<int>(floorf((p.x - params.origin.x) * params.invCellSize)), 0), params.gridDim.x - 1);
    const int cy = min(max(static_cast<int>(floorf((p.y - params.origin.y) * params.invCellSize)), 0), params.gridDim.y - 1);
    const int cz = min(max(static_cast<int>(floorf((p.z - params.origin.z) * params.invCellSize)), 0), params.gridDim.z - 1);

    const uint32_t color = (cx & 1) | ((cy & 1) << 1) | ((cz & 1) << 2);
    colorKeys[i] = color;
    colorIndices[i] = i;
}

// -----------------------------------------------------------------------------
// 7. Find 8-Color Bounds Kernel
// Finds start and end boundaries for each of the 8 colors
// -----------------------------------------------------------------------------
static __global__ void findColorBoundsKernel(
    const uint32_t* __restrict__ sortedColorKeys,
    uint32_t* __restrict__ colorOffsets, // 9 elements: offsets for 0..7, [8]=grainCount
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const uint32_t currentColor = sortedColorKeys[i];

    if (i == 0) {
        for (uint32_t c = 0; c <= currentColor; ++c) {
            colorOffsets[c] = 0;
        }
    } else {
        const uint32_t prevColor = sortedColorKeys[i - 1];
        if (currentColor != prevColor) {
            for (uint32_t c = prevColor + 1; c <= currentColor; ++c) {
                colorOffsets[c] = i;
            }
        }
    }

    if (i == grainCount - 1) {
        for (uint32_t c = currentColor + 1; c <= 8; ++c) {
            colorOffsets[c] = grainCount;
        }
    }
}

// -----------------------------------------------------------------------------
// 8. Ground-Truth Brute Force Neighbor Search (for 10K grain validation)
// -----------------------------------------------------------------------------
static __global__ void bruteForceNeighborsKernel(
    const float4* __restrict__ positions,
    uint32_t* __restrict__ bfCount,
    uint32_t* __restrict__ bfList,
    uint32_t grainCount,
    float skin,
    float defaultRadius)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const float4 p4 = positions[i];
    const float3 p = makeNeighborFloat3(p4);
    const float ri = p4.w > 0.0f ? p4.w : defaultRadius;

    uint32_t count = 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    for (uint32_t j = 0; j < grainCount; ++j) {
        if (j == i) continue;

        const float4 otherP4 = positions[j];
        const float3 otherP = makeNeighborFloat3(otherP4);
        const float rj = otherP4.w > 0.0f ? otherP4.w : defaultRadius;

        const float3 diff = subNeighbor3(otherP, p);
        const float distSq = dotNeighbor3(diff, diff);
        const float maxDist = ri + rj + skin;

        if (distSq < maxDist * maxDist) {
            if (count < kMaxNeighborsPerGrain) {
                bfList[baseOffset + count] = j;
            }
            count++;
        }
    }

    bfCount[i] = min(count, kMaxNeighborsPerGrain);
}

} // namespace vparticles
