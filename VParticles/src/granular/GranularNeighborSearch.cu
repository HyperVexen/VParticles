#include "granular/GranularNeighborSearch.h"
#include "granular/GranularNeighborKernels.cuh"
#include "VParticles/CudaCheck.h"

#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace vparticles {

namespace {

constexpr uint32_t kDefaultMaxCells = 4'194'304; // 4M cells (~16 MB per cell bounds table)
constexpr uint32_t kMaxBlocksForBounds = 1024;

__device__ inline float warpReduceMin(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceMax(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// -----------------------------------------------------------------------------
// Two-pass deterministic bounding box reduction
// -----------------------------------------------------------------------------
__global__ void computeBlockBoundsKernel(
    const float4* __restrict__ pos,
    uint32_t count,
    float3* __restrict__ blockMins,
    float3* __restrict__ blockMaxs)
{
    __shared__ float sMinX[8], sMinY[8], sMinZ[8];
    __shared__ float sMaxX[8], sMaxY[8], sMaxZ[8];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 minP = make_float3(1e30f, 1e30f, 1e30f);
    float3 maxP = make_float3(-1e30f, -1e30f, -1e30f);

    if (i < count) {
        const float4 p = pos[i];
        minP = make_float3(p.x, p.y, p.z);
        maxP = make_float3(p.x, p.y, p.z);
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;

    minP.x = warpReduceMin(minP.x);
    minP.y = warpReduceMin(minP.y);
    minP.z = warpReduceMin(minP.z);
    maxP.x = warpReduceMax(maxP.x);
    maxP.y = warpReduceMax(maxP.y);
    maxP.z = warpReduceMax(maxP.z);

    if (lane == 0) {
        sMinX[warpId] = minP.x; sMinY[warpId] = minP.y; sMinZ[warpId] = minP.z;
        sMaxX[warpId] = maxP.x; sMaxY[warpId] = maxP.y; sMaxZ[warpId] = maxP.z;
    }
    __syncthreads();

    if (warpId == 0) {
        float bMinX = (lane < 8) ? sMinX[lane] : 1e30f;
        float bMinY = (lane < 8) ? sMinY[lane] : 1e30f;
        float bMinZ = (lane < 8) ? sMinZ[lane] : 1e30f;
        float bMaxX = (lane < 8) ? sMaxX[lane] : -1e30f;
        float bMaxY = (lane < 8) ? sMaxY[lane] : -1e30f;
        float bMaxZ = (lane < 8) ? sMaxZ[lane] : -1e30f;

        bMinX = warpReduceMin(bMinX);
        bMinY = warpReduceMin(bMinY);
        bMinZ = warpReduceMin(bMinZ);
        bMaxX = warpReduceMax(bMaxX);
        bMaxY = warpReduceMax(bMaxY);
        bMaxZ = warpReduceMax(bMaxZ);

        if (lane == 0) {
            blockMins[blockIdx.x] = make_float3(bMinX, bMinY, bMinZ);
            blockMaxs[blockIdx.x] = make_float3(bMaxX, bMaxY, bMaxZ);
        }
    }
}

__global__ void reduceFinalBoundsKernel(
    const float3* __restrict__ blockMins,
    const float3* __restrict__ blockMaxs,
    uint32_t blockCount,
    float3* __restrict__ outMin,
    float3* __restrict__ outMax)
{
    __shared__ float sMinX[8], sMinY[8], sMinZ[8];
    __shared__ float sMaxX[8], sMaxY[8], sMaxZ[8];

    float3 minP = make_float3(1e30f, 1e30f, 1e30f);
    float3 maxP = make_float3(-1e30f, -1e30f, -1e30f);

    for (uint32_t i = threadIdx.x; i < blockCount; i += blockDim.x) {
        const float3 bMin = blockMins[i];
        const float3 bMax = blockMaxs[i];
        minP.x = fminf(minP.x, bMin.x);
        minP.y = fminf(minP.y, bMin.y);
        minP.z = fminf(minP.z, bMin.z);
        maxP.x = fmaxf(maxP.x, bMax.x);
        maxP.y = fmaxf(maxP.y, bMax.y);
        maxP.z = fmaxf(maxP.z, bMax.z);
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;

    minP.x = warpReduceMin(minP.x);
    minP.y = warpReduceMin(minP.y);
    minP.z = warpReduceMin(minP.z);
    maxP.x = warpReduceMax(maxP.x);
    maxP.y = warpReduceMax(maxP.y);
    maxP.z = warpReduceMax(maxP.z);

    if (lane == 0) {
        sMinX[warpId] = minP.x; sMinY[warpId] = minP.y; sMinZ[warpId] = minP.z;
        sMaxX[warpId] = maxP.x; sMaxY[warpId] = maxP.y; sMaxZ[warpId] = maxP.z;
    }
    __syncthreads();

    if (warpId == 0) {
        float fMinX = (lane < 8) ? sMinX[lane] : 1e30f;
        float fMinY = (lane < 8) ? sMinY[lane] : 1e30f;
        float fMinZ = (lane < 8) ? sMinZ[lane] : 1e30f;
        float fMaxX = (lane < 8) ? sMaxX[lane] : -1e30f;
        float fMaxY = (lane < 8) ? sMaxY[lane] : -1e30f;
        float fMaxZ = (lane < 8) ? sMaxZ[lane] : -1e30f;

        fMinX = warpReduceMin(fMinX);
        fMinY = warpReduceMin(fMinY);
        fMinZ = warpReduceMin(fMinZ);
        fMaxX = warpReduceMax(fMaxX);
        fMaxY = warpReduceMax(fMaxY);
        fMaxZ = warpReduceMax(fMaxZ);

        if (lane == 0) {
            *outMin = make_float3(fMinX, fMinY, fMinZ);
            *outMax = make_float3(fMaxX, fMaxY, fMaxZ);
        }
    }
}

} // namespace

struct GranularNeighborSearch::Impl {
    uint32_t capacity = 0;
    uint32_t maxCells = kDefaultMaxCells;

    GridParams gridParams = {};
    NeighborData neighborData = {};
    NeighborStats stats = {};

    // CUB sort storage
    void* d_sortStorage = nullptr;
    size_t sortStorageBytes = 0;

    void* d_colorSortStorage = nullptr;
    size_t colorSortStorageBytes = 0;

    // Keys and indices for spatial sort
    uint32_t* d_keysIn = nullptr;
    uint32_t* d_keysOut = nullptr;
    uint32_t* d_indicesIn = nullptr;
    uint32_t* d_indicesOut = nullptr;

    // Color keys for 8-color sort
    uint32_t* d_colorKeysIn = nullptr;
    uint32_t* d_colorKeysOut = nullptr;
    uint32_t* d_colorIndicesIn = nullptr;

    // Scratch buffers for dense SoA reorder
    float4* d_scratchPos = nullptr;
    float4* d_scratchVel = nullptr;
    float4* d_scratchPred = nullptr;
    uint32_t* d_scratchTint = nullptr;
    uint16_t* d_scratchMaterialId = nullptr;
    uint16_t* d_scratchFlags = nullptr;
    uint32_t* d_scratchGrainId = nullptr;

    // Bounds reduction scratch buffers
    float3* d_blockMins = nullptr;
    float3* d_blockMaxs = nullptr;
    float3* d_globalMin = nullptr;
    float3* d_globalMax = nullptr;

    // Timing events
    cudaEvent_t evStart = nullptr;
    cudaEvent_t evKeygen = nullptr;
    cudaEvent_t evSort = nullptr;
    cudaEvent_t evBounds = nullptr;
    cudaEvent_t evReorder = nullptr;
    cudaEvent_t evVerlet = nullptr;
    cudaEvent_t evEnd = nullptr;

    Impl(uint32_t cap)
        : capacity(cap)
    {
        neighborData.capacity = cap;
        neighborData.maxNeighborsPerGrain = kMaxNeighborsPerGrain;

        // Allocate neighbor list and counts
        VP_CUDA_CHECK(cudaMalloc(&neighborData.list, sizeof(uint32_t) * cap * kMaxNeighborsPerGrain));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.counts, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.referencePos, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.cellStart, sizeof(uint32_t) * maxCells));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.cellEnd, sizeof(uint32_t) * maxCells));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.colorOffsets, sizeof(uint32_t) * 9));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.colorIndices, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.d_overflowCounter, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&neighborData.d_rebuildNeeded, sizeof(uint32_t)));

        // Allocate spatial sort buffers
        VP_CUDA_CHECK(cudaMalloc(&d_keysIn, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_keysOut, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_indicesIn, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_indicesOut, sizeof(uint32_t) * cap));

        // Allocate color sort buffers
        VP_CUDA_CHECK(cudaMalloc(&d_colorKeysIn, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_colorKeysOut, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_colorIndicesIn, sizeof(uint32_t) * cap));

        // Allocate reorder scratch buffers
        VP_CUDA_CHECK(cudaMalloc(&d_scratchPos, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchVel, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchPred, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchTint, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchMaterialId, sizeof(uint16_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchFlags, sizeof(uint16_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchGrainId, sizeof(uint32_t) * cap));

        // Bounds reduction buffers
        VP_CUDA_CHECK(cudaMalloc(&d_blockMins, sizeof(float3) * kMaxBlocksForBounds));
        VP_CUDA_CHECK(cudaMalloc(&d_blockMaxs, sizeof(float3) * kMaxBlocksForBounds));
        VP_CUDA_CHECK(cudaMalloc(&d_globalMin, sizeof(float3)));
        VP_CUDA_CHECK(cudaMalloc(&d_globalMax, sizeof(float3)));

        // Query CUB sort temporary storage requirement
        cub::DeviceRadixSort::SortPairs(
            nullptr, sortStorageBytes,
            d_keysIn, d_keysOut,
            d_indicesIn, d_indicesOut,
            cap, 0, 32);
        VP_CUDA_CHECK(cudaMalloc(&d_sortStorage, sortStorageBytes));

        // Query CUB color sort temporary storage requirement (3 bits: 0..3)
        cub::DeviceRadixSort::SortPairs(
            nullptr, colorSortStorageBytes,
            d_colorKeysIn, d_colorKeysOut,
            d_colorIndicesIn, neighborData.colorIndices,
            cap, 0, 3);
        VP_CUDA_CHECK(cudaMalloc(&d_colorSortStorage, colorSortStorageBytes));

        // Create timing events
        VP_CUDA_CHECK(cudaEventCreate(&evStart));
        VP_CUDA_CHECK(cudaEventCreate(&evKeygen));
        VP_CUDA_CHECK(cudaEventCreate(&evSort));
        VP_CUDA_CHECK(cudaEventCreate(&evBounds));
        VP_CUDA_CHECK(cudaEventCreate(&evReorder));
        VP_CUDA_CHECK(cudaEventCreate(&evVerlet));
        VP_CUDA_CHECK(cudaEventCreate(&evEnd));
    }

    ~Impl()
    {
        if (evStart) cudaEventDestroy(evStart);
        if (evKeygen) cudaEventDestroy(evKeygen);
        if (evSort) cudaEventDestroy(evSort);
        if (evBounds) cudaEventDestroy(evBounds);
        if (evReorder) cudaEventDestroy(evReorder);
        if (evVerlet) cudaEventDestroy(evVerlet);
        if (evEnd) cudaEventDestroy(evEnd);

        if (d_globalMin) cudaFree(d_globalMin);
        if (d_globalMax) cudaFree(d_globalMax);
        if (d_blockMins) cudaFree(d_blockMins);
        if (d_blockMaxs) cudaFree(d_blockMaxs);

        if (d_scratchGrainId) cudaFree(d_scratchGrainId);
        if (d_scratchFlags) cudaFree(d_scratchFlags);
        if (d_scratchMaterialId) cudaFree(d_scratchMaterialId);
        if (d_scratchTint) cudaFree(d_scratchTint);
        if (d_scratchPred) cudaFree(d_scratchPred);
        if (d_scratchVel) cudaFree(d_scratchVel);
        if (d_scratchPos) cudaFree(d_scratchPos);

        if (d_colorIndicesIn) cudaFree(d_colorIndicesIn);
        if (d_colorKeysOut) cudaFree(d_colorKeysOut);
        if (d_colorKeysIn) cudaFree(d_colorKeysIn);
        if (d_colorSortStorage) cudaFree(d_colorSortStorage);

        if (d_sortStorage) cudaFree(d_sortStorage);
        if (d_indicesOut) cudaFree(d_indicesOut);
        if (d_indicesIn) cudaFree(d_indicesIn);
        if (d_keysOut) cudaFree(d_keysOut);
        if (d_keysIn) cudaFree(d_keysIn);

        if (neighborData.d_rebuildNeeded) cudaFree(neighborData.d_rebuildNeeded);
        if (neighborData.d_overflowCounter) cudaFree(neighborData.d_overflowCounter);
        if (neighborData.colorIndices) cudaFree(neighborData.colorIndices);
        if (neighborData.colorOffsets) cudaFree(neighborData.colorOffsets);
        if (neighborData.cellEnd) cudaFree(neighborData.cellEnd);
        if (neighborData.cellStart) cudaFree(neighborData.cellStart);
        if (neighborData.referencePos) cudaFree(neighborData.referencePos);
        if (neighborData.counts) cudaFree(neighborData.counts);
        if (neighborData.list) cudaFree(neighborData.list);
    }
};

GranularNeighborSearch::GranularNeighborSearch(uint32_t capacity)
    : impl_(std::make_unique<Impl>(capacity))
{
}

GranularNeighborSearch::~GranularNeighborSearch() = default;

void GranularNeighborSearch::build(
    GranularPool& pool,
    const GranularSettings& settings,
    cudaStream_t stream)
{
    const uint32_t count = pool.aliveCount;
    if (count == 0) return;

    impl_->neighborData.aliveCount = count;
    VP_CUDA_CHECK(cudaEventRecord(impl_->evStart, stream));
    const float maxRadius = settings.grainRadius * (1.0f + std::max(0.0f, settings.radiusJitter));
    const float skin = settings.skinRatio * settings.grainRadius;
    const float maxSearchDist = 2.0f * maxRadius + skin;
    const float cellSize = maxSearchDist;

    // 1. Compute particle domain bounds
    const uint32_t numBlocks = std::min(
        kMaxBlocksForBounds,
        (count + kNeighborThreadsPerBlock - 1) / kNeighborThreadsPerBlock);

    computeBlockBoundsKernel<<<numBlocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, count, impl_->d_blockMins, impl_->d_blockMaxs);
    reduceFinalBoundsKernel<<<1, kNeighborThreadsPerBlock, 0, stream>>>(
        impl_->d_blockMins, impl_->d_blockMaxs, numBlocks, impl_->d_globalMin, impl_->d_globalMax);

    float3 hostMin, hostMax;
    VP_CUDA_CHECK(cudaMemcpyAsync(&hostMin, impl_->d_globalMin, sizeof(float3), cudaMemcpyDeviceToHost, stream));
    VP_CUDA_CHECK(cudaMemcpyAsync(&hostMax, impl_->d_globalMax, sizeof(float3), cudaMemcpyDeviceToHost, stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pad domain by 2 cell margins
    const float margin = cellSize * 2.0f;
    hostMin.x -= margin; hostMin.y -= margin; hostMin.z -= margin;
    hostMax.x += margin; hostMax.y += margin; hostMax.z += margin;

    int3 gridDim = make_int3(
        std::max(1, static_cast<int>(std::ceil((hostMax.x - hostMin.x) / cellSize))),
        std::max(1, static_cast<int>(std::ceil((hostMax.y - hostMin.y) / cellSize))),
        std::max(1, static_cast<int>(std::ceil((hostMax.z - hostMin.z) / cellSize))));

    uint64_t totalCells = static_cast<uint64_t>(gridDim.x) * gridDim.y * gridDim.z;
    if (totalCells > impl_->maxCells) {
        // Clamp grid dimensions to stay within allocated bounds table
        const float scale = std::cbrt(static_cast<double>(impl_->maxCells) / static_cast<double>(totalCells));
        gridDim.x = std::max(1, static_cast<int>(gridDim.x * scale));
        gridDim.y = std::max(1, static_cast<int>(gridDim.y * scale));
        gridDim.z = std::max(1, static_cast<int>(gridDim.z * scale));
        totalCells = static_cast<uint64_t>(gridDim.x) * gridDim.y * gridDim.z;
    }
    const uint32_t numCells = static_cast<uint32_t>(totalCells);

    impl_->gridParams = {
        hostMin, cellSize, 1.0f / cellSize, gridDim, numCells,
        skin, settings.grainRadius, maxSearchDist,
        maxSearchDist * maxSearchDist,
        count, false
    };

    const uint32_t blocks = (count + kNeighborThreadsPerBlock - 1) / kNeighborThreadsPerBlock;

    // 2. Generate linear spatial keys
    generateKeysKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, impl_->d_keysIn, impl_->d_indicesIn, impl_->gridParams);
    VP_CUDA_CHECK(cudaEventRecord(impl_->evKeygen, stream));

    // 3. CUB DeviceRadixSort on 32-bit linear keys
    VP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        impl_->d_sortStorage, impl_->sortStorageBytes,
        impl_->d_keysIn, impl_->d_keysOut,
        impl_->d_indicesIn, impl_->d_indicesOut,
        count, 0, 32, stream));
    VP_CUDA_CHECK(cudaEventRecord(impl_->evSort, stream));

    // 4. Find cell intervals [cellStart, cellEnd)
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->neighborData.cellStart, 0xFF, sizeof(uint32_t) * numCells, stream));
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->neighborData.cellEnd, 0xFF, sizeof(uint32_t) * numCells, stream));

    findCellBoundsKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        impl_->d_keysOut, impl_->neighborData.cellStart, impl_->neighborData.cellEnd, count, false);
    VP_CUDA_CHECK(cudaEventRecord(impl_->evBounds, stream));

    // 5. Reorder GranularPool SoA into spatial order
    reorderGranularPoolKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, pool.vel, pool.pred, pool.tint, pool.materialId, pool.flags, pool.grainId,
        impl_->d_indicesOut,
        impl_->d_scratchPos, impl_->d_scratchVel, impl_->d_scratchPred,
        impl_->d_scratchTint, impl_->d_scratchMaterialId, impl_->d_scratchFlags, impl_->d_scratchGrainId,
        count);

    // Swap pointers so pool reflects sorted order with zero copy
    std::swap(pool.pos, impl_->d_scratchPos);
    std::swap(pool.vel, impl_->d_scratchVel);
    std::swap(pool.pred, impl_->d_scratchPred);
    std::swap(pool.tint, impl_->d_scratchTint);
    std::swap(pool.materialId, impl_->d_scratchMaterialId);
    std::swap(pool.flags, impl_->d_scratchFlags);
    std::swap(pool.grainId, impl_->d_scratchGrainId);
    VP_CUDA_CHECK(cudaEventRecord(impl_->evReorder, stream));

    // 6. Build Verlet neighbor lists (32 slots)
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->neighborData.d_overflowCounter, 0, sizeof(uint32_t), stream));
    buildVerletListKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, impl_->neighborData.cellStart, impl_->neighborData.cellEnd,
        impl_->neighborData.list, impl_->neighborData.counts,
        impl_->neighborData.d_overflowCounter, impl_->gridParams);
    VP_CUDA_CHECK(cudaEventRecord(impl_->evVerlet, stream));

    // 7. Store reference positions for displacement checks
    VP_CUDA_CHECK(cudaMemcpyAsync(
        impl_->neighborData.referencePos, pool.pos, sizeof(float4) * count,
        cudaMemcpyDeviceToDevice, stream));
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->neighborData.d_rebuildNeeded, 0, sizeof(uint32_t), stream));

    // 8. Generate 8-color partition indices for Phase 11D Gauss-Seidel
    generateColorKeysKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, impl_->d_colorKeysIn, impl_->d_colorIndicesIn, impl_->gridParams);

    VP_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        impl_->d_colorSortStorage, impl_->colorSortStorageBytes,
        impl_->d_colorKeysIn, impl_->d_colorKeysOut,
        impl_->d_colorIndicesIn, impl_->neighborData.colorIndices,
        count, 0, 3, stream));

    findColorBoundsKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        impl_->d_colorKeysOut, impl_->neighborData.colorOffsets, count);

    VP_CUDA_CHECK(cudaMemcpyAsync(
        impl_->neighborData.hostColorOffsets, impl_->neighborData.colorOffsets,
        sizeof(uint32_t) * 9, cudaMemcpyDeviceToHost, stream));

    VP_CUDA_CHECK(cudaEventRecord(impl_->evEnd, stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timing metrics
    float keygenMs = 0.0f, sortMs = 0.0f, boundsMs = 0.0f;
    float reorderMs = 0.0f, verletMs = 0.0f, totalMs = 0.0f;
    VP_CUDA_CHECK(cudaEventElapsedTime(&keygenMs, impl_->evStart, impl_->evKeygen));
    VP_CUDA_CHECK(cudaEventElapsedTime(&sortMs, impl_->evKeygen, impl_->evSort));
    VP_CUDA_CHECK(cudaEventElapsedTime(&boundsMs, impl_->evSort, impl_->evBounds));
    VP_CUDA_CHECK(cudaEventElapsedTime(&reorderMs, impl_->evBounds, impl_->evReorder));
    VP_CUDA_CHECK(cudaEventElapsedTime(&verletMs, impl_->evReorder, impl_->evVerlet));
    VP_CUDA_CHECK(cudaEventElapsedTime(&totalMs, impl_->evStart, impl_->evEnd));

    uint32_t overflow = 0;
    VP_CUDA_CHECK(cudaMemcpy(&overflow, impl_->neighborData.d_overflowCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    impl_->stats.totalRebuilds++;
    impl_->stats.lastOverflowCount = overflow;
    impl_->stats.lastKeygenMs = keygenMs;
    impl_->stats.lastSortMs = sortMs;
    impl_->stats.lastBoundsMs = boundsMs;
    impl_->stats.lastReorderMs = reorderMs;
    impl_->stats.lastVerletBuildMs = verletMs;
    impl_->stats.lastRebuildMs = totalMs;
}

bool GranularNeighborSearch::checkRebuildTrigger(
    const GranularPool& pool,
    const GranularSettings& settings,
    cudaStream_t stream)
{
    const uint32_t count = pool.aliveCount;
    if (count == 0) return false;

    impl_->stats.displacementChecks++;

    const float skin = settings.skinRatio * settings.grainRadius;
    const float halfSkinSq = (0.5f * skin) * (0.5f * skin);
    const uint32_t blocks = (count + kNeighborThreadsPerBlock - 1) / kNeighborThreadsPerBlock;

    checkDisplacementKernel<<<blocks, kNeighborThreadsPerBlock, 0, stream>>>(
        pool.pos, impl_->neighborData.referencePos, impl_->neighborData.d_rebuildNeeded,
        halfSkinSq, count);

    uint32_t trigger = 0;
    VP_CUDA_CHECK(cudaMemcpyAsync(&trigger, impl_->neighborData.d_rebuildNeeded, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(stream));

    return trigger != 0;
}

void GranularNeighborSearch::resetReferencePositions(const GranularPool& pool, cudaStream_t stream)
{
    if (pool.aliveCount == 0) return;
    VP_CUDA_CHECK(cudaMemcpyAsync(
        impl_->neighborData.referencePos, pool.pos, sizeof(float4) * pool.aliveCount,
        cudaMemcpyDeviceToDevice, stream));
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->neighborData.d_rebuildNeeded, 0, sizeof(uint32_t), stream));
}

const NeighborData& GranularNeighborSearch::neighborData() const
{
    return impl_->neighborData;
}

const NeighborStats& GranularNeighborSearch::stats() const
{
    return impl_->stats;
}

const GridParams& GranularNeighborSearch::gridParams() const
{
    return impl_->gridParams;
}

} // namespace vparticles
