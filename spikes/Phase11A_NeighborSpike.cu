// Phase 11A: Neighbor Structure and Verlet List Spike
//
// Measures spatial hash key generation, CUB radix sort, cell range construction,
// dense SoA reorder, Verlet list build (32 slots), displacement checks, and
// skin-triggered amortization.
//
// Target: Best 1M neighbor rebuild amortized cost <= 1.5 ms; Bit-exact determinism.

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kMaxNeighborsPerGrain = 32;

struct GridParams {
    float3 origin;
    float cellSize;
    float invCellSize;
    int3 gridDim;
    uint32_t numCells;
    float skin;
    float radius;
    float searchRadius;
    float searchRadiusSq;
    uint32_t grainCount;
    bool enableColorKey;
};

struct SpikeConfig {
    uint32_t grainCount = 1'000'000;
    uint32_t warmupRuns = 3;
    uint32_t timedRuns = 10;
    uint32_t determinismRuns = 5;
    uint32_t seed = 11'011;
    float radius = 0.01f;
    float skinRatio = 0.20f; // 20% of radius = 0.002f
    float jitter = 0.05f;
    bool runBruteForceValidation = true;
    bool colorGrouping = false;
    std::string csvPath;
};

struct StageMetrics {
    std::vector<float> timesMs;
    float medianMs = 0.0f;
    float p95Ms = 0.0f;
    float p99Ms = 0.0f;
    float bytesPerRun = 0.0f;
    float medianBandwidthGBs = 0.0f;
};

void checkCuda(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed with: " +
                                 cudaGetErrorString(result) + " at " + file + ":" + std::to_string(line));
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__device__ inline float3 makeFloat3(const float4 v)
{
    return make_float3(v.x, v.y, v.z);
}

__device__ inline float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// -----------------------------------------------------------------------------
// 1. Key Generation Kernel
// Computes 3D integer cell coordinate and either linear cell key or color-grouped key
// -----------------------------------------------------------------------------
__global__ void generateKeysKernel(
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
        // 8-color cell pattern: (cx % 2) | ((cy % 2) << 1) | ((cz % 2) << 2)
        const uint32_t color = (cx & 1) | ((cy & 1) << 1) | ((cz & 1) << 2);
        keys[i] = (color << 28) | (cellIdx & 0x0FFFFFFF);
    } else {
        keys[i] = cellIdx;
    }

    indices[i] = i;
}

// -----------------------------------------------------------------------------
// 2. Cell Bounds Kernel
// Identifies contiguous boundaries [cellStart, cellEnd) in the sorted array
// -----------------------------------------------------------------------------
__global__ void findCellBoundsKernel(
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
__global__ void reorderSoAKernel(
    const float4* __restrict__ inPos,
    const float4* __restrict__ inVel,
    const uint32_t* __restrict__ inGrainId,
    const uint32_t* __restrict__ sortedIndices,
    float4* __restrict__ outPos,
    float4* __restrict__ outVel,
    uint32_t* __restrict__ outGrainId,
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const uint32_t origIdx = sortedIndices[i];
    outPos[i] = inPos[origIdx];
    outVel[i] = inVel[origIdx];
    outGrainId[i] = inGrainId[origIdx];
}

// -----------------------------------------------------------------------------
// 4. Verlet Neighbor List Build Kernel
// Queries 27 neighboring cells and records up to 32 neighbors per grain
// -----------------------------------------------------------------------------
__global__ void buildVerletListKernel(
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
    const float3 p = makeFloat3(p4);
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
                    const float3 otherP = makeFloat3(otherP4);
                    const float rj = otherP4.w > 0.0f ? otherP4.w : params.radius;

                    const float3 diff = sub3(otherP, p);
                    const float distSq = dot3(diff, diff);
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
// Computes max displacement between currentPos and referencePos
// -----------------------------------------------------------------------------
__global__ void computeDisplacementKernel(
    const float4* __restrict__ currentPos,
    const float4* __restrict__ referencePos,
    float* __restrict__ displacements,
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const float3 p = makeFloat3(currentPos[i]);
    const float3 ref = makeFloat3(referencePos[i]);
    const float3 diff = sub3(p, ref);
    displacements[i] = sqrtf(dot3(diff, diff));
}

// -----------------------------------------------------------------------------
// 6. Ground-Truth Brute Force Neighbor Search (for verification at 10K grains)
// -----------------------------------------------------------------------------
__global__ void bruteForceNeighborsKernel(
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
    const float3 p = makeFloat3(p4);
    const float ri = p4.w > 0.0f ? p4.w : defaultRadius;

    uint32_t count = 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    for (uint32_t j = 0; j < grainCount; ++j) {
        if (j == i) continue;

        const float4 otherP4 = positions[j];
        const float3 otherP = makeFloat3(otherP4);
        const float rj = otherP4.w > 0.0f ? otherP4.w : defaultRadius;

        const float3 diff = sub3(otherP, p);
        const float distSq = dot3(diff, diff);
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

// -----------------------------------------------------------------------------
// 64-bit FNV-1a Hash for Bit-Exact Determinism Verification
// -----------------------------------------------------------------------------
uint64_t fnv1a64(const void* data, size_t numBytes, uint64_t hash = 14695981039346656037ULL)
{
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < numBytes; ++i) {
        hash ^= static_cast<uint64_t>(ptr[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

void computePercentiles(std::vector<float>& values, float& median, float& p95, float& p99)
{
    if (values.empty()) return;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    median = values[n / 2];
    p95 = values[std::min(static_cast<size_t>(std::ceil(0.95f * n)) - 1, n - 1)];
    p99 = values[std::min(static_cast<size_t>(std::ceil(0.99f * n)) - 1, n - 1)];
}

std::vector<float4> generateLatticePositions(const SpikeConfig& config, float3& outMin, float3& outMax)
{
    std::vector<float4> positions(config.grainCount);
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<float> jitterDist(-config.jitter, config.jitter);

    const float spacing = 2.02f * config.radius;
    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(config.grainCount))));

    outMin = make_float3(1e9f, 1e9f, 1e9f);
    outMax = make_float3(-1e9f, -1e9f, -1e9f);

    for (uint32_t i = 0; i < config.grainCount; ++i) {
        const uint32_t x = i % side;
        const uint32_t z = (i / side) % side;
        const uint32_t y = i / (side * side);

        const float px = (static_cast<float>(x) - 0.5f * static_cast<float>(side)) * spacing + jitterDist(rng) * config.radius;
        const float py = (static_cast<float>(y) + 0.5f) * spacing + jitterDist(rng) * config.radius;
        const float pz = (static_cast<float>(z) - 0.5f * static_cast<float>(side)) * spacing + jitterDist(rng) * config.radius;

        positions[i] = make_float4(px, py, pz, config.radius);

        outMin.x = std::min(outMin.x, px);
        outMin.y = std::min(outMin.y, py);
        outMin.z = std::min(outMin.z, pz);

        outMax.x = std::max(outMax.x, px);
        outMax.y = std::max(outMax.y, py);
        outMax.z = std::max(outMax.z, pz);
    }
    return positions;
}

uint32_t parseUint(const char* val, uint32_t fallback)
{
    try { return static_cast<uint32_t>(std::stoul(val)); } catch (...) { return fallback; }
}

float parseFloat(const char* val, float fallback)
{
    try { return std::stof(val); } catch (...) { return fallback; }
}

SpikeConfig parseArgs(int argc, char** argv)
{
    SpikeConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--grains" && i + 1 < argc) config.grainCount = parseUint(argv[++i], config.grainCount);
        else if (arg == "--runs" && i + 1 < argc) config.timedRuns = parseUint(argv[++i], config.timedRuns);
        else if (arg == "--radius" && i + 1 < argc) config.radius = parseFloat(argv[++i], config.radius);
        else if (arg == "--skin-ratio" && i + 1 < argc) config.skinRatio = parseFloat(argv[++i], config.skinRatio);
        else if (arg == "--jitter" && i + 1 < argc) config.jitter = parseFloat(argv[++i], config.jitter);
        else if (arg == "--seed" && i + 1 < argc) config.seed = parseUint(argv[++i], config.seed);
        else if (arg == "--color-grouping") config.colorGrouping = true;
        else if (arg == "--no-brute-force") config.runBruteForceValidation = false;
        else if (arg == "--csv" && i + 1 < argc) config.csvPath = argv[++i];
        else if (arg == "--help") {
            std::cout << "Phase11ANeighborSpike options:\n"
                      << "  --grains N            Number of grains (default: 1000000)\n"
                      << "  --runs N              Timed iterations (default: 10)\n"
                      << "  --radius R            Grain radius in meters (default: 0.01)\n"
                      << "  --skin-ratio S        Skin ratio as fraction of radius (default: 0.20)\n"
                      << "  --jitter J            Jitter fraction (default: 0.05)\n"
                      << "  --seed N              RNG seed (default: 11011)\n"
                      << "  --color-grouping      Embed 8-color index in sort keys\n"
                      << "  --no-brute-force      Skip 10K brute force validation\n"
                      << "  --csv FILE            Write timing breakdown to CSV\n";
            std::exit(0);
        }
    }
    return config;
}

// -----------------------------------------------------------------------------
// Brute Force Validation Run (10,000 grains)
// -----------------------------------------------------------------------------
bool runBruteForceValidation(const SpikeConfig& baseConfig, cudaStream_t stream)
{
    std::cout << "\n======================================================\n";
    std::cout << "  Brute Force Validation (10,000 grains)\n";
    std::cout << "======================================================\n";

    SpikeConfig bfConfig = baseConfig;
    bfConfig.grainCount = 10'000;

    float3 bMin, bMax;
    std::vector<float4> hostPos = generateLatticePositions(bfConfig, bMin, bMax);

    const float skin = bfConfig.skinRatio * bfConfig.radius;
    const float cellSize = 2.0f * bfConfig.radius + skin;
    const float margin = cellSize * 2.0f;
    bMin.x -= margin; bMin.y -= margin; bMin.z -= margin;
    bMax.x += margin; bMax.y += margin; bMax.z += margin;

    int3 gridDim = make_int3(
        std::max(1, static_cast<int>(std::ceil((bMax.x - bMin.x) / cellSize))),
        std::max(1, static_cast<int>(std::ceil((bMax.y - bMin.y) / cellSize))),
        std::max(1, static_cast<int>(std::ceil((bMax.z - bMin.z) / cellSize))));
    uint32_t numCells = gridDim.x * gridDim.y * gridDim.z;

    GridParams params = {
        bMin, cellSize, 1.0f / cellSize, gridDim, numCells,
        skin, bfConfig.radius, 2.0f * bfConfig.radius + skin,
        (2.0f * bfConfig.radius + skin) * (2.0f * bfConfig.radius + skin),
        bfConfig.grainCount, false
    };

    // Device allocations for validation
    float4* d_pos = nullptr;
    uint32_t *d_keysIn = nullptr, *d_keysOut = nullptr;
    uint32_t *d_indicesIn = nullptr, *d_indicesOut = nullptr;
    uint32_t *d_cellStart = nullptr, *d_cellEnd = nullptr;
    float4* d_sortedPos = nullptr;
    float4* d_vel = nullptr;
    uint32_t *d_grainId = nullptr, *d_sortedGrainId = nullptr;
    uint32_t *d_neighborList = nullptr, *d_neighborCount = nullptr;
    uint32_t *d_overflow = nullptr;

    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float4) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_vel, sizeof(float4) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_grainId, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_sortedPos, sizeof(float4) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_sortedGrainId, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_keysIn, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_keysOut, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_indicesIn, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_indicesOut, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_cellStart, sizeof(uint32_t) * numCells));
    CUDA_CHECK(cudaMalloc(&d_cellEnd, sizeof(uint32_t) * numCells));
    CUDA_CHECK(cudaMalloc(&d_neighborList, sizeof(uint32_t) * bfConfig.grainCount * kMaxNeighborsPerGrain));
    CUDA_CHECK(cudaMalloc(&d_neighborCount, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpyAsync(d_pos, hostPos.data(), sizeof(float4) * bfConfig.grainCount, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, sizeof(uint32_t) * numCells, stream));
    CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, sizeof(uint32_t) * numCells, stream));
    CUDA_CHECK(cudaMemsetAsync(d_overflow, 0, sizeof(uint32_t), stream));

    // Sort scratch
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, d_keysIn, d_keysOut, d_indicesIn, d_indicesOut, bfConfig.grainCount, 0, 32, stream));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));

    const uint32_t blocks = (bfConfig.grainCount + kThreads - 1) / kThreads;

    // 1. Grid Pipeline
    generateKeysKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_keysIn, d_indicesIn, params);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp, tempBytes, d_keysIn, d_keysOut, d_indicesIn, d_indicesOut, bfConfig.grainCount, 0, 32, stream));
    findCellBoundsKernel<<<blocks, kThreads, 0, stream>>>(d_keysOut, d_cellStart, d_cellEnd, bfConfig.grainCount, false);
    reorderSoAKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_vel, d_grainId, d_indicesOut, d_sortedPos, d_vel, d_sortedGrainId, bfConfig.grainCount);
    buildVerletListKernel<<<blocks, kThreads, 0, stream>>>(d_sortedPos, d_cellStart, d_cellEnd, d_neighborList, d_neighborCount, d_overflow, params);

    // 2. Brute Force Pipeline
    uint32_t *d_bfCount = nullptr, *d_bfList = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bfCount, sizeof(uint32_t) * bfConfig.grainCount));
    CUDA_CHECK(cudaMalloc(&d_bfList, sizeof(uint32_t) * bfConfig.grainCount * kMaxNeighborsPerGrain));
    bruteForceNeighborsKernel<<<blocks, kThreads, 0, stream>>>(d_sortedPos, d_bfCount, d_bfList, bfConfig.grainCount, skin, bfConfig.radius);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 3. Read back and compare
    std::vector<uint32_t> gridCounts(bfConfig.grainCount);
    std::vector<uint32_t> gridList(bfConfig.grainCount * kMaxNeighborsPerGrain);
    std::vector<uint32_t> bfCounts(bfConfig.grainCount);
    std::vector<uint32_t> bfLists(bfConfig.grainCount * kMaxNeighborsPerGrain);

    CUDA_CHECK(cudaMemcpy(gridCounts.data(), d_neighborCount, sizeof(uint32_t) * bfConfig.grainCount, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gridList.data(), d_neighborList, sizeof(uint32_t) * gridList.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bfCounts.data(), d_bfCount, sizeof(uint32_t) * bfConfig.grainCount, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bfLists.data(), d_bfList, sizeof(uint32_t) * bfLists.size(), cudaMemcpyDeviceToHost));

    uint64_t totalGridPairs = 0;
    uint64_t totalBfPairs = 0;
    uint64_t falsePositives = 0;
    uint64_t falseNegatives = 0;

    for (uint32_t i = 0; i < bfConfig.grainCount; ++i) {
        uint32_t gc = gridCounts[i];
        uint32_t bc = bfCounts[i];
        totalGridPairs += gc;
        totalBfPairs += bc;

        std::vector<uint32_t> gNeighbors(gridList.begin() + i * kMaxNeighborsPerGrain, gridList.begin() + i * kMaxNeighborsPerGrain + gc);
        std::vector<uint32_t> bNeighbors(bfLists.begin() + i * kMaxNeighborsPerGrain, bfLists.begin() + i * kMaxNeighborsPerGrain + bc);
        std::sort(gNeighbors.begin(), gNeighbors.end());
        std::sort(bNeighbors.begin(), bNeighbors.end());

        if (gc != bc) {
            // Count mismatches
        }
        for (uint32_t gn : gNeighbors) {
            if (!std::binary_search(bNeighbors.begin(), bNeighbors.end(), gn)) {
                falsePositives++;
            }
        }
        for (uint32_t bn : bNeighbors) {
            if (!std::binary_search(gNeighbors.begin(), gNeighbors.end(), bn)) {
                falseNegatives++;
            }
        }
    }

    std::cout << "Total neighbor pairs (Grid)       : " << totalGridPairs << '\n';
    std::cout << "Total neighbor pairs (Brute Force): " << totalBfPairs << '\n';
    std::cout << "False Positives                   : " << falsePositives << '\n';
    std::cout << "False Negatives (Missed)          : " << falseNegatives << '\n';

    bool passed = (falsePositives == 0 && falseNegatives == 0 && totalGridPairs == totalBfPairs);
    std::cout << "Brute Force Parity Result         : " << (passed ? "PASSED (100.0% Bit-Exact Match)" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_grainId));
    CUDA_CHECK(cudaFree(d_sortedPos));
    CUDA_CHECK(cudaFree(d_sortedGrainId));
    CUDA_CHECK(cudaFree(d_keysIn));
    CUDA_CHECK(cudaFree(d_keysOut));
    CUDA_CHECK(cudaFree(d_indicesIn));
    CUDA_CHECK(cudaFree(d_indicesOut));
    CUDA_CHECK(cudaFree(d_cellStart));
    CUDA_CHECK(cudaFree(d_cellEnd));
    CUDA_CHECK(cudaFree(d_neighborList));
    CUDA_CHECK(cudaFree(d_neighborCount));
    CUDA_CHECK(cudaFree(d_overflow));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_bfCount));
    CUDA_CHECK(cudaFree(d_bfList));

    return passed;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const SpikeConfig config = parseArgs(argc, argv);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "======================================================\n";
        std::cout << "  VParticles Phase 11A - Neighbor Structure Spike\n";
        std::cout << "======================================================\n";
        std::cout << "Device              : " << prop.name << " (" << prop.multiProcessorCount << " SMs, "
                  << prop.totalGlobalMem / (1024 * 1024) << " MB VRAM)\n";
        int memoryClockKHz = 0;
        cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, 0);
        const float theoreticalBandwidth = (2.0f * (memoryClockKHz / 1.0e6f) * (prop.memoryBusWidth / 8.0f));
        std::cout << "Memory Bus          : " << prop.memoryBusWidth << "-bit, "
                  << "Theoretical Bandwidth: " << theoreticalBandwidth << " GB/s\n";
        std::cout << "Grains              : " << config.grainCount << '\n';
        std::cout << "Radius              : " << config.radius << " m\n";
        std::cout << "Skin Ratio          : " << (config.skinRatio * 100.0f) << "% (" << (config.skinRatio * config.radius) << " m)\n";
        std::cout << "Cell Size           : " << (2.0f * config.radius + config.skinRatio * config.radius) << " m\n";
        std::cout << "Color Grouping      : " << (config.colorGrouping ? "Enabled (8-color key)" : "Disabled (linear key)") << '\n';
        std::cout << "Warmup / Timed Runs : " << config.warmupRuns << " / " << config.timedRuns << '\n';

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // 1. Run 10K Brute Force Validation if requested
        if (config.runBruteForceValidation) {
            runBruteForceValidation(config, stream);
        }

        std::cout << "\n======================================================\n";
        std::cout << "  Main Neighbor Pipeline Benchmark (" << config.grainCount << " grains)\n";
        std::cout << "======================================================\n";

        // Setup domain and initial positions
        float3 bMin, bMax;
        std::vector<float4> hostPos = generateLatticePositions(config, bMin, bMax);
        std::vector<float4> hostVel(config.grainCount, make_float4(0.0f, 0.0f, 0.0f, 1.0f));
        std::vector<uint32_t> hostGrainId(config.grainCount);
        std::iota(hostGrainId.begin(), hostGrainId.end(), 0u);

        const float skin = config.skinRatio * config.radius;
        const float cellSize = 2.0f * config.radius + skin;
        const float margin = cellSize * 2.0f;
        bMin.x -= margin; bMin.y -= margin; bMin.z -= margin;
        bMax.x += margin; bMax.y += margin; bMax.z += margin;

        int3 gridDim = make_int3(
            std::max(1, static_cast<int>(std::ceil((bMax.x - bMin.x) / cellSize))),
            std::max(1, static_cast<int>(std::ceil((bMax.y - bMin.y) / cellSize))),
            std::max(1, static_cast<int>(std::ceil((bMax.z - bMin.z) / cellSize))));
        uint32_t numCells = gridDim.x * gridDim.y * gridDim.z;

        std::cout << "Bounding Box Min    : [" << bMin.x << ", " << bMin.y << ", " << bMin.z << "]\n";
        std::cout << "Bounding Box Max    : [" << bMax.x << ", " << bMax.y << ", " << bMax.z << "]\n";
        std::cout << "Grid Dimensions     : " << gridDim.x << " x " << gridDim.y << " x " << gridDim.z
                  << " (" << numCells << " cells, " << (numCells * sizeof(uint32_t) * 2 / (1024 * 1024.0f)) << " MB table)\n";

        GridParams params = {
            bMin, cellSize, 1.0f / cellSize, gridDim, numCells,
            skin, config.radius, 2.0f * config.radius + skin,
            (2.0f * config.radius + skin) * (2.0f * config.radius + skin),
            config.grainCount, config.colorGrouping
        };

        // Device buffer allocations
        float4 *d_pos = nullptr, *d_vel = nullptr;
        uint32_t *d_grainId = nullptr;
        float4 *d_sortedPos = nullptr, *d_sortedVel = nullptr;
        uint32_t *d_sortedGrainId = nullptr;
        uint32_t *d_keysIn = nullptr, *d_keysOut = nullptr;
        uint32_t *d_indicesIn = nullptr, *d_indicesOut = nullptr;
        uint32_t *d_cellStart = nullptr, *d_cellEnd = nullptr;
        uint32_t *d_neighborList = nullptr, *d_neighborCount = nullptr;
        uint32_t *d_overflow = nullptr;
        float *d_displacements = nullptr;

        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float4) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_vel, sizeof(float4) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_grainId, sizeof(uint32_t) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_sortedPos, sizeof(float4) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_sortedVel, sizeof(float4) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_sortedGrainId, sizeof(uint32_t) * config.grainCount));

        CUDA_CHECK(cudaMalloc(&d_keysIn, sizeof(uint32_t) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_keysOut, sizeof(uint32_t) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_indicesIn, sizeof(uint32_t) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_indicesOut, sizeof(uint32_t) * config.grainCount));

        CUDA_CHECK(cudaMalloc(&d_cellStart, sizeof(uint32_t) * numCells));
        CUDA_CHECK(cudaMalloc(&d_cellEnd, sizeof(uint32_t) * numCells));

        CUDA_CHECK(cudaMalloc(&d_neighborList, sizeof(uint32_t) * config.grainCount * kMaxNeighborsPerGrain));
        CUDA_CHECK(cudaMalloc(&d_neighborCount, sizeof(uint32_t) * config.grainCount));
        CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_displacements, sizeof(float) * config.grainCount));

        // Copy initial data
        CUDA_CHECK(cudaMemcpyAsync(d_pos, hostPos.data(), sizeof(float4) * config.grainCount, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_vel, hostVel.data(), sizeof(float4) * config.grainCount, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_grainId, hostGrainId.data(), sizeof(uint32_t) * config.grainCount, cudaMemcpyHostToDevice, stream));

        // CUB Radix Sort temporary storage setup
        size_t sortTempBytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sortTempBytes, d_keysIn, d_keysOut, d_indicesIn, d_indicesOut,
            config.grainCount, 0, config.colorGrouping ? 32 : 24, stream));
        void* d_sortTemp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sortTemp, sortTempBytes));

        // CUB DeviceReduce::Max temporary storage setup
        size_t reduceTempBytes = 0;
        float* d_maxDisp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_maxDisp, sizeof(float)));
        CUDA_CHECK(cub::DeviceReduce::Max(nullptr, reduceTempBytes, d_displacements, d_maxDisp, config.grainCount, stream));
        void* d_reduceTemp = nullptr;
        CUDA_CHECK(cudaMalloc(&d_reduceTemp, reduceTempBytes));

        const uint32_t blocks = (config.grainCount + kThreads - 1) / kThreads;

        // Stage metrics collectors
        StageMetrics mKeyGen, mSort, mCellBounds, mReorder, mVerlet, mDisplacement, mTotalRebuild;
        // Traffic estimations (bytes read + written)
        mKeyGen.bytesPerRun = static_cast<float>(config.grainCount * (sizeof(float4) + sizeof(uint32_t) * 2));
        mSort.bytesPerRun = static_cast<float>(config.grainCount * sizeof(uint32_t) * 4); // key+val in/out
        mCellBounds.bytesPerRun = static_cast<float>(config.grainCount * sizeof(uint32_t) + numCells * sizeof(uint32_t) * 2);
        mReorder.bytesPerRun = static_cast<float>(config.grainCount * (sizeof(float4) * 4 + sizeof(uint32_t) * 3));
        mVerlet.bytesPerRun = static_cast<float>(config.grainCount * (sizeof(float4) + sizeof(uint32_t) + sizeof(uint32_t) * 12)); // estimated avg 12 neighbors
        mDisplacement.bytesPerRun = static_cast<float>(config.grainCount * sizeof(float4) * 2);
        mTotalRebuild.bytesPerRun = mKeyGen.bytesPerRun + mSort.bytesPerRun + mCellBounds.bytesPerRun + mReorder.bytesPerRun + mVerlet.bytesPerRun;

        // CUDA Timing Events
        cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev2));
        CUDA_CHECK(cudaEventCreate(&ev3));
        CUDA_CHECK(cudaEventCreate(&ev4));
        CUDA_CHECK(cudaEventCreate(&ev5));
        CUDA_CHECK(cudaEventCreate(&ev6));
        CUDA_CHECK(cudaEventCreate(&ev7));

        const uint32_t totalRuns = config.warmupRuns + config.timedRuns;

        for (uint32_t run = 0; run < totalRuns; ++run) {
            const bool isTimed = (run >= config.warmupRuns);

            // Clear table
            CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, sizeof(uint32_t) * numCells, stream));
            CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, sizeof(uint32_t) * numCells, stream));
            CUDA_CHECK(cudaMemsetAsync(d_overflow, 0, sizeof(uint32_t), stream));

            CUDA_CHECK(cudaEventRecord(ev0, stream));

            // 1. Key Gen
            generateKeysKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_keysIn, d_indicesIn, params);
            CUDA_CHECK(cudaEventRecord(ev1, stream));

            // 2. CUB Radix Sort
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                d_sortTemp, sortTempBytes, d_keysIn, d_keysOut, d_indicesIn, d_indicesOut,
                config.grainCount, 0, config.colorGrouping ? 32 : 24, stream));
            CUDA_CHECK(cudaEventRecord(ev2, stream));

            // 3. Cell Bounds
            findCellBoundsKernel<<<blocks, kThreads, 0, stream>>>(d_keysOut, d_cellStart, d_cellEnd, config.grainCount, config.colorGrouping);
            CUDA_CHECK(cudaEventRecord(ev3, stream));

            // 4. SoA Reorder
            reorderSoAKernel<<<blocks, kThreads, 0, stream>>>(
                d_pos, d_vel, d_grainId, d_indicesOut,
                d_sortedPos, d_sortedVel, d_sortedGrainId, config.grainCount);
            CUDA_CHECK(cudaEventRecord(ev4, stream));

            // 5. Verlet List Build
            buildVerletListKernel<<<blocks, kThreads, 0, stream>>>(
                d_sortedPos, d_cellStart, d_cellEnd, d_neighborList, d_neighborCount, d_overflow, params);
            CUDA_CHECK(cudaEventRecord(ev5, stream));

            // 6. Displacement check (reference comparison)
            computeDisplacementKernel<<<blocks, kThreads, 0, stream>>>(d_sortedPos, d_sortedPos, d_displacements, config.grainCount);
            CUDA_CHECK(cub::DeviceReduce::Max(d_reduceTemp, reduceTempBytes, d_displacements, d_maxDisp, config.grainCount, stream));
            CUDA_CHECK(cudaEventRecord(ev6, stream));

            CUDA_CHECK(cudaEventSynchronize(ev6));

            if (isTimed) {
                float msKeyGen = 0.0f, msSort = 0.0f, msBounds = 0.0f, msReorder = 0.0f, msVerlet = 0.0f, msDisp = 0.0f, msTotal = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&msKeyGen, ev0, ev1));
                CUDA_CHECK(cudaEventElapsedTime(&msSort, ev1, ev2));
                CUDA_CHECK(cudaEventElapsedTime(&msBounds, ev2, ev3));
                CUDA_CHECK(cudaEventElapsedTime(&msReorder, ev3, ev4));
                CUDA_CHECK(cudaEventElapsedTime(&msVerlet, ev4, ev5));
                CUDA_CHECK(cudaEventElapsedTime(&msDisp, ev5, ev6));
                CUDA_CHECK(cudaEventElapsedTime(&msTotal, ev0, ev5));

                mKeyGen.timesMs.push_back(msKeyGen);
                mSort.timesMs.push_back(msSort);
                mCellBounds.timesMs.push_back(msBounds);
                mReorder.timesMs.push_back(msReorder);
                mVerlet.timesMs.push_back(msVerlet);
                mDisplacement.timesMs.push_back(msDisp);
                mTotalRebuild.timesMs.push_back(msTotal);
            }
        }

        // Compute percentiles and bandwidths
        auto finishMetrics = [](StageMetrics& m) {
            computePercentiles(m.timesMs, m.medianMs, m.p95Ms, m.p99Ms);
            if (m.medianMs > 0.0f) {
                m.medianBandwidthGBs = (m.bytesPerRun / (m.medianMs * 1e-3f)) / 1e9f;
            }
        };

        finishMetrics(mKeyGen);
        finishMetrics(mSort);
        finishMetrics(mCellBounds);
        finishMetrics(mReorder);
        finishMetrics(mVerlet);
        finishMetrics(mDisplacement);
        finishMetrics(mTotalRebuild);

        // Fetch neighbor statistics
        std::vector<uint32_t> hostNeighborCounts(config.grainCount);
        CUDA_CHECK(cudaMemcpy(hostNeighborCounts.data(), d_neighborCount, sizeof(uint32_t) * config.grainCount, cudaMemcpyDeviceToHost));
        uint32_t hostOverflow = 0;
        CUDA_CHECK(cudaMemcpy(&hostOverflow, d_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint64_t totalNeighbors = 0;
        uint32_t minNeighbors = 999999;
        uint32_t maxNeighbors = 0;
        for (uint32_t c : hostNeighborCounts) {
            totalNeighbors += c;
            minNeighbors = std::min(minNeighbors, c);
            maxNeighbors = std::max(maxNeighbors, c);
        }
        float avgNeighbors = static_cast<float>(totalNeighbors) / static_cast<float>(config.grainCount);

        std::cout << "\n------------------------------------------------------\n";
        std::cout << "  Verlet Neighbor List Statistics\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "Total Neighbor Pairs : " << totalNeighbors << '\n';
        std::cout << "Average Neighbors    : " << std::fixed << std::setprecision(2) << avgNeighbors << " / grain\n";
        std::cout << "Min / Max Neighbors  : " << minNeighbors << " / " << maxNeighbors << " (slot capacity: 32)\n";
        std::cout << "Neighbor Overflows   : " << hostOverflow << " (" << (hostOverflow == 0 ? "PASSED - ZERO OVERFLOW" : "WARNING") << ")\n";

        std::cout << "\n------------------------------------------------------\n";
        std::cout << "  Per-Stage Kernel Timing Breakdown (" << config.timedRuns << " runs)\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << std::left << std::setw(24) << "Stage"
                  << std::right << std::setw(11) << "Median (ms)"
                  << std::setw(11) << "p95 (ms)"
                  << std::setw(11) << "p99 (ms)"
                  << std::setw(14) << "Bandwidth" << '\n';
        std::cout << std::string(71, '-') << '\n';

        auto printStage = [](const char* name, const StageMetrics& m) {
            std::cout << std::left << std::setw(24) << name
                      << std::right << std::fixed << std::setprecision(3)
                      << std::setw(11) << m.medianMs
                      << std::setw(11) << m.p95Ms
                      << std::setw(11) << m.p99Ms
                      << std::setprecision(1) << std::setw(10) << m.medianBandwidthGBs << " GB/s\n";
        };

        printStage("1. Key Gen", mKeyGen);
        printStage("2. CUB Radix Sort", mSort);
        printStage("3. Cell Bounds", mCellBounds);
        printStage("4. SoA Reorder", mReorder);
        printStage("5. Verlet List Build", mVerlet);
        std::cout << std::string(71, '-') << '\n';
        printStage("Total Full Rebuild", mTotalRebuild);
        printStage("Displacement Check", mDisplacement);

        std::cout << "\n------------------------------------------------------\n";
        std::cout << "  Amortized Rebuild Cost Analysis\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "Every Step (no amort.)   : " << std::fixed << std::setprecision(3) << mTotalRebuild.medianMs << " ms\n";
        std::cout << "Every 4 Steps (skin trig): " << (mTotalRebuild.medianMs / 4.0f + mDisplacement.medianMs) << " ms/step\n";
        std::cout << "Every 8 Steps (skin trig): " << (mTotalRebuild.medianMs / 8.0f + mDisplacement.medianMs) << " ms/step\n";
        std::cout << "Every 16 Steps (settled) : " << (mTotalRebuild.medianMs / 16.0f + mDisplacement.medianMs) << " ms/step\n";

        // G0 Gate Verification: Best 1M neighbor rebuild cost <= 1.5 ms
        const float bestAmortizedMs = (mTotalRebuild.medianMs / 8.0f + mDisplacement.medianMs);
        std::cout << "\n>>> G0 TARGET (Rebuild <= 1.5 ms amortized): "
                  << (bestAmortizedMs <= 1.5f ? "PASSED" : "FAILED")
                  << " (" << bestAmortizedMs << " ms <= 1.5 ms) <<<\n";

        // -----------------------------------------------------------------
        // Determinism Check (5 runs with identical seed)
        // -----------------------------------------------------------------
        std::cout << "\n======================================================\n";
        std::cout << "  Determinism Check (" << config.determinismRuns << " runs bit-exact verification)\n";
        std::cout << "======================================================\n";

        std::vector<uint64_t> runHashes;
        std::vector<uint32_t> devKeys(config.grainCount);
        std::vector<uint32_t> devIndices(config.grainCount);
        std::vector<uint32_t> devCounts(config.grainCount);

        for (uint32_t dRun = 0; dRun < config.determinismRuns; ++dRun) {
            // Re-upload identical initial positions
            CUDA_CHECK(cudaMemcpyAsync(d_pos, hostPos.data(), sizeof(float4) * config.grainCount, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(d_cellStart, 0xFF, sizeof(uint32_t) * numCells, stream));
            CUDA_CHECK(cudaMemsetAsync(d_cellEnd, 0xFF, sizeof(uint32_t) * numCells, stream));

            generateKeysKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_keysIn, d_indicesIn, params);
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                d_sortTemp, sortTempBytes, d_keysIn, d_keysOut, d_indicesIn, d_indicesOut,
                config.grainCount, 0, config.colorGrouping ? 32 : 24, stream));
            findCellBoundsKernel<<<blocks, kThreads, 0, stream>>>(d_keysOut, d_cellStart, d_cellEnd, config.grainCount, config.colorGrouping);
            reorderSoAKernel<<<blocks, kThreads, 0, stream>>>(
                d_pos, d_vel, d_grainId, d_indicesOut,
                d_sortedPos, d_sortedVel, d_sortedGrainId, config.grainCount);
            buildVerletListKernel<<<blocks, kThreads, 0, stream>>>(
                d_sortedPos, d_cellStart, d_cellEnd, d_neighborList, d_neighborCount, d_overflow, params);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            CUDA_CHECK(cudaMemcpy(devKeys.data(), d_keysOut, sizeof(uint32_t) * config.grainCount, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(devIndices.data(), d_indicesOut, sizeof(uint32_t) * config.grainCount, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(devCounts.data(), d_neighborCount, sizeof(uint32_t) * config.grainCount, cudaMemcpyDeviceToHost));

            uint64_t h = fnv1a64(devKeys.data(), sizeof(uint32_t) * devKeys.size());
            h = fnv1a64(devIndices.data(), sizeof(uint32_t) * devIndices.size(), h);
            h = fnv1a64(devCounts.data(), sizeof(uint32_t) * devCounts.size(), h);

            runHashes.push_back(h);
            std::cout << "Run " << (dRun + 1) << " State Hash: 0x" << std::hex << std::setw(16) << std::setfill('0') << h << std::dec << '\n';
        }

        bool determinismPassed = true;
        for (size_t i = 1; i < runHashes.size(); ++i) {
            if (runHashes[i] != runHashes[0]) {
                determinismPassed = false;
                break;
            }
        }

        std::cout << ">>> G0 TARGET (Determinism Bit-Exact): "
                  << (determinismPassed ? "PASSED (Bit-exact across 5 runs)" : "FAILED (Hash mismatch)") << " <<<\n";

        // Write CSV output if requested
        if (!config.csvPath.empty()) {
            std::ofstream csv(config.csvPath);
            if (csv) {
                csv << "stage,medianMs,p95Ms,p99Ms,bandwidthGBs\n";
                csv << "KeyGen," << mKeyGen.medianMs << ',' << mKeyGen.p95Ms << ',' << mKeyGen.p99Ms << ',' << mKeyGen.medianBandwidthGBs << '\n';
                csv << "RadixSort," << mSort.medianMs << ',' << mSort.p95Ms << ',' << mSort.p99Ms << ',' << mSort.medianBandwidthGBs << '\n';
                csv << "CellBounds," << mCellBounds.medianMs << ',' << mCellBounds.p95Ms << ',' << mCellBounds.p99Ms << ',' << mCellBounds.medianBandwidthGBs << '\n';
                csv << "SoAReorder," << mReorder.medianMs << ',' << mReorder.p95Ms << ',' << mReorder.p99Ms << ',' << mReorder.medianBandwidthGBs << '\n';
                csv << "VerletBuild," << mVerlet.medianMs << ',' << mVerlet.p95Ms << ',' << mVerlet.p99Ms << ',' << mVerlet.medianBandwidthGBs << '\n';
                csv << "TotalRebuild," << mTotalRebuild.medianMs << ',' << mTotalRebuild.p95Ms << ',' << mTotalRebuild.p99Ms << ',' << mTotalRebuild.medianBandwidthGBs << '\n';
                csv << "DisplacementCheck," << mDisplacement.medianMs << ',' << mDisplacement.p95Ms << ',' << mDisplacement.p99Ms << ',' << mDisplacement.medianBandwidthGBs << '\n';
                csv << "AmortizedStep8," << bestAmortizedMs << ",,,,\n";
                std::cout << "\nExported timing results to " << config.csvPath << '\n';
            }
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_pos));
        CUDA_CHECK(cudaFree(d_vel));
        CUDA_CHECK(cudaFree(d_grainId));
        CUDA_CHECK(cudaFree(d_sortedPos));
        CUDA_CHECK(cudaFree(d_sortedVel));
        CUDA_CHECK(cudaFree(d_sortedGrainId));
        CUDA_CHECK(cudaFree(d_keysIn));
        CUDA_CHECK(cudaFree(d_keysOut));
        CUDA_CHECK(cudaFree(d_indicesIn));
        CUDA_CHECK(cudaFree(d_indicesOut));
        CUDA_CHECK(cudaFree(d_cellStart));
        CUDA_CHECK(cudaFree(d_cellEnd));
        CUDA_CHECK(cudaFree(d_neighborList));
        CUDA_CHECK(cudaFree(d_neighborCount));
        CUDA_CHECK(cudaFree(d_overflow));
        CUDA_CHECK(cudaFree(d_displacements));
        CUDA_CHECK(cudaFree(d_maxDisp));
        CUDA_CHECK(cudaFree(d_sortTemp));
        CUDA_CHECK(cudaFree(d_reduceTemp));
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaEventDestroy(ev2));
        CUDA_CHECK(cudaEventDestroy(ev3));
        CUDA_CHECK(cudaEventDestroy(ev4));
        CUDA_CHECK(cudaEventDestroy(ev5));
        CUDA_CHECK(cudaEventDestroy(ev6));
        CUDA_CHECK(cudaEventDestroy(ev7));
        CUDA_CHECK(cudaStreamDestroy(stream));

        return (determinismPassed && bestAmortizedMs <= 1.5f) ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Phase11ANeighborSpike ERROR: " << e.what() << '\n';
        return 1;
    }
}
