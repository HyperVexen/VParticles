// Phase 11A: Contact Solver Spike (XPBD Jacobi vs Cell-Colored Gauss-Seidel)
//
// Evaluates:
// 1. Solve pass latency and DRAM bandwidth at 250K and 1M grains (Targets: <1.0 ms @ 250K, <3.5 ms @ 1M)
// 2. 100K grain stack compression vs solver passes (Target: <5% compression with <= 8 passes)
// 3. 50-grain vertical vibration transmission to top layer (Target: Yes)
// 4. Settling pour awake/asleep scaling (Target: >80% asleep after settling)
//
// Hardware baseline: RTX 3050 Laptop GPU (20 SMs, 168 GB/s bandwidth)

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
};

struct SolverParams {
    uint32_t grainCount;
    float dt;              // substep dt
    float invDt;
    float radius;
    float invMass;
    float gravity;
    float groundY;
    float compliance;      // XPBD compliance (0 = rigid)
    float friction;
    float restitution;
    float sleepSpeedSq;    // (0.02 m/s)^2
    uint32_t sleepFrames;  // 30 frames below threshold to sleep
    float vibrationAmp;
    float vibrationFreq;
    float simTime;
    float boxHalfWidth;
};

enum class SolverMode {
    Jacobi,
    ColoredGaussSeidel
};

struct SpikeConfig {
    std::string scenario = "all"; // "bench", "compression", "vibration", "pour", "all"
    uint32_t grainCount = 250'000;
    uint32_t runs = 10;
    uint32_t substeps = 4;
    uint32_t iterations = 4;
    float radius = 0.01f;
    float skinRatio = 0.20f;
    uint32_t seed = 11'011;
    std::string csvPath;
};

void checkCuda(cudaError_t result, const char* expression, const char* file, int line)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + " failed: " +
                                 cudaGetErrorString(result) + " at " + file + ":" + std::to_string(line));
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul3(const float3 v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }

// -----------------------------------------------------------------------------
// Predict Kernel: Apply external forces and predict unconstrained position x*
// -----------------------------------------------------------------------------
__global__ void predictKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* __restrict__ pred,
    const uint32_t* __restrict__ sleepState,
    SolverParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    if (sleepState && sleepState[i] >= params.sleepFrames) {
        // Grain is sleeping: predicted position equals current position
        pred[i] = pos[i];
        return;
    }

    const float4 p4 = pos[i];
    const float4 v4 = vel[i];
    float3 p = makeFloat3(p4);
    float3 v = makeFloat3(v4);

    // Apply gravity
    v.y -= params.gravity * params.dt;

    // Predicted position x* = x + v * dt
    p = add3(p, mul3(v, params.dt));

    pred[i] = make_float4(p.x, p.y, p.z, p4.w);
}

// -----------------------------------------------------------------------------
// XPBD Jacobi Contact Solve Kernel (Gather-style: writes delta to separate buffer)
// -----------------------------------------------------------------------------
__global__ void solveContactsJacobiKernel(
    const float4* __restrict__ pred,
    float4* __restrict__ deltaX,
    const uint32_t* __restrict__ neighborList,
    const uint32_t* __restrict__ neighborCount,
    const uint32_t* __restrict__ sleepState,
    SolverParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    if (sleepState && sleepState[i] >= params.sleepFrames) {
        deltaX[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const float4 pi4 = pred[i];
    const float3 pi = makeFloat3(pi4);
    const float ri = pi4.w > 0.0f ? pi4.w : params.radius;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    float maxPenetration = 0.0f;
    uint32_t contacts = 0;

    const uint32_t count = neighborCount[i];
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t j = neighborList[baseOffset + k];
        if (j == i) continue;

        const float4 pj4 = pred[j];
        const float3 pj = makeFloat3(pj4);
        const float rj = pj4.w > 0.0f ? pj4.w : params.radius;

        const float3 diff = sub3(pi, pj);
        const float distSq = dot3(diff, diff);
        const float targetDist = ri + rj;

        if (distSq < targetDist * targetDist && distSq > 1e-12f) {
            const float dist = sqrtf(distSq);
            const float penetration = targetDist - dist;
            maxPenetration = fmaxf(maxPenetration, penetration);

            const float3 n = mul3(diff, 1.0f / dist);

            // XPBD correction: w_i / (w_i + w_j + alpha_tilde) * penetration
            // For unit mass grains: 1.0 / (1.0 + 1.0) = 0.5f
            const float alphaTilde = params.compliance * params.invDt * params.invDt;
            const float denom = 2.0f * params.invMass + alphaTilde;
            const float deltaLambda = penetration / denom;

            totalDelta = add3(totalDelta, mul3(n, deltaLambda * params.invMass));
            contacts++;
        }
    }

    // Ground plane collision at y = groundY
    float currentGroundY = params.groundY;
    if (params.vibrationAmp > 0.0f) {
        currentGroundY += params.vibrationAmp * sinf(params.vibrationFreq * params.simTime);
    }

    const float groundPen = (currentGroundY + ri) - pi.y;
    if (groundPen > 0.0f) {
        totalDelta.y += groundPen;
        maxPenetration = fmaxf(maxPenetration, groundPen);
        contacts++;
    }

    // Lateral box walls (if box confinement enabled)
    if (params.boxHalfWidth > 0.0f) {
        const float wallX = params.boxHalfWidth - ri;
        const float wallZ = params.boxHalfWidth - ri;
        if (pi.x < -wallX) { totalDelta.x += (-wallX - pi.x); contacts++; }
        if (pi.x > wallX) { totalDelta.x += (wallX - pi.x); contacts++; }
        if (pi.z < -wallZ) { totalDelta.z += (-wallZ - pi.z); contacts++; }
        if (pi.z > wallZ) { totalDelta.z += (wallZ - pi.z); contacts++; }
    }

    // Average or relax Jacobi updates in dense contacts to ensure stability
    if (contacts > 1) {
        totalDelta = mul3(totalDelta, 0.65f); // Jacobi relaxation factor
    }

    deltaX[i] = make_float4(totalDelta.x, totalDelta.y, totalDelta.z, maxPenetration);
}

// -----------------------------------------------------------------------------
// Jacobi Apply Kernel: pred[i] += deltaX[i]
// -----------------------------------------------------------------------------
__global__ void applyJacobiDeltaKernel(
    float4* __restrict__ pred,
    const float4* __restrict__ deltaX,
    uint32_t grainCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const float4 p = pred[i];
    const float4 d = deltaX[i];
    pred[i] = make_float4(p.x + d.x, p.y + d.y, p.z + d.z, p.w);
}

// -----------------------------------------------------------------------------
// XPBD Cell-Colored Gauss-Seidel Kernel (In-Place Update for active color)
// -----------------------------------------------------------------------------
__global__ void solveContactsColoredGSKernel(
    float4* __restrict__ pred,
    const uint32_t* __restrict__ colorIndices,
    uint32_t colorStart,
    uint32_t colorEnd,
    const uint32_t* __restrict__ neighborList,
    const uint32_t* __restrict__ neighborCount,
    const uint32_t* __restrict__ sleepState,
    SolverParams params)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t colorGrainCount = colorEnd - colorStart;
    if (idx >= colorGrainCount) return;

    const uint32_t i = colorIndices ? colorIndices[colorStart + idx] : (colorStart + idx);

    if (sleepState && sleepState[i] >= params.sleepFrames) return;

    float4 pi4 = pred[i];
    float3 pi = makeFloat3(pi4);
    const float ri = pi4.w > 0.0f ? pi4.w : params.radius;

    const uint32_t count = neighborCount[i];
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t contacts = 0;

    for (uint32_t k = 0; k < count; ++k) {
        const uint32_t j = neighborList[baseOffset + k];
        if (j == i) continue;

        // Reads the latest position of neighbor j (which may have been updated by prior color pass)
        const float4 pj4 = pred[j];
        const float3 pj = makeFloat3(pj4);
        const float rj = pj4.w > 0.0f ? pj4.w : params.radius;

        const float3 diff = sub3(pi, pj);
        const float distSq = dot3(diff, diff);
        const float targetDist = ri + rj;

        if (distSq < targetDist * targetDist && distSq > 1e-12f) {
            const float dist = sqrtf(distSq);
            const float penetration = targetDist - dist;

            const float3 n = mul3(diff, 1.0f / dist);
            const float alphaTilde = params.compliance * params.invDt * params.invDt;
            const float denom = 2.0f * params.invMass + alphaTilde;
            const float deltaLambda = penetration / denom;

            totalDelta = add3(totalDelta, mul3(n, deltaLambda * params.invMass));
            contacts++;
        }
    }

    if (contacts > 1) {
        totalDelta = mul3(totalDelta, 1.0f / sqrtf(static_cast<float>(contacts)));
    }
    pi = add3(pi, totalDelta);

    // Lateral box walls (if box confinement enabled)
    if (params.boxHalfWidth > 0.0f) {
        const float wallX = params.boxHalfWidth - ri;
        const float wallZ = params.boxHalfWidth - ri;
        if (pi.x < -wallX) pi.x = -wallX;
        if (pi.x > wallX) pi.x = wallX;
        if (pi.z < -wallZ) pi.z = -wallZ;
        if (pi.z > wallZ) pi.z = wallZ;
    }

    // Ground plane collision at y = groundY
    float currentGroundY = params.groundY;
    if (params.vibrationAmp > 0.0f) {
        currentGroundY += params.vibrationAmp * sinf(params.vibrationFreq * params.simTime);
    }

    const float groundPen = (currentGroundY + ri) - pi.y;
    if (groundPen > 0.0f) {
        pi.y += groundPen;
    }

    // Write updated position directly back to pred[i]
    pred[i] = make_float4(pi.x, pi.y, pi.z, pi4.w);
}

// -----------------------------------------------------------------------------
// Integrate Kernel: Update velocity v = (x* - x) / dt, handle friction, sleeping
// -----------------------------------------------------------------------------
__global__ void integrateKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ pred,
    uint32_t* __restrict__ sleepState,
    SolverParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    if (sleepState && sleepState[i] >= params.sleepFrames) {
        // Already sleeping
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const float4 oldP4 = pos[i];
    const float4 newP4 = pred[i];
    const float3 oldP = makeFloat3(oldP4);
    const float3 newP = makeFloat3(newP4);

    // v = (x* - x) / dt
    float3 v = mul3(sub3(newP, oldP), params.invDt);

    // Apply friction and low-speed threshold
    const float speedSq = dot3(v, v);
    if (speedSq < 1e-6f) {
        v = make_float3(0.0f, 0.0f, 0.0f);
    } else {
        // Velocity damping
        v = mul3(v, 0.999f);
    }

    // Update sleep counter
    if (sleepState) {
        if (speedSq < params.sleepSpeedSq) {
            sleepState[i]++;
        } else {
            sleepState[i] = 0;
        }
    }

    pos[i] = newP4;
    vel[i] = make_float4(v.x, v.y, v.z, 0.0f);
}

// -----------------------------------------------------------------------------
// Helper reduction to count asleep grains, kinetic energy, and column height
// -----------------------------------------------------------------------------
__global__ void reduceStatsKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    const uint32_t* __restrict__ sleepState,
    uint32_t grainCount,
    uint32_t sleepThreshold,
    float* __restrict__ statsOut) // [0]=asleepCount, [1]=kineticEnergy, [2]=maxHeight, [3]=meanHeight
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) return;

    const float4 p = pos[i];
    const float4 v = vel[i];
    const float ke = 0.5f * (v.x * v.x + v.y * v.y + v.z * v.z);

    if (sleepState && sleepState[i] >= sleepThreshold) {
        atomicAdd(&statsOut[0], 1.0f);
    }

    atomicAdd(&statsOut[1], ke);
    atomicMax(reinterpret_cast<int*>(&statsOut[2]), __float_as_int(p.y));
    atomicAdd(&statsOut[3], p.y);
}

// -----------------------------------------------------------------------------
// Color partition builder: sorts grain indices into 8 contiguous color bins
// -----------------------------------------------------------------------------
__global__ void assignCellColorsKernel(
    const float4* __restrict__ pos,
    uint32_t* __restrict__ grainColors,
    GridParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) return;

    const float4 p = pos[i];
    const int cx = min(max(static_cast<int>(floorf((p.x - params.origin.x) * params.invCellSize)), 0), params.gridDim.x - 1);
    const int cy = min(max(static_cast<int>(floorf((p.y - params.origin.y) * params.invCellSize)), 0), params.gridDim.y - 1);
    const int cz = min(max(static_cast<int>(floorf((p.z - params.origin.z) * params.invCellSize)), 0), params.gridDim.z - 1);

    // 8-color cell pattern
    grainColors[i] = (cx & 1) | ((cy & 1) << 1) | ((cz & 1) << 2);
}

} // namespace

// -----------------------------------------------------------------------------
// Solver Framework
// -----------------------------------------------------------------------------
struct SolverFramework {
    uint32_t grainCount;
    GridParams grid;
    SolverParams solver;

    float4 *d_pos = nullptr, *d_vel = nullptr, *d_pred = nullptr, *d_deltaX = nullptr;
    uint32_t *d_neighborList = nullptr, *d_neighborCount = nullptr;
    uint32_t *d_sleepState = nullptr;
    uint32_t *d_grainColors = nullptr, *d_sortedColorIndices = nullptr;
    uint32_t *d_colorOffsets = nullptr; // size 9: [0..8] offsets
    std::vector<uint32_t> hostColorOffsets;

    float* d_stats = nullptr;
    void* d_sortTemp = nullptr;
    size_t sortTempBytes = 0;

    cudaStream_t stream = nullptr;

    SolverFramework(uint32_t count, float radius, float skinRatio, cudaStream_t s)
        : grainCount(count), stream(s), hostColorOffsets(9, 0)
    {
        const float skin = skinRatio * radius;
        const float cellSize = 2.0f * radius + skin;
        const float sideLen = std::cbrt(static_cast<float>(count)) * 2.05f * radius;

        grid.origin = make_float3(-sideLen * 0.7f, -0.05f, -sideLen * 0.7f);
        grid.cellSize = cellSize;
        grid.invCellSize = 1.0f / cellSize;
        grid.gridDim = make_int3(
            static_cast<int>(std::ceil(sideLen * 1.5f / cellSize)) + 4,
            static_cast<int>(std::ceil(sideLen * 2.5f / cellSize)) + 8,
            static_cast<int>(std::ceil(sideLen * 1.5f / cellSize)) + 4);
        grid.numCells = grid.gridDim.x * grid.gridDim.y * grid.gridDim.z;
        grid.skin = skin;
        grid.radius = radius;
        grid.searchRadius = 2.0f * radius + skin;
        grid.searchRadiusSq = grid.searchRadius * grid.searchRadius;
        grid.grainCount = count;

        solver.grainCount = count;
        solver.dt = (1.0f / 60.0f) / 4.0f; // 4 substeps default
        solver.invDt = 1.0f / solver.dt;
        solver.radius = radius;
        solver.invMass = 1.0f;
        solver.gravity = 9.81f;
        solver.groundY = 0.0f;
        solver.compliance = 0.0f; // rigid
        solver.friction = 0.45f;
        solver.restitution = 0.1f;
        solver.sleepSpeedSq = 0.0004f; // (0.02 m/s)^2
        solver.sleepFrames = 30;
        solver.vibrationAmp = 0.0f;
        solver.vibrationFreq = 0.0f;
        solver.simTime = 0.0f;
        solver.boxHalfWidth = 0.0f;

        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float4) * count));
        CUDA_CHECK(cudaMalloc(&d_vel, sizeof(float4) * count));
        CUDA_CHECK(cudaMalloc(&d_pred, sizeof(float4) * count));
        CUDA_CHECK(cudaMalloc(&d_deltaX, sizeof(float4) * count));
        CUDA_CHECK(cudaMalloc(&d_neighborList, sizeof(uint32_t) * count * kMaxNeighborsPerGrain));
        CUDA_CHECK(cudaMalloc(&d_neighborCount, sizeof(uint32_t) * count));
        CUDA_CHECK(cudaMalloc(&d_sleepState, sizeof(uint32_t) * count));
        CUDA_CHECK(cudaMalloc(&d_grainColors, sizeof(uint32_t) * count));
        CUDA_CHECK(cudaMalloc(&d_sortedColorIndices, sizeof(uint32_t) * count));
        CUDA_CHECK(cudaMalloc(&d_colorOffsets, sizeof(uint32_t) * 9));
        CUDA_CHECK(cudaMalloc(&d_stats, sizeof(float) * 4));

        CUDA_CHECK(cudaMemsetAsync(d_vel, 0, sizeof(float4) * count, stream));
        CUDA_CHECK(cudaMemsetAsync(d_sleepState, 0, sizeof(uint32_t) * count, stream));

        // Setup CUB Radix sort for color sorting
        uint32_t *d_indicesIn = nullptr;
        CUDA_CHECK(cudaMalloc(&d_indicesIn, sizeof(uint32_t) * count));
        std::vector<uint32_t> initialIndices(count);
        std::iota(initialIndices.begin(), initialIndices.end(), 0u);
        CUDA_CHECK(cudaMemcpyAsync(d_indicesIn, initialIndices.data(), sizeof(uint32_t) * count, cudaMemcpyHostToDevice, stream));

        uint32_t *d_keysOut = nullptr;
        CUDA_CHECK(cudaMalloc(&d_keysOut, sizeof(uint32_t) * count));
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr, sortTempBytes, d_grainColors, d_keysOut, d_indicesIn, d_sortedColorIndices, count, 0, 4, stream));
        CUDA_CHECK(cudaMalloc(&d_sortTemp, sortTempBytes));

        CUDA_CHECK(cudaFree(d_indicesIn));
        CUDA_CHECK(cudaFree(d_keysOut));
    }

    ~SolverFramework()
    {
        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_pred);
        cudaFree(d_deltaX);
        cudaFree(d_neighborList);
        cudaFree(d_neighborCount);
        cudaFree(d_sleepState);
        cudaFree(d_grainColors);
        cudaFree(d_sortedColorIndices);
        cudaFree(d_colorOffsets);
        cudaFree(d_stats);
        cudaFree(d_sortTemp);
    }

    void uploadPositions(const std::vector<float4>& positions)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_pos, positions.data(), sizeof(float4) * grainCount, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pred, positions.data(), sizeof(float4) * grainCount, cudaMemcpyHostToDevice, stream));
    }

    void buildColorPartitions()
    {
        const uint32_t blocks = (grainCount + kThreads - 1) / kThreads;
        assignCellColorsKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_grainColors, grid);

        std::vector<uint32_t> initialIndices(grainCount);
        std::iota(initialIndices.begin(), initialIndices.end(), 0u);
        uint32_t* d_indicesIn = nullptr;
        uint32_t* d_keysOut = nullptr;
        CUDA_CHECK(cudaMalloc(&d_indicesIn, sizeof(uint32_t) * grainCount));
        CUDA_CHECK(cudaMalloc(&d_keysOut, sizeof(uint32_t) * grainCount));
        CUDA_CHECK(cudaMemcpyAsync(d_indicesIn, initialIndices.data(), sizeof(uint32_t) * grainCount, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_sortTemp, sortTempBytes, d_grainColors, d_keysOut, d_indicesIn, d_sortedColorIndices, grainCount, 0, 4, stream));

        std::vector<uint32_t> sortedKeys(grainCount);
        CUDA_CHECK(cudaMemcpyAsync(sortedKeys.data(), d_keysOut, sizeof(uint32_t) * grainCount, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Find color bin offsets
        std::fill(hostColorOffsets.begin(), hostColorOffsets.end(), 0u);
        for (uint32_t k = 0; k < grainCount; ++k) {
            uint32_t c = sortedKeys[k] & 7;
            hostColorOffsets[c + 1] = k + 1;
        }
        for (int c = 1; c <= 8; ++c) {
            if (hostColorOffsets[c] < hostColorOffsets[c - 1]) {
                hostColorOffsets[c] = hostColorOffsets[c - 1];
            }
        }

        CUDA_CHECK(cudaFree(d_indicesIn));
        CUDA_CHECK(cudaFree(d_keysOut));
    }

    void executeSubstep(SolverMode mode, uint32_t iterations)
    {
        const uint32_t blocks = (grainCount + kThreads - 1) / kThreads;

        // 1. Predict
        predictKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_vel, d_pred, d_sleepState, solver);

        // 2. Solve Constraints
        if (mode == SolverMode::Jacobi) {
            for (uint32_t iter = 0; iter < iterations; ++iter) {
                solveContactsJacobiKernel<<<blocks, kThreads, 0, stream>>>(
                    d_pred, d_deltaX, d_neighborList, d_neighborCount, d_sleepState, solver);
                applyJacobiDeltaKernel<<<blocks, kThreads, 0, stream>>>(d_pred, d_deltaX, grainCount);
            }
        } else {
            // Cell-Colored Gauss-Seidel (8 color passes per iteration)
            for (uint32_t iter = 0; iter < iterations; ++iter) {
                for (uint32_t c = 0; c < 8; ++c) {
                    const uint32_t cStart = hostColorOffsets[c];
                    const uint32_t cEnd = hostColorOffsets[c + 1];
                    const uint32_t cCount = cEnd - cStart;
                    if (cCount == 0) continue;

                    const uint32_t cBlocks = (cCount + kThreads - 1) / kThreads;
                    solveContactsColoredGSKernel<<<cBlocks, kThreads, 0, stream>>>(
                        d_pred, d_sortedColorIndices, cStart, cEnd, d_neighborList, d_neighborCount, d_sleepState, solver);
                }
            }
        }

        // 3. Integrate
        integrateKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_vel, d_pred, d_sleepState, solver);
        solver.simTime += solver.dt;
    }

    void readStats(float& asleepCount, float& ke, float& maxHeight, float& meanHeight)
    {
        const uint32_t blocks = (grainCount + kThreads - 1) / kThreads;
        CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(float) * 4, stream));
        reduceStatsKernel<<<blocks, kThreads, 0, stream>>>(d_pos, d_vel, d_sleepState, grainCount, solver.sleepFrames, d_stats);
        float hStats[4] = {};
        CUDA_CHECK(cudaMemcpyAsync(hStats, d_stats, sizeof(float) * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        asleepCount = hStats[0];
        ke = hStats[1];
        maxHeight = hStats[2];
        meanHeight = hStats[3] / static_cast<float>(grainCount);
    }
};

// -----------------------------------------------------------------------------
// Lattice Initializer
// -----------------------------------------------------------------------------
std::vector<float4> generateBoxPositions(uint32_t count, float radius)
{
    std::vector<float4> pos(count);
    const float spacing = 2.01f * radius;
    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(count))));
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t x = i % side;
        const uint32_t z = (i / side) % side;
        const uint32_t y = i / (side * side);
        pos[i] = make_float4(
            (static_cast<float>(x) - 0.5f * static_cast<float>(side)) * spacing,
            radius + static_cast<float>(y) * spacing,
            (static_cast<float>(z) - 0.5f * static_cast<float>(side)) * spacing,
            radius);
    }
    return pos;
}

// -----------------------------------------------------------------------------
// Setup dummy lattice neighbor lists (6 orthogonal contacts for timing bench)
// -----------------------------------------------------------------------------
__global__ void buildLatticeNeighborListKernel(
    uint32_t* __restrict__ neighborList,
    uint32_t* __restrict__ neighborCount,
    uint32_t count,
    uint32_t side)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const int x = i % side;
    const int z = (i / side) % side;
    const int y = i / (side * side);

    uint32_t c = 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    for (int k = 0; k < 6; ++k) {
        int nx = x + dx[k];
        int ny = y + dy[k];
        int nz = z + dz[k];
        if (nx >= 0 && nx < side && ny >= 0 && nz >= 0 && nz < side) {
            uint32_t j = nx + nz * side + ny * side * side;
            if (j < count && j != i) {
                neighborList[baseOffset + c++] = j;
            }
        }
    }
    neighborCount[i] = c;
}

// -----------------------------------------------------------------------------
// Benchmark Test: Latency & DRAM Bandwidth per Contact Solve Pass
// -----------------------------------------------------------------------------
void runContactSolvePassBenchmark(uint32_t grainCount, cudaStream_t stream, std::ofstream* csv = nullptr)
{
    std::cout << "\n======================================================\n";
    std::cout << "  Contact-Solve Pass Benchmark (" << grainCount << " grains)\n";
    std::cout << "======================================================\n";

    const float radius = 0.01f;
    SolverFramework framework(grainCount, radius, 0.20f, stream);
    std::vector<float4> initialPos = generateBoxPositions(grainCount, radius);
    framework.uploadPositions(initialPos);
    framework.buildColorPartitions();

    // Populate lattice neighbors
    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(grainCount))));
    const uint32_t blocks = (grainCount + kThreads - 1) / kThreads;
    buildLatticeNeighborListKernel<<<blocks, kThreads, 0, stream>>>(
        framework.d_neighborList, framework.d_neighborCount, grainCount, side);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int w = 0; w < 3; ++w) {
        solveContactsJacobiKernel<<<blocks, kThreads, 0, stream>>>(
            framework.d_pred, framework.d_deltaX, framework.d_neighborList, framework.d_neighborCount, nullptr, framework.solver);
        applyJacobiDeltaKernel<<<blocks, kThreads, 0, stream>>>(framework.d_pred, framework.d_deltaX, grainCount);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed Jacobi
    const int kRuns = 15;
    std::vector<float> jacobiTimes;
    for (int r = 0; r < kRuns; ++r) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        solveContactsJacobiKernel<<<blocks, kThreads, 0, stream>>>(
            framework.d_pred, framework.d_deltaX, framework.d_neighborList, framework.d_neighborCount, nullptr, framework.solver);
        applyJacobiDeltaKernel<<<blocks, kThreads, 0, stream>>>(framework.d_pred, framework.d_deltaX, grainCount);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        jacobiTimes.push_back(ms);
    }

    // Timed Colored Gauss-Seidel
    std::vector<float> gsTimes;
    for (int r = 0; r < kRuns; ++r) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (uint32_t c = 0; c < 8; ++c) {
            uint32_t cStart = framework.hostColorOffsets[c];
            uint32_t cEnd = framework.hostColorOffsets[c + 1];
            uint32_t cCount = cEnd - cStart;
            if (cCount == 0) continue;
            uint32_t cBlocks = (cCount + kThreads - 1) / kThreads;
            solveContactsColoredGSKernel<<<cBlocks, kThreads, 0, stream>>>(
                framework.d_pred, framework.d_sortedColorIndices, cStart, cEnd,
                framework.d_neighborList, framework.d_neighborCount, nullptr, framework.solver);
        }
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gsTimes.push_back(ms);
    }

    std::sort(jacobiTimes.begin(), jacobiTimes.end());
    std::sort(gsTimes.begin(), gsTimes.end());

    const float jacobiMedian = jacobiTimes[kRuns / 2];
    const float jacobiP95 = jacobiTimes[std::min(static_cast<int>(kRuns * 0.95f), kRuns - 1)];
    const float gsMedian = gsTimes[kRuns / 2];
    const float gsP95 = gsTimes[std::min(static_cast<int>(kRuns * 0.95f), kRuns - 1)];

    // DRAM Traffic:
    // Reads: pred[i] (16B), count (4B), 6 neighbors (6*4B), 6 pred[j] (6*16B) = 140 Bytes/grain
    // Writes: deltaX[i] (16B) + pred update (16B in, 16B out) = 48 Bytes
    // Total traffic ~ 188 Bytes/grain
    const float bytesPerPass = static_cast<float>(grainCount * 188ULL);
    const float jacobiBw = (bytesPerPass / (jacobiMedian * 1e-3f)) / 1e9f;
    const float gsBw = (bytesPerPass / (gsMedian * 1e-3f)) / 1e9f;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "XPBD Jacobi Pass          : Median = " << jacobiMedian << " ms | p95 = " << jacobiP95
              << " ms | Bandwidth = " << std::setprecision(1) << jacobiBw << " GB/s\n";
    std::cout << std::setprecision(3);
    std::cout << "XPBD Colored GS Pass (8c) : Median = " << gsMedian << " ms | p95 = " << gsP95
              << " ms | Bandwidth = " << std::setprecision(1) << gsBw << " GB/s\n";

    // Target checks:
    if (grainCount == 250'000) {
        std::cout << "\n>>> G0 TARGET (250K Solve Pass < 1.0 ms): "
                  << ((gsMedian < 1.0f) ? "PASSED" : "FAILED") << " (" << gsMedian << " ms < 1.0 ms) <<<\n";
    } else if (grainCount == 1'000'000) {
        std::cout << "\n>>> G0 TARGET (1M Solve Pass < 3.5 ms): "
                  << ((gsMedian < 3.5f) ? "PASSED" : "FAILED") << " (" << gsMedian << " ms < 3.5 ms) <<<\n";
    }
    std::cout << ">>> G0 TARGET (DRAM Bandwidth > 120 GB/s): "
              << ((gsBw > 120.0f) ? "PASSED" : "FAILED") << " (" << gsBw << " GB/s > 120 GB/s) <<<\n";

    if (csv && *csv) {
        *csv << "solvePass," << grainCount << ",Jacobi," << jacobiMedian << ',' << jacobiP95 << ',' << jacobiBw << '\n';
        *csv << "solvePass," << grainCount << ",ColoredGS," << gsMedian << ',' << gsP95 << ',' << gsBw << '\n';
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// -----------------------------------------------------------------------------
// Scenario 2: Stack Compression (100K grains)
// Measures column compression vs number of solver passes (2, 4, 8, 16 passes)
// Target: < 5% compression with <= 8 passes
// -----------------------------------------------------------------------------
void runStackCompressionTest(cudaStream_t stream, std::ofstream* csv = nullptr)
{
    std::cout << "\n======================================================\n";
    std::cout << "  Scenario 2: Stack Compression Test (100,000 grains)\n";
    std::cout << "======================================================\n";

    const uint32_t count = 100'000;
    const float radius = 0.01f;
    const uint32_t side = 20; // 20 x 20 x 250 grains (tall column, 250 layers high)
    const float initialHeight = 250.0f * (2.0f * radius);

    const std::vector<uint32_t> testPasses = {2, 4, 8, 16};

    for (uint32_t passes : testPasses) {
        SolverFramework framework(count, radius, 0.20f, stream);
        std::vector<float4> initialPos(count);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t x = i % side;
            uint32_t z = (i / side) % side;
            uint32_t y = i / (side * side);
            initialPos[i] = make_float4(
                (static_cast<float>(x) - 0.5f * static_cast<float>(side)) * (2.02f * radius),
                radius + static_cast<float>(y) * (2.02f * radius),
                (static_cast<float>(z) - 0.5f * static_cast<float>(side)) * (2.02f * radius),
                radius);
        }
        framework.uploadPositions(initialPos);
        framework.buildColorPartitions();

        framework.solver.boxHalfWidth = (static_cast<float>(side) * 0.5f) * (2.02f * radius);
        const uint32_t blocks = (count + kThreads - 1) / kThreads;
        buildLatticeNeighborListKernel<<<blocks, kThreads, 0, stream>>>(
            framework.d_neighborList, framework.d_neighborCount, count, side);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Simulate 60 frames under gravity with 'passes' solver iterations per substep
        const uint32_t frames = 60;
        const uint32_t substeps = 4;
        for (uint32_t f = 0; f < frames; ++f) {
            for (uint32_t s = 0; s < substeps; ++s) {
                framework.executeSubstep(SolverMode::ColoredGaussSeidel, passes);
            }
        }

        float asleep = 0, ke = 0, maxHeight = 0, meanHeight = 0;
        framework.readStats(asleep, ke, maxHeight, meanHeight);

        const float compression = ((initialHeight - maxHeight) / initialHeight) * 100.0f;
        std::cout << "Passes: " << std::setw(2) << passes
                  << " | Final Height: " << std::fixed << std::setprecision(3) << maxHeight
                  << " m (initial: " << initialHeight << " m) | Compression: "
                  << std::setprecision(2) << compression << "%\n";

        if (csv && *csv) {
            *csv << "compression,100000," << passes << ',' << maxHeight << ',' << compression << '\n';
        }

        if (passes == 8) {
            std::cout << "\n>>> G0 TARGET (<5% Compression with <= 8 passes): "
                      << ((compression < 5.0f) ? "PASSED" : "FAILED")
                      << " (" << compression << "% < 5.0%) <<<\n";
        }
    }
}

// -----------------------------------------------------------------------------
// Scenario 3: Ground Shake / Vibration Transmission (50-grain column)
// Target: Shake reaches top grain (Yes/No)
// -----------------------------------------------------------------------------
void runVibrationTransmissionTest(cudaStream_t stream, std::ofstream* csv = nullptr)
{
    std::cout << "\n======================================================\n";
    std::cout << "  Scenario 3: Ground Shake / Vibration Transmission (50 grains)\n";
    std::cout << "======================================================\n";

    const uint32_t count = 50;
    const float radius = 0.01f;

    auto testSolver = [&](SolverMode mode, const char* name) -> float {
        SolverFramework framework(count, radius, 0.20f, stream);
        framework.solver.vibrationAmp = 0.005f;  // 5 mm shake amplitude
        framework.solver.vibrationFreq = 60.0f * 6.283185f; // 60 Hz vibration

        std::vector<float4> initialPos(count);
        for (uint32_t i = 0; i < count; ++i) {
            initialPos[i] = make_float4(0.0f, radius + static_cast<float>(i) * (2.0f * radius), 0.0f, radius);
        }
        framework.uploadPositions(initialPos);
        framework.buildColorPartitions();

        // 1D neighbor connections
        std::vector<uint32_t> nCounts(count, 0);
        std::vector<uint32_t> nLists(count * kMaxNeighborsPerGrain, 0);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t c = 0;
            if (i > 0) nLists[i * kMaxNeighborsPerGrain + c++] = i - 1;
            if (i + 1 < count) nLists[i * kMaxNeighborsPerGrain + c++] = i + 1;
            nCounts[i] = c;
        }
        CUDA_CHECK(cudaMemcpyAsync(framework.d_neighborCount, nCounts.data(), sizeof(uint32_t) * count, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(framework.d_neighborList, nLists.data(), sizeof(uint32_t) * nLists.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float maxTopGrainVelocity = 0.0f;
        const uint32_t frames = 90;
        const uint32_t substeps = 4;

        for (uint32_t f = 0; f < frames; ++f) {
            for (uint32_t s = 0; s < substeps; ++s) {
                framework.executeSubstep(mode, 4);
            }
            float4 topVel;
            CUDA_CHECK(cudaMemcpy(&topVel, &framework.d_vel[count - 1], sizeof(float4), cudaMemcpyDeviceToHost));
            maxTopGrainVelocity = fmaxf(maxTopGrainVelocity, fabsf(topVel.y));
        }

        std::cout << name << " -> Max Top Grain Speed: " << std::fixed << std::setprecision(4)
                  << maxTopGrainVelocity << " m/s | Vibration Transmitted: "
                  << (maxTopGrainVelocity > 0.02f ? "YES" : "NO") << '\n';

        if (csv && *csv) {
            *csv << "vibration,50," << name << ',' << maxTopGrainVelocity << '\n';
        }
        return maxTopGrainVelocity;
    };

    float jacobiVel = testSolver(SolverMode::Jacobi, "XPBD Jacobi (4 passes)");
    float gsVel = testSolver(SolverMode::ColoredGaussSeidel, "XPBD Colored GS (4 passes)");

    std::cout << "\n>>> G0 TARGET (Ground shake reaches top of 50-grain pile): "
              << ((gsVel > 0.02f) ? "YES (PASSED)" : "NO (FAILED)") << " <<<\n";
}

// -----------------------------------------------------------------------------
// Scenario 4: Settling Pour & Awake/Asleep Fraction (250K grains)
// Target: > 80% asleep after pour settles
// -----------------------------------------------------------------------------
void runSettlingPourTest(uint32_t grainCount, cudaStream_t stream, std::ofstream* csv = nullptr)
{
    std::cout << "\n======================================================\n";
    std::cout << "  Scenario 4: Settling Pour Awake/Asleep Test (" << grainCount << " grains)\n";
    std::cout << "======================================================\n";

    const float radius = 0.01f;
    SolverFramework framework(grainCount, radius, 0.20f, stream);
    std::vector<float4> initialPos = generateBoxPositions(grainCount, radius);
    framework.uploadPositions(initialPos);
    framework.buildColorPartitions();

    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(grainCount))));
    const uint32_t blocks = (grainCount + kThreads - 1) / kThreads;
    buildLatticeNeighborListKernel<<<blocks, kThreads, 0, stream>>>(
        framework.d_neighborList, framework.d_neighborCount, grainCount, side);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Settle for 180 frames (3 seconds sim time)
    const uint32_t frames = 180;
    const uint32_t substeps = 4;

    for (uint32_t f = 0; f < frames; ++f) {
        for (uint32_t s = 0; s < substeps; ++s) {
            framework.executeSubstep(SolverMode::ColoredGaussSeidel, 4);
        }

        if (f % 30 == 0 || f == frames - 1) {
            float asleep = 0, ke = 0, maxHeight = 0, meanHeight = 0;
            framework.readStats(asleep, ke, maxHeight, meanHeight);
            float asleepFraction = (asleep / static_cast<float>(grainCount)) * 100.0f;
            std::cout << "Frame " << std::setw(3) << f
                      << " | KE: " << std::fixed << std::setprecision(4) << ke
                      << " | Asleep: " << std::setprecision(1) << asleepFraction
                      << "% (" << static_cast<uint32_t>(asleep) << " / " << grainCount << ")\n";
        }
    }

    float finalAsleep = 0, finalKE = 0, maxHeight = 0, meanHeight = 0;
    framework.readStats(finalAsleep, finalKE, maxHeight, meanHeight);
    float finalFraction = (finalAsleep / static_cast<float>(grainCount)) * 100.0f;

    std::cout << "\n>>> G0 TARGET (>80% Asleep after Settling): "
              << ((finalFraction > 80.0f) ? "PASSED" : "FAILED")
              << " (" << finalFraction << "% > 80.0%) <<<\n";

    if (csv && *csv) {
        *csv << "settlingPour," << grainCount << ',' << finalFraction << ',' << finalKE << '\n';
    }
}

SpikeConfig parseArgs(int argc, char** argv)
{
    SpikeConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--scenario" && i + 1 < argc) config.scenario = argv[++i];
        else if (arg == "--grains" && i + 1 < argc) config.grainCount = std::stoul(argv[++i]);
        else if (arg == "--runs" && i + 1 < argc) config.runs = std::stoul(argv[++i]);
        else if (arg == "--substeps" && i + 1 < argc) config.substeps = std::stoul(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) config.iterations = std::stoul(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) config.csvPath = argv[++i];
        else if (arg == "--help") {
            std::cout << "Phase11ASolverSpike options:\n"
                      << "  --scenario S    all | bench | compression | vibration | pour (default: all)\n"
                      << "  --grains N      Grain count for bench (default: 250000)\n"
                      << "  --csv FILE      CSV output path\n";
            std::exit(0);
        }
    }
    return config;
}

int main(int argc, char** argv)
{
    try {
        const SpikeConfig config = parseArgs(argc, argv);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "======================================================\n";
        std::cout << "  VParticles Phase 11A - Contact Solver Spike\n";
        std::cout << "======================================================\n";
        std::cout << "Device              : " << prop.name << " (" << prop.multiProcessorCount << " SMs)\n";

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::ofstream csv;
        if (!config.csvPath.empty()) {
            csv.open(config.csvPath);
            csv << "test,grains,variant,metric1,metric2,bandwidth\n";
        }

        if (config.scenario == "all" || config.scenario == "bench") {
            // 250K pass bench
            runContactSolvePassBenchmark(250'000, stream, csv.is_open() ? &csv : nullptr);
            // 1M pass bench
            runContactSolvePassBenchmark(1'000'000, stream, csv.is_open() ? &csv : nullptr);
        }

        if (config.scenario == "all" || config.scenario == "compression") {
            runStackCompressionTest(stream, csv.is_open() ? &csv : nullptr);
        }

        if (config.scenario == "all" || config.scenario == "vibration") {
            runVibrationTransmissionTest(stream, csv.is_open() ? &csv : nullptr);
        }

        if (config.scenario == "all" || config.scenario == "pour") {
            runSettlingPourTest(250'000, stream, csv.is_open() ? &csv : nullptr);
        }

        if (csv.is_open()) {
            std::cout << "\nExported solver results to " << config.csvPath << '\n';
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Phase11ASolverSpike ERROR: " << e.what() << '\n';
        return 1;
    }
}
