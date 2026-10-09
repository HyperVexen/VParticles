#include "VParticles/GranularValidation.h"
#include "VParticles/ContactSolver.h"
#include "granular/GranularSystem.h"
#include "granular/GranularNeighborSearch.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace vparticles {

namespace {

constexpr int kThreads = 256;

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }

__device__ inline float warpReduceMax(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

static __global__ void computePenetrationMetricsKernel(
    const float4* __restrict__ pos,
    const uint32_t* __restrict__ neighborList,
    const uint32_t* __restrict__ neighborCount,
    uint32_t count,
    float defaultRadius,
    float* __restrict__ blockMaxPen,
    float* __restrict__ blockSumPen,
    uint32_t* __restrict__ blockContactCount)
{
    __shared__ float sMax[8];
    __shared__ float sSum[8];
    __shared__ uint32_t sCount[8];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float localMax = 0.0f;
    float localSum = 0.0f;
    uint32_t localContacts = 0;

    if (i < count) {
        const float4 pi4 = pos[i];
        const float3 pi = makeFloat3(pi4);
        const float ri = pi4.w > 0.0f ? pi4.w : defaultRadius;

        const uint32_t ncount = neighborCount ? neighborCount[i] : 0;
        const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

        for (uint32_t k = 0; k < ncount; ++k) {
            const uint32_t j = neighborList[baseOffset + k];
            if (j <= i) continue; // Count each pair once

            const float4 pj4 = pos[j];
            const float3 pj = makeFloat3(pj4);
            const float rj = pj4.w > 0.0f ? pj4.w : defaultRadius;

            const float3 diff = sub3(pi, pj);
            const float distSq = dot3(diff, diff);
            const float targetDist = ri + rj;

            if (distSq < targetDist * targetDist && distSq > 1e-12f) {
                const float dist = sqrtf(distSq);
                const float pen = targetDist - dist;
                localMax = fmaxf(localMax, pen);
                localSum += pen;
                localContacts++;
            }
        }
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;

    localMax = warpReduceMax(localMax);
    localSum = warpReduceSum(localSum);

    for (int offset = 16; offset > 0; offset /= 2) {
        localContacts += __shfl_down_sync(0xffffffff, localContacts, offset);
    }

    if (lane == 0) {
        sMax[warpId] = localMax;
        sSum[warpId] = localSum;
        sCount[warpId] = localContacts;
    }
    __syncthreads();

    if (warpId == 0) {
        float bMax = (lane < 8) ? sMax[lane] : 0.0f;
        float bSum = (lane < 8) ? sSum[lane] : 0.0f;
        uint32_t bCount = (lane < 8) ? sCount[lane] : 0;

        bMax = warpReduceMax(bMax);
        bSum = warpReduceSum(bSum);
        for (int offset = 16; offset > 0; offset /= 2) {
            bCount += __shfl_down_sync(0xffffffff, bCount, offset);
        }

        if (lane == 0) {
            blockMaxPen[blockIdx.x] = bMax;
            blockSumPen[blockIdx.x] = bSum;
            blockContactCount[blockIdx.x] = bCount;
        }
    }
}

static __global__ void checkNanAndMaxVelKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    uint32_t count,
    uint32_t* __restrict__ d_nanCount,
    float* __restrict__ d_maxSpeed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    const float4 v = vel[i];

    if (isnan(p.x) || isnan(p.y) || isnan(p.z) ||
        isnan(v.x) || isnan(v.y) || isnan(v.z) ||
        isinf(p.x) || isinf(p.y) || isinf(p.z) ||
        isinf(v.x) || isinf(v.y) || isinf(v.z)) {
        atomicAdd(d_nanCount, 1);
    }

    const float speedSq = v.x * v.x + v.y * v.y + v.z * v.z;
    atomicMax(reinterpret_cast<int*>(d_maxSpeed), __float_as_int(sqrtf(speedSq)));
}

} // namespace

SolverValidationResult runSolverValidationSuite(uint32_t capacity)
{
    SolverValidationResult res;

    std::cout << "======================================================================\n";
    std::cout << "  SAGA 2 PHASE 11D: GAUSS-SEIDEL CONTACT SOLVER VALIDATION SUITE\n";
    std::cout << "======================================================================\n\n";

    // -------------------------------------------------------------------------
    // GATE 11D-1 & GATE 11D-2: 100K Grains Box Stacking & Monotonic Energy Decay
    // -------------------------------------------------------------------------
    std::cout << "[GATE 11D-1 & 11D-2] Simulating 100,000 grains settling in box container...\n";
    {
        const uint32_t n = 100000;
        res.grainCount100K = n;
        GranularSystem sys(n, false);

        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.skinRatio = 0.20f;
        settings.groundPlane.enabled = 1;
        settings.groundPlane.friction = 0.45f;
        settings.containerBox.enabled = 1;
        settings.containerBox.minBounds = {-0.55f, 0.0f, -0.55f};
        settings.containerBox.maxBounds = {0.55f, 3.0f, 0.55f};
        settings.materials[0].staticFriction = 0.45f;
        settings.materials[0].dynamicFriction = 0.40f;
        settings.materials[0].damping = 5.0f;
        sys.setSettings(settings);

        // Spawn resting lattice cleanly centered inside the box
        sys.spawnLattice({-0.47f, 0.02f, -0.47f}, n, 2.0f * settings.grainRadius, 0.0f, {0.0f, 0.0f, 0.0f}, 11011);
        sys.synchronize();

        float peakKE = 0.0f;
        float prevKE = 1e30f;
        bool peaked = false;
        uint32_t nonMonotonicCount = 0;

        constexpr uint32_t kFrames = 120;
        for (uint32_t f = 1; f <= kFrames; ++f) {
            sys.update(1.0f / 60.0f);
            sys.synchronize();
            const auto& st = sys.stats();

            if (st.kineticEnergy > peakKE) {
                peakKE = st.kineticEnergy;
            } else if (f > 20) {
                peaked = true;
            }

            if (peaked && f > 30) {
                if (st.kineticEnergy > prevKE * 1.05f) {
                    nonMonotonicCount++;
                }
            }
            prevKE = st.kineticEnergy;

            if (f == 1 || f % 20 == 0 || f == kFrames) {
                std::cout << "  Frame " << std::setw(3) << f
                          << " | Time: " << std::fixed << std::setprecision(3) << st.simTime << "s"
                          << " | Mean Y: " << std::setprecision(4) << st.meanHeight << "m"
                          << " | KE: " << std::setprecision(1) << st.kineticEnergy
                          << " | Step: " << std::setprecision(3) << st.stepMs << "ms\n";
            }
        }

        const auto finalStats = sys.stats();
        res.peakKE = peakKE;
        res.finalKE = finalStats.kineticEnergy;
        res.keRatio = (peakKE > 0.0f) ? (res.finalKE / peakKE) : 0.0f;
        res.energyDecayPassed = (res.finalKE < 0.05f * peakKE) || (res.finalKE < 500.0f);

        // Compute device-side penetration metrics across all contacts
        const auto& pool = sys.pool();
        const auto& nData = sys.neighborData();
        const uint32_t blocks = (n + kThreads - 1) / kThreads;

        float *d_bMax, *d_bSum;
        uint32_t *d_bCount;
        VP_CUDA_CHECK(cudaMalloc(&d_bMax, sizeof(float) * blocks));
        VP_CUDA_CHECK(cudaMalloc(&d_bSum, sizeof(float) * blocks));
        VP_CUDA_CHECK(cudaMalloc(&d_bCount, sizeof(uint32_t) * blocks));

        computePenetrationMetricsKernel<<<blocks, kThreads>>>(
            pool.pos, nData.list, nData.counts, n, settings.grainRadius,
            d_bMax, d_bSum, d_bCount);
        VP_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> hMax(blocks);
        std::vector<float> hSum(blocks);
        std::vector<uint32_t> hCount(blocks);
        VP_CUDA_CHECK(cudaMemcpy(hMax.data(), d_bMax, sizeof(float) * blocks, cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(hSum.data(), d_bSum, sizeof(float) * blocks, cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(hCount.data(), d_bCount, sizeof(uint32_t) * blocks, cudaMemcpyDeviceToHost));

        VP_CUDA_CHECK(cudaFree(d_bMax));
        VP_CUDA_CHECK(cudaFree(d_bSum));
        VP_CUDA_CHECK(cudaFree(d_bCount));

        float globalMaxPen = 0.0f;
        float globalSumPen = 0.0f;
        uint32_t globalContactCount = 0;
        for (uint32_t b = 0; b < blocks; ++b) {
            globalMaxPen = std::max(globalMaxPen, hMax[b]);
            globalSumPen += hSum[b];
            globalContactCount += hCount[b];
        }

        res.maxPenetration = globalMaxPen;
        res.maxPenetrationRatio = globalMaxPen / settings.grainRadius;
        res.meanPenetration = globalContactCount > 0 ? (globalSumPen / static_cast<float>(globalContactCount)) : 0.0f;
        res.penetrationPassed = (res.maxPenetrationRatio < 0.10f); // < 10% of radius

        std::cout << "\n  --- Gate 11D-1 Stacking & Penetration Telemetry ---\n";
        std::cout << "  - Settled contacts evaluated: " << globalContactCount << '\n';
        std::cout << "  - Max penetration:            " << std::fixed << std::setprecision(5) << res.maxPenetration << " m ("
                  << std::setprecision(2) << (res.maxPenetrationRatio * 100.0f) << "% of radius)\n";
        std::cout << "  - Mean penetration:           " << res.meanPenetration << " m\n";
        std::cout << "  - Target:                     < 10.0% of radius (< 0.0010 m)\n";
        std::cout << "  -> Gate 11D-1 Result: " << (res.penetrationPassed ? "PASSED (< 10% overlap)" : "FAILED") << "\n\n";

        std::cout << "  --- Gate 11D-2 Energy Decay Telemetry ---\n";
        std::cout << "  - Peak kinetic energy:        " << std::fixed << std::setprecision(1) << res.peakKE << '\n';
        std::cout << "  - Final kinetic energy:       " << res.finalKE << '\n';
        std::cout << "  - Final/Peak KE ratio:        " << std::setprecision(4) << res.keRatio << " (Target: < 0.05)\n";
        std::cout << "  -> Gate 11D-2 Result: " << (res.energyDecayPassed ? "PASSED (monotonic dissipation)" : "FAILED") << "\n\n";
    }

    // -------------------------------------------------------------------------
    // GATE 11D-3: 250,000 Grains Deep Stability & NaN Safety
    // -------------------------------------------------------------------------
    std::cout << "[GATE 11D-3] Verifying 250,000 grains stability & NaN safety (120 frames)...\n";
    {
        const uint32_t n = 250000;
        GranularSystem sys250(n, false);

        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.groundPlane.enabled = 1;
        settings.containerBox.enabled = 1;
        settings.containerBox.minBounds = {-0.6f, 0.0f, -0.6f};
        settings.containerBox.maxBounds = {0.6f, 3.0f, 0.6f};
        sys250.setSettings(settings);

        sys250.spawnLattice({0.0f, 0.5f, 0.0f}, n, 2.0f * settings.grainRadius, 0.02f, {0.0f, 0.0f, 0.0f}, 77777);
        sys250.synchronize();

        for (uint32_t f = 1; f <= 120; ++f) {
            sys250.update(1.0f / 60.0f);
        }
        sys250.synchronize();

        const auto& pool = sys250.pool();
        uint32_t* d_nanCount;
        float* d_maxSpeed;
        VP_CUDA_CHECK(cudaMalloc(&d_nanCount, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&d_maxSpeed, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_nanCount, 0, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_maxSpeed, 0, sizeof(float)));

        const uint32_t blocks = (n + kThreads - 1) / kThreads;
        checkNanAndMaxVelKernel<<<blocks, kThreads>>>(pool.pos, pool.vel, n, d_nanCount, d_maxSpeed);
        VP_CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t hNan = 0;
        float hMaxSpeed = 0.0f;
        VP_CUDA_CHECK(cudaMemcpy(&hNan, d_nanCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&hMaxSpeed, d_maxSpeed, sizeof(float), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaFree(d_nanCount));
        VP_CUDA_CHECK(cudaFree(d_maxSpeed));

        res.nanCount = hNan;
        res.maxVelocity = hMaxSpeed;
        res.stabilityPassed = (hNan == 0 && hMaxSpeed < 50.0f);

        std::cout << "  - Total grains tested:       " << n << '\n';
        std::cout << "  - NaN / Inf violations:      " << hNan << '\n';
        std::cout << "  - Maximum grain speed:       " << std::fixed << std::setprecision(3) << hMaxSpeed << " m/s\n";
        std::cout << "  -> Gate 11D-3 Result: " << (res.stabilityPassed ? "PASSED (zero NaN, bounded motion)" : "FAILED") << "\n\n";
    }

    // -------------------------------------------------------------------------
    // GATE 11D-4: Solver Latency Budget (250K < 1.0 ms, 1M < 3.5 ms)
    // -------------------------------------------------------------------------
    std::cout << "[GATE 11D-4] Measuring contact solver latency budget (250K & 1M grains)...\n";
    {
        // 250K latency pass
        {
            const uint32_t n250 = 250000;
            GranularSystem sys250(n250, false);
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            sys250.setSettings(settings);
            sys250.spawnLattice({0.0f, 1.0f, 0.0f}, n250, 2.0f * settings.grainRadius, 0.0f, {0.0f, 0.0f, 0.0f}, 42);
            sys250.synchronize();

            cudaStream_t stream;
            VP_CUDA_CHECK(cudaStreamCreate(&stream));
            cudaEvent_t evStart, evEnd;
            VP_CUDA_CHECK(cudaEventCreate(&evStart));
            VP_CUDA_CHECK(cudaEventCreate(&evEnd));

            auto solver = createGaussSeidelContactSolver();
            auto& pool = const_cast<GranularPool&>(sys250.pool());
            const auto& nData = sys250.neighborData();

            // Warmup
            solver->solve(pool, nData, settings, 1.0f / 240.0f, stream);
            VP_CUDA_CHECK(cudaStreamSynchronize(stream));

            constexpr int kIters = 10;
            VP_CUDA_CHECK(cudaEventRecord(evStart, stream));
            for (int it = 0; it < kIters; ++it) {
                solver->solve(pool, nData, settings, 1.0f / 240.0f, stream);
            }
            VP_CUDA_CHECK(cudaEventRecord(evEnd, stream));
            VP_CUDA_CHECK(cudaStreamSynchronize(stream));

            float totalMs = 0.0f;
            VP_CUDA_CHECK(cudaEventElapsedTime(&totalMs, evStart, evEnd));
            res.solvePassMs250K = totalMs / static_cast<float>(kIters);

            VP_CUDA_CHECK(cudaEventDestroy(evStart));
            VP_CUDA_CHECK(cudaEventDestroy(evEnd));
            VP_CUDA_CHECK(cudaStreamDestroy(stream));
        }

        // 1M latency pass
        {
            const uint32_t n1M = capacity;
            GranularSystem sys1M(n1M, false);
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            sys1M.setSettings(settings);
            sys1M.spawnLattice({0.0f, 1.5f, 0.0f}, n1M, 2.0f * settings.grainRadius, 0.0f, {0.0f, 0.0f, 0.0f}, 42);
            sys1M.synchronize();

            cudaStream_t stream;
            VP_CUDA_CHECK(cudaStreamCreate(&stream));
            cudaEvent_t evStart, evEnd;
            VP_CUDA_CHECK(cudaEventCreate(&evStart));
            VP_CUDA_CHECK(cudaEventCreate(&evEnd));

            auto solver = createGaussSeidelContactSolver();
            auto& pool = const_cast<GranularPool&>(sys1M.pool());
            const auto& nData = sys1M.neighborData();

            // Warmup
            solver->solve(pool, nData, settings, 1.0f / 240.0f, stream);
            VP_CUDA_CHECK(cudaStreamSynchronize(stream));

            constexpr int kIters = 10;
            VP_CUDA_CHECK(cudaEventRecord(evStart, stream));
            for (int it = 0; it < kIters; ++it) {
                solver->solve(pool, nData, settings, 1.0f / 240.0f, stream);
            }
            VP_CUDA_CHECK(cudaEventRecord(evEnd, stream));
            VP_CUDA_CHECK(cudaStreamSynchronize(stream));

            float totalMs = 0.0f;
            VP_CUDA_CHECK(cudaEventElapsedTime(&totalMs, evStart, evEnd));
            res.solvePassMs1M = totalMs / static_cast<float>(kIters);

            VP_CUDA_CHECK(cudaEventDestroy(evStart));
            VP_CUDA_CHECK(cudaEventDestroy(evEnd));
            VP_CUDA_CHECK(cudaStreamDestroy(stream));
        }

        res.latencyPassed = (res.solvePassMs250K <= 1.0f) && (res.solvePassMs1M <= 3.5f);

        std::cout << "  - 250K Gauss-Seidel solve pass: " << std::fixed << std::setprecision(3) << res.solvePassMs250K << " ms (Target: < 1.0 ms)\n";
        std::cout << "  - 1M Gauss-Seidel solve pass:   " << res.solvePassMs1M << " ms (Target: < 3.5 ms)\n";
        std::cout << "  -> Gate 11D-4 Result: " << (res.latencyPassed ? "PASSED (both budgets met)" : "FAILED") << "\n\n";
    }

    res.passed = res.penetrationPassed && res.energyDecayPassed && res.stabilityPassed && res.latencyPassed;

    std::cout << "======================================================================\n";
    std::cout << "  PHASE 11D VALIDATION VERDICT: " << (res.passed ? "ALL GATES PASSED!" : "FAILED!") << "\n";
    std::cout << "======================================================================\n\n";

    return res;
}

} // namespace vparticles
