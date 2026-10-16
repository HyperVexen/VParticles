#include "granular/GranularBondManager.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vparticles {

namespace {

constexpr int kThreads = 256;

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 mul3(const float3 v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }

__device__ inline uint32_t hash3D(int x, int y, int z, uint32_t seed)
{
    uint32_t h = seed ^ (static_cast<uint32_t>(x) * 73856093u);
    h ^= (static_cast<uint32_t>(y) * 19349663u);
    h ^= (static_cast<uint32_t>(z) * 83492791u);
    h ^= (h >> 13);
    h *= 0x85ebca6bu;
    h ^= (h >> 16);
    return h;
}

// -----------------------------------------------------------------------------
// 1. Initialize grainIdToIndex
// -----------------------------------------------------------------------------
__global__ void initGrainIdToIndexKernel(
    const uint32_t* __restrict__ grainId,
    uint32_t* __restrict__ grainIdToIndex,
    uint32_t aliveCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= aliveCount) return;

    const uint32_t gid = grainId ? grainId[i] : i;
    grainIdToIndex[gid] = i;
}

// -----------------------------------------------------------------------------
// 2. Create Lattice Bonds (Tiled N-body style)
// -----------------------------------------------------------------------------
__global__ void createLatticeBondsTiledKernel(
    const float4* __restrict__ pos,
    const uint32_t* __restrict__ grainId,
    GrainBond* __restrict__ bonds,
    uint32_t* __restrict__ bondCounts,
    uint32_t aliveCount,
    float maxDist,
    float compliance,
    float breakForce)
{
    __shared__ float4 sPos[kThreads];
    __shared__ uint32_t sGid[kThreads];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pi4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 pi = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t count = 0;
    const float maxDistSq = maxDist * maxDist;

    if (i < aliveCount) {
        pi4 = pos[i];
        pi = makeFloat3(pi4);
    }

    const uint32_t numTiles = (aliveCount + kThreads - 1) / kThreads;

    for (uint32_t tile = 0; tile < numTiles; ++tile) {
        const uint32_t tileIdx = tile * kThreads + threadIdx.x;
        if (tileIdx < aliveCount) {
            sPos[threadIdx.x] = pos[tileIdx];
            sGid[threadIdx.x] = grainId ? grainId[tileIdx] : tileIdx;
        }
        __syncthreads();

        if (i < aliveCount) {
            const uint32_t tileEnd = min(static_cast<uint32_t>(kThreads), aliveCount - tile * kThreads);
            for (uint32_t t = 0; t < tileEnd; ++t) {
                const uint32_t j = tile * kThreads + t;
                if (j == i) continue;

                const float3 pj = makeFloat3(sPos[t]);
                const float3 diff = sub3(pi, pj);
                const float distSq = dot3(diff, diff);

                if (distSq <= maxDistSq && distSq > 1e-10f) {
                    if (count < kMaxBondsPerGrain) {
                        const float dist = sqrtf(distSq);
                        GrainBond b;
                        b.targetGrainId = sGid[t];
                        b.restLength = dist;
                        b.compliance = compliance;
                        b.breakForce = breakForce;
                        bonds[i * kMaxBondsPerGrain + count] = b;
                        count++;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < aliveCount) {
        bondCounts[i] = count;
    }
}

// -----------------------------------------------------------------------------
// 3. Create Clump Bonds (Cluster grains by Voronoi/cell clumps)
// -----------------------------------------------------------------------------
__global__ void createClumpBondsTiledKernel(
    const float4* __restrict__ pos,
    const uint32_t* __restrict__ grainId,
    GrainBond* __restrict__ bonds,
    uint32_t* __restrict__ bondCounts,
    uint32_t aliveCount,
    float clumpRadius,
    float maxDist,
    float compliance,
    float breakForce,
    uint32_t seed)
{
    __shared__ float4 sPos[kThreads];
    __shared__ uint32_t sGid[kThreads];
    __shared__ uint32_t sClump[kThreads];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float4 pi4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 pi = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t myClump = 0;
    uint32_t count = 0;
    const float maxDistSq = maxDist * maxDist;
    const float clumpDiameter = 2.0f * clumpRadius;
    const float invClumpDiameter = 1.0f / clumpDiameter;

    if (i < aliveCount) {
        pi4 = pos[i];
        pi = makeFloat3(pi4);
        const int cx = static_cast<int>(floorf(pi.x * invClumpDiameter));
        const int cy = static_cast<int>(floorf(pi.y * invClumpDiameter));
        const int cz = static_cast<int>(floorf(pi.z * invClumpDiameter));
        myClump = hash3D(cx, cy, cz, seed);
    }

    const uint32_t numTiles = (aliveCount + kThreads - 1) / kThreads;

    for (uint32_t tile = 0; tile < numTiles; ++tile) {
        const uint32_t tileIdx = tile * kThreads + threadIdx.x;
        if (tileIdx < aliveCount) {
            const float4 tp4 = pos[tileIdx];
            sPos[threadIdx.x] = tp4;
            sGid[threadIdx.x] = grainId ? grainId[tileIdx] : tileIdx;
            const int cx = static_cast<int>(floorf(tp4.x * invClumpDiameter));
            const int cy = static_cast<int>(floorf(tp4.y * invClumpDiameter));
            const int cz = static_cast<int>(floorf(tp4.z * invClumpDiameter));
            sClump[threadIdx.x] = hash3D(cx, cy, cz, seed);
        }
        __syncthreads();

        if (i < aliveCount) {
            const uint32_t tileEnd = min(static_cast<uint32_t>(kThreads), aliveCount - tile * kThreads);
            for (uint32_t t = 0; t < tileEnd; ++t) {
                const uint32_t j = tile * kThreads + t;
                if (j == i) continue;

                // Form bond ONLY if grains belong to the same clump cluster
                if (sClump[t] == myClump) {
                    const float3 pj = makeFloat3(sPos[t]);
                    const float3 diff = sub3(pi, pj);
                    const float distSq = dot3(diff, diff);

                    if (distSq <= maxDistSq && distSq > 1e-10f) {
                        if (count < kMaxBondsPerGrain) {
                            const float dist = sqrtf(distSq);
                            GrainBond b;
                            b.targetGrainId = sGid[t];
                            b.restLength = dist;
                            b.compliance = compliance;
                            b.breakForce = breakForce;
                            bonds[i * kMaxBondsPerGrain + count] = b;
                            count++;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (i < aliveCount) {
        bondCounts[i] = count;
    }
}

// -----------------------------------------------------------------------------
// 4. Reorder Bonds Kernel (Permutes bonds and updates grainIdToIndex)
// -----------------------------------------------------------------------------
__global__ void reorderBondsKernel(
    const GrainBond* __restrict__ inBonds,
    const uint32_t* __restrict__ inBondCounts,
    const uint32_t* __restrict__ sortedIndices,
    const uint32_t* __restrict__ sortedGrainId,
    GrainBond* __restrict__ outBonds,
    uint32_t* __restrict__ outBondCounts,
    uint32_t* __restrict__ grainIdToIndex,
    uint32_t aliveCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= aliveCount) return;

    const uint32_t origIdx = sortedIndices[i];
    const uint32_t count = inBondCounts[origIdx];
    outBondCounts[i] = count;

    const uint32_t srcBase = origIdx * kMaxBondsPerGrain;
    const uint32_t dstBase = i * kMaxBondsPerGrain;

    for (uint32_t k = 0; k < kMaxBondsPerGrain; ++k) {
        outBonds[dstBase + k] = inBonds[srcBase + k];
    }

    const uint32_t gid = sortedGrainId[i];
    grainIdToIndex[gid] = i;
}

// -----------------------------------------------------------------------------
// 5. XPBD Distance Constraint Solve Kernel
// -----------------------------------------------------------------------------
__global__ void solveBondsKernel(
    float4* __restrict__ pred,
    const float4* __restrict__ vel,
    const uint32_t* __restrict__ grainIdToIndex,
    GrainBond* __restrict__ bonds,
    const uint32_t* __restrict__ bondCounts,
    const uint16_t* __restrict__ flags,
    uint32_t aliveCount,
    float substepDt,
    float invSubstepDt,
    uint32_t* __restrict__ d_brokenCounter)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= aliveCount) return;

    if (flags && (flags[i] & kGrainFlagAsleep)) return;

    const uint32_t count = bondCounts ? bondCounts[i] : 0;
    if (count == 0) return;

    const float4 pi4 = pred[i];
    float3 pi = makeFloat3(pi4);
    const float invMassI = (vel && vel[i].w >= 0.0f) ? vel[i].w : 1.0f;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t activeEvaluated = 0;

    const uint32_t baseOffset = i * kMaxBondsPerGrain;
    const float invDtSq = invSubstepDt * invSubstepDt;

    for (uint32_t k = 0; k < count; ++k) {
        GrainBond& b = bonds[baseOffset + k];
        if (b.targetGrainId == 0xFFFFFFFF) continue;

        const uint32_t targetGid = b.targetGrainId;
        const uint32_t j = grainIdToIndex[targetGid];
        if (j >= aliveCount || j == i) continue;

        const float4 pj4 = pred[j];
        const float3 pj = makeFloat3(pj4);
        const float invMassJ = (vel && vel[j].w >= 0.0f) ? vel[j].w : 1.0f;

        const float3 diff = sub3(pi, pj);
        const float distSq = dot3(diff, diff);
        if (distSq < 1e-12f) continue;

        const float dist = sqrtf(distSq);
        const float C = dist - b.restLength;
        const float alphaTilde = b.compliance * invDtSq;
        const float denom = invMassI + invMassJ + alphaTilde;
        if (denom <= 1e-12f) continue;

        const float deltaLambda = -C / denom;

        // Rupture / break check on tensile stress (elongation)
        if (C > 0.0f && b.breakForce > 0.0f) {
            const float forceTensile = fabsf(deltaLambda) * invDtSq;
            if (forceTensile > b.breakForce) {
                // Break this bond
                b.targetGrainId = 0xFFFFFFFF;
                if (d_brokenCounter) atomicAdd(d_brokenCounter, 1);
                continue;
            }
        }

        // Apply XPBD displacement
        if (invMassI > 0.0f) {
            const float3 n = mul3(diff, 1.0f / dist);
            const float3 deltaX = mul3(n, -invMassI * deltaLambda);
            totalDelta = add3(totalDelta, deltaX);
            activeEvaluated++;
        }
    }

    if (activeEvaluated > 0 && invMassI > 0.0f) {
        const float relaxation = 1.0f / fmaxf(1.0f, static_cast<float>(activeEvaluated) * 0.4f);
        pi = add3(pi, mul3(totalDelta, relaxation));
        pred[i] = make_float4(pi.x, pi.y, pi.z, pi4.w);
    }
}

// -----------------------------------------------------------------------------
// 6. Active Bonds Count Reduction
// -----------------------------------------------------------------------------
__global__ void countActiveBondsKernel(
    const GrainBond* __restrict__ bonds,
    const uint32_t* __restrict__ bondCounts,
    uint32_t aliveCount,
    uint32_t* __restrict__ d_activeCounter)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= aliveCount) return;

    const uint32_t count = bondCounts[i];
    const uint32_t baseOffset = i * kMaxBondsPerGrain;
    uint32_t active = 0;

    for (uint32_t k = 0; k < count; ++k) {
        if (bonds[baseOffset + k].targetGrainId != 0xFFFFFFFF) {
            active++;
        }
    }

    if (active > 0) {
        atomicAdd(d_activeCounter, active);
    }
}

} // namespace

struct GranularBondManager::Impl {
    uint32_t capacity = 0;

    GrainBond* d_bonds = nullptr;
    GrainBond* d_scratchBonds = nullptr;
    uint32_t* d_bondCounts = nullptr;
    uint32_t* d_scratchBondCounts = nullptr;
    uint32_t* d_grainIdToIndex = nullptr;
    uint32_t* d_brokenCounter = nullptr;
    uint32_t* d_activeCounter = nullptr;

    explicit Impl(uint32_t cap)
        : capacity(cap)
    {
        const size_t bondBytes = sizeof(GrainBond) * cap * kMaxBondsPerGrain;
        VP_CUDA_CHECK(cudaMalloc(&d_bonds, bondBytes));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchBonds, bondBytes));
        VP_CUDA_CHECK(cudaMalloc(&d_bondCounts, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_scratchBondCounts, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_grainIdToIndex, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_brokenCounter, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&d_activeCounter, sizeof(uint32_t)));

        reset();
    }

    ~Impl()
    {
        if (d_activeCounter) cudaFree(d_activeCounter);
        if (d_brokenCounter) cudaFree(d_brokenCounter);
        if (d_grainIdToIndex) cudaFree(d_grainIdToIndex);
        if (d_scratchBondCounts) cudaFree(d_scratchBondCounts);
        if (d_bondCounts) cudaFree(d_bondCounts);
        if (d_scratchBonds) cudaFree(d_scratchBonds);
        if (d_bonds) cudaFree(d_bonds);
    }

    void reset()
    {
        const size_t bondBytes = sizeof(GrainBond) * capacity * kMaxBondsPerGrain;
        VP_CUDA_CHECK(cudaMemset(d_bonds, 0xFF, bondBytes));
        VP_CUDA_CHECK(cudaMemset(d_bondCounts, 0, sizeof(uint32_t) * capacity));
        VP_CUDA_CHECK(cudaMemset(d_brokenCounter, 0, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_activeCounter, 0, sizeof(uint32_t)));
    }
};

GranularBondManager::GranularBondManager(uint32_t capacity)
    : impl_(std::make_unique<Impl>(capacity))
{
}

GranularBondManager::~GranularBondManager() = default;

void GranularBondManager::reset()
{
    impl_->reset();
}

void GranularBondManager::createLatticeBonds(
    const GranularPool& pool,
    float maxDist,
    float compliance,
    float breakForce,
    cudaStream_t stream)
{
    if (pool.aliveCount == 0) return;

    const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;

    initGrainIdToIndexKernel<<<blocks, kThreads, 0, stream>>>(
        pool.grainId, impl_->d_grainIdToIndex, pool.aliveCount);

    createLatticeBondsTiledKernel<<<blocks, kThreads, 0, stream>>>(
        pool.pos, pool.grainId, impl_->d_bonds, impl_->d_bondCounts,
        pool.aliveCount, maxDist, compliance, breakForce);

    VP_CUDA_CHECK(cudaGetLastError());
}

void GranularBondManager::createClumpBonds(
    const GranularPool& pool,
    float clumpRadius,
    float maxDist,
    float compliance,
    float breakForce,
    uint32_t seed,
    cudaStream_t stream)
{
    if (pool.aliveCount == 0) return;

    const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;

    initGrainIdToIndexKernel<<<blocks, kThreads, 0, stream>>>(
        pool.grainId, impl_->d_grainIdToIndex, pool.aliveCount);

    createClumpBondsTiledKernel<<<blocks, kThreads, 0, stream>>>(
        pool.pos, pool.grainId, impl_->d_bonds, impl_->d_bondCounts,
        pool.aliveCount, clumpRadius, maxDist, compliance, breakForce, seed);

    VP_CUDA_CHECK(cudaGetLastError());
}

void GranularBondManager::reorder(
    const uint32_t* sortedIndices,
    const uint32_t* sortedGrainId,
    uint32_t aliveCount,
    cudaStream_t stream)
{
    if (aliveCount == 0) return;

    const uint32_t blocks = (aliveCount + kThreads - 1) / kThreads;

    reorderBondsKernel<<<blocks, kThreads, 0, stream>>>(
        impl_->d_bonds, impl_->d_bondCounts,
        sortedIndices, sortedGrainId,
        impl_->d_scratchBonds, impl_->d_scratchBondCounts,
        impl_->d_grainIdToIndex, aliveCount);

    std::swap(impl_->d_bonds, impl_->d_scratchBonds);
    std::swap(impl_->d_bondCounts, impl_->d_scratchBondCounts);

    VP_CUDA_CHECK(cudaGetLastError());
}

void GranularBondManager::solveBonds(
    GranularPool& pool,
    float substepDt,
    cudaStream_t stream)
{
    if (pool.aliveCount == 0) return;

    const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;
    const float invSubstepDt = 1.0f / substepDt;

    solveBondsKernel<<<blocks, kThreads, 0, stream>>>(
        pool.pred, pool.vel, impl_->d_grainIdToIndex,
        impl_->d_bonds, impl_->d_bondCounts, pool.flags,
        pool.aliveCount, substepDt, invSubstepDt, impl_->d_brokenCounter);

    VP_CUDA_CHECK(cudaGetLastError());
}

uint32_t GranularBondManager::getActiveBondCount(uint32_t aliveCount, cudaStream_t stream)
{
    if (aliveCount == 0) return 0;

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_activeCounter, 0, sizeof(uint32_t), stream));

    const uint32_t blocks = (aliveCount + kThreads - 1) / kThreads;
    countActiveBondsKernel<<<blocks, kThreads, 0, stream>>>(
        impl_->d_bonds, impl_->d_bondCounts, aliveCount, impl_->d_activeCounter);

    uint32_t hostCount = 0;
    VP_CUDA_CHECK(cudaMemcpyAsync(&hostCount, impl_->d_activeCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Each symmetric bond is recorded at both grains, so true unique bonds = count / 2
    return hostCount / 2;
}

uint32_t GranularBondManager::getBrokenBondCount(cudaStream_t stream)
{
    uint32_t hostCount = 0;
    VP_CUDA_CHECK(cudaMemcpyAsync(&hostCount, impl_->d_brokenCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(stream));
    return hostCount / 2;
}

const GrainBond* GranularBondManager::bonds() const
{
    return impl_->d_bonds;
}

const uint32_t* GranularBondManager::bondCounts() const
{
    return impl_->d_bondCounts;
}

const uint32_t* GranularBondManager::grainIdToIndex() const
{
    return impl_->d_grainIdToIndex;
}

} // namespace vparticles
