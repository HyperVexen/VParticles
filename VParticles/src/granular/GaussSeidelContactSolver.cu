#include "VParticles/ContactSolver.h"
#include "VParticles/GranularTypes.h"
#include "VParticles/NeighborTypes.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <memory>

namespace vparticles {

namespace {

constexpr int kThreads = 256;

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float3 makeFloat3(const Float3 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul3(const float3 v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }

struct ContactSolverParams {
    CollisionPlane groundPlane;
    CollisionSphere obstacleSphere;
    CollisionBox containerBox;
    GrainMaterial material;
    GrainMaterial materials[kMaxGrainMaterials];
    uint32_t materialCount;
    float grainRadius;
    float substepDt;
    float invSubstepDt;
    bool enableSleeping;
    float sleepSpeedThreshold;
};

// -----------------------------------------------------------------------------
// XPBD Cell-Colored Gauss-Seidel Kernel (In-Place Update for active color)
// -----------------------------------------------------------------------------
static __global__ void solveContactsColoredGSKernel(
    float4* __restrict__ pred,
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    const uint32_t* __restrict__ colorIndices,
    uint32_t colorStart,
    uint32_t colorEnd,
    const uint32_t* __restrict__ neighborList,
    const uint32_t* __restrict__ neighborCount,
    uint16_t* __restrict__ flags,
    const uint16_t* __restrict__ materialId,
    ContactSolverParams params)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t colorGrainCount = colorEnd - colorStart;
    if (idx >= colorGrainCount) return;

    const uint32_t i = colorIndices ? colorIndices[colorStart + idx] : (colorStart + idx);

    // Skip asleep grains
    if (flags && (flags[i] & kGrainFlagAsleep)) return;

    const float4 pi4 = pred[i];
    float3 pi = makeFloat3(pi4);
    const float ri = pi4.w > 0.0f ? pi4.w : params.grainRadius;

    const float invMassI = (vel && vel[i].w > 0.0f) ? vel[i].w : 1.0f;

    const uint32_t count = neighborCount ? neighborCount[i] : 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    float3 totalFriction = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t contacts = 0;

    const uint16_t matId = materialId ? materialId[i] : 0;
    const GrainMaterial& mat = (matId < params.materialCount) ? params.materials[matId] : params.material;
    const float alphaTilde = (mat.cohesion > 0.0f ? 0.0f : (1.0f - mat.repulsion) * 1e-6f) * (params.invSubstepDt * params.invSubstepDt);
    const float3 oldPi = pos ? makeFloat3(pos[i]) : pi;

    if (neighborList && count > 0) {
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t j = neighborList[baseOffset + k];
            if (j == i) continue;

            // In Gauss-Seidel, we read pred[j] which reflects latest updates from earlier colors
            const float4 pj4 = pred[j];
            const float3 pj = makeFloat3(pj4);
            const float rj = pj4.w > 0.0f ? pj4.w : params.grainRadius;

            const float3 diff = sub3(pi, pj);
            const float distSq = dot3(diff, diff);
            const float targetDist = ri + rj;
            const float ruptureRatio = (mat.cohesionRuptureRatio > 0.0f) ? mat.cohesionRuptureRatio : 0.15f;
            const float maxDist = (mat.cohesion > 0.0f) ? (targetDist * (1.0f + ruptureRatio)) : targetDist;
            const float maxDistSq = maxDist * maxDist;

            if (distSq < maxDistSq && distSq > 1e-12f) {
                // Wake up asleep neighbor on active contact only if grain i has dynamic velocity (impact/avalanche)
                if (flags && (flags[j] & kGrainFlagAsleep) && params.enableSleeping) {
                    const float4 vi4 = vel ? vel[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    const float velSqI = vi4.x * vi4.x + vi4.y * vi4.y + vi4.z * vi4.z;
                    if (velSqI > (4.0f * params.sleepSpeedThreshold) * (4.0f * params.sleepSpeedThreshold)) {
                        flags[j] = 0;
                    }
                }

                const float dist = sqrtf(distSq);
                const float3 n = mul3(diff, 1.0f / dist);

                const float invMassJ = (vel && vel[j].w > 0.0f) ? vel[j].w : 1.0f;
                const float denom = invMassI + invMassJ + alphaTilde;

                constexpr float kSOR = 1.45f;
                float deltaLambda = 0.0f;
                float3 deltaX = make_float3(0.0f, 0.0f, 0.0f);

                if (dist < targetDist) {
                    const float penetration = targetDist - dist;
                    deltaLambda = (kSOR * penetration) / denom;
                    deltaX = mul3(n, deltaLambda * invMassI);
                }

                // Cohesion (compliant XPBD capillary attraction across meniscus gap)
                if (mat.cohesion > 0.0f && dist >= targetDist) {
                    const float maxCapGap = targetDist * ruptureRatio;
                    const float gap = dist - targetDist;
                    if (gap <= maxCapGap) {
                        const float alphaCohesion = (50.0f / mat.cohesion) * (params.invSubstepDt * params.invSubstepDt);
                        const float denomCohesion = invMassI + invMassJ + alphaCohesion;
                        const float deltaLambdaCohesion = -gap / denomCohesion;
                        deltaX = mul3(n, deltaLambdaCohesion * invMassI);
                    }
                }

                // Position-level Coulomb friction (relative to start-of-substep displacement)
                // In cohesive granular materials, capillary cohesion adds to the effective normal load (Mohr-Coulomb criterion)
                if (mat.staticFriction > 0.0f || mat.dynamicFriction > 0.0f) {
                    const float3 dXi = sub3(pi, oldPi);
                    const float3 dXt = sub3(dXi, mul3(n, dot3(dXi, n)));
                    const float dXtMag = sqrtf(dot3(dXt, dXt));

                    if (dXtMag > 1e-8f) {
                        const float cohesionNormalLoad = (mat.cohesion > 0.0f) ? (mat.cohesion * 0.0001f) : 0.0f;
                        const float normalEff = fmaxf(deltaLambda * invMassI + cohesionNormalLoad, fabsf(dot3(dXi, n)));
                        const float maxFricDisp = mat.staticFriction * normalEff;
                        const float dynFricDisp = mat.dynamicFriction * normalEff;
                        const float fricLimit = (dXtMag <= maxFricDisp) ? dXtMag : dynFricDisp;
                        totalFriction = add3(totalFriction, mul3(dXt, fricLimit / dXtMag));
                    }
                }

                totalDelta = add3(totalDelta, deltaX);
                contacts++;
            }
        }
    }

    // Multi-contact friction clamping: total friction opposing displacement can never exceed dXi
    const float3 dXi = sub3(pi, oldPi);
    const float dXiMag = sqrtf(dot3(dXi, dXi));
    const float fMag = sqrtf(dot3(totalFriction, totalFriction));
    if (fMag > dXiMag && fMag > 1e-8f) {
        totalFriction = mul3(totalFriction, dXiMag / fMag);
    }

    pi = add3(pi, totalDelta);
    pi = sub3(pi, totalFriction);

    // Colliders inside the same solve iteration

    // 1. Ground Plane Collision
    if (params.groundPlane.enabled) {
        const float3 pPlane = makeFloat3(params.groundPlane.point);
        const float3 nPlane = makeFloat3(params.groundPlane.normal);
        const float dist = dot3(sub3(pi, pPlane), nPlane);
        const float planePen = ri - dist;

        if (planePen > 0.0f) {
            pi = add3(pi, mul3(nPlane, planePen));

            // Tangential ground friction for grains touching the plane
            if (pos && params.groundPlane.friction > 0.0f) {
                const float3 dXPlane = sub3(pi, makeFloat3(pos[i]));
                const float3 dXtPlane = sub3(dXPlane, mul3(nPlane, dot3(dXPlane, nPlane)));
                const float dXtMagPlane = sqrtf(dot3(dXtPlane, dXtPlane));
                if (dXtMagPlane > 1e-8f) {
                    const float frictionFactor = fminf(1.0f, params.groundPlane.friction * 1.8f);
                    pi = sub3(pi, mul3(dXtPlane, frictionFactor));
                }
            }
        }
    }

    // 2. Container Box Confinement
    if (params.containerBox.enabled) {
        const float3 minB = makeFloat3(params.containerBox.minBounds);
        const float3 maxB = makeFloat3(params.containerBox.maxBounds);
        pi.x = fminf(fmaxf(pi.x, minB.x + ri), maxB.x - ri);
        pi.y = fminf(fmaxf(pi.y, minB.y + ri), maxB.y - ri);
        pi.z = fminf(fmaxf(pi.z, minB.z + ri), maxB.z - ri);
    }

    // 3. Obstacle Sphere
    if (params.obstacleSphere.enabled) {
        const float3 cSph = makeFloat3(params.obstacleSphere.center);
        const float3 diffSph = sub3(pi, cSph);
        const float distSq = dot3(diffSph, diffSph);
        if (params.obstacleSphere.invert) {
            const float maxR = params.obstacleSphere.radius - ri;
            if (distSq > maxR * maxR && distSq > 1e-12f) {
                const float dist = sqrtf(distSq);
                pi = add3(cSph, mul3(diffSph, maxR / dist));
                if (flags && (flags[i] & kGrainFlagAsleep)) flags[i] = 0;
            }
        } else {
            const float minR = params.obstacleSphere.radius + ri;
            if (distSq < minR * minR && distSq > 1e-12f) {
                const float dist = sqrtf(distSq);
                pi = add3(cSph, mul3(diffSph, minR / dist));
                if (flags && (flags[i] & kGrainFlagAsleep)) flags[i] = 0;
            }
        }
    }

    pred[i] = make_float4(pi.x, pi.y, pi.z, pi4.w);
}

} // namespace

class GaussSeidelContactSolver : public IContactSolver {
public:
    const char* name() const override { return "CellColoredGaussSeidel"; }

    void solve(
        GranularPool& pool,
        const NeighborData& neighbors,
        const GranularSettings& settings,
        float substepDt,
        cudaStream_t stream) override
    {
        if (pool.aliveCount == 0) return;

        ContactSolverParams params;
        params.groundPlane = settings.groundPlane;
        params.obstacleSphere = settings.obstacleSphere;
        params.containerBox = settings.containerBox;
        params.material = settings.materials[0];
        params.materialCount = settings.materialCount;
        for (uint32_t m = 0; m < settings.materialCount && m < kMaxGrainMaterials; ++m) {
            params.materials[m] = settings.materials[m];
        }
        params.grainRadius = settings.grainRadius;
        params.substepDt = substepDt;
        params.invSubstepDt = 1.0f / substepDt;
        params.enableSleeping = settings.enableSleeping;
        params.sleepSpeedThreshold = settings.sleepSpeedThreshold;

        // If color indices are not ready, execute single parallel pass
        if (neighbors.colorIndices == nullptr || neighbors.hostColorOffsets[8] == 0) {
            const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;
            solveContactsColoredGSKernel<<<blocks, kThreads, 0, stream>>>(
                pool.pred, pool.pos, pool.vel,
                nullptr, 0, pool.aliveCount,
                neighbors.list, neighbors.counts, pool.flags, pool.materialId,
                params);
            return;
        }

        // 8-color cell-partitioned Gauss-Seidel solve
        for (uint32_t c = 0; c < 8; ++c) {
            const uint32_t colorStart = neighbors.hostColorOffsets[c];
            const uint32_t colorEnd = neighbors.hostColorOffsets[c + 1];
            if (colorEnd <= colorStart) continue;

            const uint32_t count = colorEnd - colorStart;
            const uint32_t blocks = (count + kThreads - 1) / kThreads;

            solveContactsColoredGSKernel<<<blocks, kThreads, 0, stream>>>(
                pool.pred, pool.pos, pool.vel,
                neighbors.colorIndices, colorStart, colorEnd,
                neighbors.list, neighbors.counts, pool.flags, pool.materialId,
                params);
        }
    }
};

std::unique_ptr<IContactSolver> createGaussSeidelContactSolver()
{
    return std::make_unique<GaussSeidelContactSolver>();
}

} // namespace vparticles
