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
    float grainRadius;
    float substepDt;
    float invSubstepDt;
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
    const uint16_t* __restrict__ flags,
    ContactSolverParams params)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t colorGrainCount = colorEnd - colorStart;
    if (idx >= colorGrainCount) return;

    const uint32_t i = colorIndices ? colorIndices[colorStart + idx] : (colorStart + idx);

    // Skip asleep grains
    if (flags && (flags[i] & 1)) return;

    const float4 pi4 = pred[i];
    float3 pi = makeFloat3(pi4);
    const float ri = pi4.w > 0.0f ? pi4.w : params.grainRadius;

    const float invMassI = (vel && vel[i].w > 0.0f) ? vel[i].w : 1.0f;

    const uint32_t count = neighborCount ? neighborCount[i] : 0;
    const uint32_t baseOffset = i * kMaxNeighborsPerGrain;

    float3 totalDelta = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t contacts = 0;

    const GrainMaterial& mat = params.material;
    const float alphaTilde = (mat.cohesion > 0.0f ? 0.0f : (1.0f - mat.repulsion) * 1e-6f) * (params.invSubstepDt * params.invSubstepDt);
    const float denom = 2.0f * invMassI + alphaTilde;
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

            if (distSq < targetDist * targetDist && distSq > 1e-12f) {
                const float dist = sqrtf(distSq);
                const float penetration = targetDist - dist;
                const float3 n = mul3(diff, 1.0f / dist);

                constexpr float kSOR = 1.45f;
                const float deltaLambda = (kSOR * penetration) / denom;
                float3 deltaX = mul3(n, deltaLambda * invMassI);

                // Position-level Coulomb friction (relative to start-of-substep displacement)
                if (mat.staticFriction > 0.0f || mat.dynamicFriction > 0.0f) {
                    const float3 dXi = sub3(pi, oldPi);
                    const float3 dXt = sub3(dXi, mul3(n, dot3(dXi, n)));
                    const float dXtMag = sqrtf(dot3(dXt, dXt));

                    if (dXtMag > 1e-8f) {
                        const float maxFriction = mat.staticFriction * deltaLambda * invMassI;
                        if (dXtMag <= maxFriction) {
                            deltaX = sub3(deltaX, mul3(dXt, 0.5f));
                        } else {
                            const float dynamicLimit = mat.dynamicFriction * deltaLambda * invMassI;
                            deltaX = sub3(deltaX, mul3(dXt, (dynamicLimit / dXtMag) * 0.5f));
                        }
                    }
                }

                totalDelta = add3(totalDelta, deltaX);
                contacts++;
            }
        }
    }

    pi = add3(pi, totalDelta);

    // Colliders inside the same solve iteration

    // 1. Ground Plane Collision
    if (params.groundPlane.enabled) {
        const float3 pPlane = makeFloat3(params.groundPlane.point);
        const float3 nPlane = makeFloat3(params.groundPlane.normal);
        const float dist = dot3(sub3(pi, pPlane), nPlane);
        const float planePen = ri - dist;

        if (planePen > 0.0f) {
            pi = add3(pi, mul3(nPlane, planePen));

            // Tangential ground friction
            if (pos && params.groundPlane.friction > 0.0f) {
                const float3 dXi = sub3(pi, makeFloat3(pos[i]));
                const float3 dXt = sub3(dXi, mul3(nPlane, dot3(dXi, nPlane)));
                const float dXtMag = sqrtf(dot3(dXt, dXt));
                if (dXtMag > 1e-8f) {
                    const float maxPlaneFric = params.groundPlane.friction * planePen;
                    if (dXtMag <= maxPlaneFric) {
                        pi = sub3(pi, dXt);
                    } else {
                        pi = sub3(pi, mul3(dXt, maxPlaneFric / dXtMag));
                    }
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
            }
        } else {
            const float minR = params.obstacleSphere.radius + ri;
            if (distSq < minR * minR && distSq > 1e-12f) {
                const float dist = sqrtf(distSq);
                pi = add3(cSph, mul3(diffSph, minR / dist));
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
        params.grainRadius = settings.grainRadius;
        params.substepDt = substepDt;
        params.invSubstepDt = 1.0f / substepDt;

        // If color indices are not ready, execute single parallel pass
        if (neighbors.colorIndices == nullptr || neighbors.hostColorOffsets[8] == 0) {
            const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;
            solveContactsColoredGSKernel<<<blocks, kThreads, 0, stream>>>(
                pool.pred, pool.pos, pool.vel,
                nullptr, 0, pool.aliveCount,
                neighbors.list, neighbors.counts, pool.flags,
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
                neighbors.list, neighbors.counts, pool.flags,
                params);
        }
    }
};

std::unique_ptr<IContactSolver> createGaussSeidelContactSolver()
{
    return std::make_unique<GaussSeidelContactSolver>();
}

} // namespace vparticles
