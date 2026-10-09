#include "VParticles/ContactSolver.h"
#include <cuda_runtime.h>

namespace vparticles {

namespace {

constexpr int kThreads = 256;

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul3(const float3 v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }

__global__ void planeCollisionKernel(
    float4* __restrict__ pred,
    float4* __restrict__ vel,
    uint32_t count,
    CollisionPlane plane,
    float substepDt)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float4 p4 = pred[i];
    float3 p = makeFloat3(p4);
    const float radius = p4.w;

    const float3 planePoint = make_float3(plane.point.x, plane.point.y, plane.point.z);
    const float3 planeNormal = make_float3(plane.normal.x, plane.normal.y, plane.normal.z);

    const float dist = dot3(sub3(p, planePoint), planeNormal);
    const float penetration = radius - dist;

    if (penetration > 0.0f) {
        p = add3(p, mul3(planeNormal, penetration));
        pred[i] = make_float4(p.x, p.y, p.z, radius);

        if (vel) {
            float4 v4 = vel[i];
            float3 v = makeFloat3(v4);
            const float vn = dot3(v, planeNormal);
            if (vn < 0.0f) {
                const float3 vNormal = mul3(planeNormal, -vn * plane.bounce);
                const float3 vTangent = sub3(v, mul3(planeNormal, vn));
                const float3 newVTangent = mul3(vTangent, fmaxf(0.0f, 1.0f - plane.friction));
                const float3 newV = add3(vNormal, newVTangent);
                vel[i] = make_float4(newV.x, newV.y, newV.z, v4.w);
            }
        }
    }
}

} // namespace

class JacobiContactSolver : public IContactSolver {
public:
    const char* name() const override { return "JacobiContactSolver"; }

    void solve(
        GranularPool& pool,
        const NeighborData& neighbors,
        const GranularSettings& settings,
        float substepDt,
        cudaStream_t stream) override
    {
        (void)neighbors; // Unused in analytical plane-only solver
        if (pool.aliveCount == 0) return;

        const uint32_t blocks = (pool.aliveCount + kThreads - 1) / kThreads;

        if (settings.groundPlane.enabled) {
            planeCollisionKernel<<<blocks, kThreads, 0, stream>>>(
                pool.pred, pool.vel, pool.aliveCount, settings.groundPlane, substepDt);
        }
    }
};

std::unique_ptr<IContactSolver> createJacobiContactSolver()
{
    return std::make_unique<JacobiContactSolver>();
}

} // namespace vparticles
