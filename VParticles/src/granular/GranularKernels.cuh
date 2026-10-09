#pragma once

#include "VParticles/GranularTypes.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace vparticles {

constexpr int kGranularThreadsPerBlock = 256;

__device__ inline float3 makeFloat3(const float4 v) { return make_float3(v.x, v.y, v.z); }
__device__ inline float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 add3(const float3 a, const float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul3(const float3 v, const float s) { return make_float3(v.x * s, v.y * s, v.z * s); }

// Predict Kernel: Apply external forces and predict unconstrained position x* = x + v * dt
__global__ void granularPredictKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* __restrict__ pred,
    const uint16_t* __restrict__ flags,
    uint32_t count,
    float3 gravity,
    float substepDt,
    float drag)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    if (flags && (flags[i] & 1)) {
        // Asleep grain skips prediction
        pred[i] = pos[i];
        return;
    }

    const float4 p4 = pos[i];
    const float4 v4 = vel[i];
    float3 p = makeFloat3(p4);
    float3 v = makeFloat3(v4);

    // External forces: gravity
    v = add3(v, mul3(gravity, substepDt));

    // Linear drag
    if (drag > 0.0f) {
        v = mul3(v, fmaxf(0.0f, 1.0f - drag * substepDt));
    }

    // Predict position
    p = add3(p, mul3(v, substepDt));

    pred[i] = make_float4(p.x, p.y, p.z, p4.w);
}


// Integrate Kernel: Update velocity v = (x* - x) / dt and store new position
__global__ void granularIntegrateKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ pred,
    uint16_t* __restrict__ flags,
    uint32_t count,
    float invSubstepDt,
    float substepDt,
    float damping)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    if (flags && (flags[i] & 1)) {
        // Asleep grain
        vel[i] = make_float4(0.0f, 0.0f, 0.0f, vel[i].w);
        return;
    }

    const float4 oldP4 = pos[i];
    const float4 newP4 = pred[i];
    const float3 oldP = makeFloat3(oldP4);
    const float3 newP = makeFloat3(newP4);

    // v = (x* - x) / dt
    float3 v = mul3(sub3(newP, oldP), invSubstepDt);

    // Apply material velocity damping
    if (damping > 0.0f) {
        v = mul3(v, fmaxf(0.0f, 1.0f - damping * substepDt));
    }

    // Tunneling velocity cap (bound displacement to <= 0.5 * r per substep)
    const float radius = oldP4.w > 0.0f ? oldP4.w : 0.01f;
    const float maxSpeed = (0.5f * radius) * invSubstepDt;
    const float speedSq = dot3(v, v);
    if (speedSq > maxSpeed * maxSpeed) {
        v = mul3(v, maxSpeed / sqrtf(speedSq));
    } else if (speedSq < 1e-4f) {
        v = make_float3(0.0f, 0.0f, 0.0f);
    }

    pos[i] = newP4;
    vel[i] = make_float4(v.x, v.y, v.z, vel[i].w);
}

__device__ inline float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
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

// Stats Reduction Kernel with warp shuffles and shared memory block reduction
__global__ void granularReduceStatsKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    uint32_t count,
    float* __restrict__ outStats) // [0]=sumHeight, [1]=maxHeight, [2]=kineticEnergy
{
    __shared__ float sSumY[8];
    __shared__ float sMaxY[8];
    __shared__ float sSumKE[8];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float y = 0.0f;
    float maxY = 0.0f;
    float ke = 0.0f;

    if (i < count) {
        const float4 p = pos[i];
        const float4 v = vel[i];
        y = p.y;
        maxY = p.y;
        ke = 0.5f * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;

    y = warpReduceSum(y);
    maxY = warpReduceMax(maxY);
    ke = warpReduceSum(ke);

    if (lane == 0) {
        sSumY[warpId] = y;
        sMaxY[warpId] = maxY;
        sSumKE[warpId] = ke;
    }
    __syncthreads();

    if (warpId == 0) {
        const float blockY = (lane < 8) ? sSumY[lane] : 0.0f;
        const float blockMaxY = (lane < 8) ? sMaxY[lane] : 0.0f;
        const float blockKE = (lane < 8) ? sSumKE[lane] : 0.0f;

        const float finalY = warpReduceSum(blockY);
        const float finalMaxY = warpReduceMax(blockMaxY);
        const float finalKE = warpReduceSum(blockKE);

        if (lane == 0) {
            atomicAdd(&outStats[0], finalY);
            atomicMax(reinterpret_cast<int*>(&outStats[1]), __float_as_int(finalMaxY));
            atomicAdd(&outStats[2], finalKE);
        }
    }
}

// Deterministic Hash helper for procedural variation
__device__ inline uint32_t granularHash(uint32_t x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

// Procedural Lattice Spawn Kernel
__global__ void granularSpawnLatticeKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    float4* __restrict__ pred,
    uint32_t* __restrict__ tint,
    uint16_t* __restrict__ materialId,
    uint16_t* __restrict__ flags,
    uint32_t* __restrict__ grainId,
    uint32_t count,
    float3 origin,
    float spacing,
    uint32_t sideX,
    uint32_t sideZ,
    float baseRadius,
    float jitterRatio,
    float3 initialVel,
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t x = i % sideX;
    const uint32_t z = (i / sideX) % sideZ;
    const uint32_t y = i / (sideX * sideZ);

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    const float jx = ((hx & 0xFFFF) / 65535.0f - 0.5f) * 2.0f * jitterRatio * baseRadius;
    const float jy = ((hy & 0xFFFF) / 65535.0f - 0.5f) * 2.0f * jitterRatio * baseRadius;
    const float jz = ((hz & 0xFFFF) / 65535.0f - 0.5f) * 2.0f * jitterRatio * baseRadius;
    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * baseRadius;

    const float px = origin.x + static_cast<float>(x) * spacing + jx;
    const float py = origin.y + static_cast<float>(y) * spacing + jy;
    const float pz = origin.z + static_cast<float>(z) * spacing + jz;
    const float r = baseRadius + jr;

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(initialVel.x, initialVel.y, initialVel.z, 1.0f); // w = invMass = 1.0

    // Subtle natural sand tint: Base golden sand with slight grain variation
    const uint8_t red = 210 + (hx % 40);
    const uint8_t green = 180 + (hy % 35);
    const uint8_t blue = 140 + (hz % 30);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = 0;
    grainId[i] = i; // stable monotonic grain ID
}

} // namespace vparticles
