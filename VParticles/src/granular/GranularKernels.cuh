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


// Integrate Kernel: Update velocity v = (x* - x) / dt, apply damping, tunneling cap, and sleep tracking
__global__ void granularIntegrateKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    const float4* __restrict__ pred,
    uint16_t* __restrict__ flags,
    uint32_t count,
    float invSubstepDt,
    float substepDt,
    float damping,
    bool enableSleeping,
    float sleepSpeedThreshold,
    uint32_t sleepThresholdSubsteps)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    if (flags && (flags[i] & kGrainFlagAsleep)) {
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

    // Phase 11E: Sleeping deactivation tracking
    if (flags && enableSleeping) {
        const float sleepSpeedSq = sleepSpeedThreshold * sleepSpeedThreshold;
        if (speedSq < sleepSpeedSq) {
            const uint32_t sleepCounter = (flags[i] >> 8) + 1;
            if (sleepCounter >= sleepThresholdSubsteps) {
                flags[i] = kGrainFlagAsleep | static_cast<uint16_t>(min(255u, sleepCounter) << 8);
                v = make_float3(0.0f, 0.0f, 0.0f);
            } else {
                flags[i] = static_cast<uint16_t>(sleepCounter << 8);
            }
        } else {
            flags[i] = 0; // Active grain, reset sleep counter
        }
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
    const uint16_t* __restrict__ flags,
    uint32_t count,
    float* __restrict__ outStats) // [0]=sumHeight, [1]=maxHeight, [2]=kineticEnergy, [3]=asleepCount
{
    __shared__ float sSumY[8];
    __shared__ float sMaxY[8];
    __shared__ float sSumKE[8];
    __shared__ float sAsleep[8];

    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float y = 0.0f;
    float maxY = 0.0f;
    float ke = 0.0f;
    float asleep = 0.0f;

    if (i < count) {
        const float4 p = pos[i];
        const float4 v = vel[i];
        y = p.y;
        maxY = p.y;
        ke = 0.5f * (v.x * v.x + v.y * v.y + v.z * v.z);
        if (flags && (flags[i] & kGrainFlagAsleep)) {
            asleep = 1.0f;
        }
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warpId = threadIdx.x >> 5;

    y = warpReduceSum(y);
    maxY = warpReduceMax(maxY);
    ke = warpReduceSum(ke);
    asleep = warpReduceSum(asleep);

    if (lane == 0) {
        sSumY[warpId] = y;
        sMaxY[warpId] = maxY;
        sSumKE[warpId] = ke;
        sAsleep[warpId] = asleep;
    }
    __syncthreads();

    if (warpId == 0) {
        const float blockY = (lane < 8) ? sSumY[lane] : 0.0f;
        const float blockMaxY = (lane < 8) ? sMaxY[lane] : 0.0f;
        const float blockKE = (lane < 8) ? sSumKE[lane] : 0.0f;
        const float blockAsleep = (lane < 8) ? sAsleep[lane] : 0.0f;

        const float finalY = warpReduceSum(blockY);
        const float finalMaxY = warpReduceMax(blockMaxY);
        const float finalKE = warpReduceSum(blockKE);
        const float finalAsleep = warpReduceSum(blockAsleep);

        if (lane == 0) {
            atomicAdd(&outStats[0], finalY);
            atomicMax(reinterpret_cast<int*>(&outStats[1]), __float_as_int(finalMaxY));
            atomicAdd(&outStats[2], finalKE);
            atomicAdd(&outStats[3], finalAsleep);
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

// Procedural Lattice Spawn Kernel with volume-scaled mass and material assignment
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
    uint32_t seed,
    uint16_t matId = 0,
    float density = 1600.0f)
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

    const uint16_t grainMat = (matId == 0xFFFF) ? static_cast<uint16_t>(i % 2) : matId;
    const float effDensity = (grainMat == 1) ? 2400.0f : (density > 0.0f ? density : 1600.0f);

    const float jx = 0.0f;
    const float jy = 0.0f;
    const float jz = 0.0f;
    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * 2.0f * jitterRatio * baseRadius;

    const float px = origin.x + static_cast<float>(x) * spacing + jx;
    const float py = origin.y + static_cast<float>(y) * spacing + jy;
    const float pz = origin.z + static_cast<float>(z) * spacing + jz;
    const float r = baseRadius + jr;

    // Normalized volume-scaled inverse mass: w_i = (baseRadius / r)^3 * (1600 / density)
    const float rRatio = baseRadius / (r > 0.0f ? r : baseRadius);
    const float invMass = (rRatio * rRatio * rRatio) * (1600.0f / effDensity);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(initialVel.x, initialVel.y, initialVel.z, invMass);

    // Natural tints per material:
    uint8_t red, green, blue;
    if (grainMat == 1) {
        red = 110 + (hx % 25);
        green = 80 + (hy % 20);
        blue = 60 + (hz % 15);
    } else if (grainMat == 2) {
        red = 140 + (hx % 20);
        green = 145 + (hy % 20);
        blue = 155 + (hz % 20);
    } else {
        red = 210 + (hx % 40);
        green = 180 + (hy % 35);
        blue = 140 + (hz % 30);
    }

    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = grainMat;
    flags[i] = 0;
    grainId[i] = i; // stable monotonic grain ID
}

// Column Collapse Scenario Kernel
__global__ void granularSpawnColumnKernel(
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
    uint32_t seed,
    uint16_t matId = 0)
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

    const float halfWidthX = 0.5f * static_cast<float>(sideX - 1) * spacing;
    const float halfWidthZ = 0.5f * static_cast<float>(sideZ - 1) * spacing;

    const float px = origin.x - halfWidthX + static_cast<float>(x) * spacing + jx;
    const float py = origin.y + static_cast<float>(y) * spacing + jy;
    const float pz = origin.z - halfWidthZ + static_cast<float>(z) * spacing + jz;
    const float r = baseRadius + jr;

    // Normalized volume-scaled inverse mass
    const float rRatio = baseRadius / (r > 0.0f ? r : baseRadius);
    const float invMass = (rRatio * rRatio * rRatio);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, invMass);

    const uint8_t red = 210 + (hx % 40);
    const uint8_t green = 180 + (hy % 35);
    const uint8_t blue = 140 + (hz % 30);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = matId;
    flags[i] = 0;
    grainId[i] = i;
}

// Pouring Sand Pile (Angle of Repose) Kernel with concentric conical heap placement
__global__ void granularSpawnPouringPileKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    float4* __restrict__ pred,
    uint32_t* __restrict__ tint,
    uint16_t* __restrict__ materialId,
    uint16_t* __restrict__ flags,
    uint32_t* __restrict__ grainId,
    uint32_t count,
    float3 origin,
    float nozzleRadius,
    float height,
    float baseRadius,
    float jitterRatio,
    uint32_t seed,
    uint16_t matId = 0)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float effRadius = baseRadius * (1.0f + jitterRatio);
    const float spacing = 2.05f * effRadius;
    const float layerH = 2.05f * effRadius;
    // Sized so that count grains form a complete cone of angle ~ 33.9 deg (dry quartz sand):
    const uint32_t K_base = max(4u, static_cast<uint32_t>(cbrtf(static_cast<float>(count)) * 1.1447f));
    const uint32_t L = max(2u, static_cast<uint32_t>(0.6667f * static_cast<float>(K_base)));

    uint32_t rem = i;
    uint32_t layer = 0;
    uint32_t Ky = 1;
    // Layer 0 is apex (Ky = 1), layer L-1 is base (Ky = K_base)
    while (layer < L) {
        Ky = max(1u, static_cast<uint32_t>(1.0f + (static_cast<float>(K_base - 1) * static_cast<float>(layer) / static_cast<float>(L - 1))));
        const uint32_t nGrainsInLayer = 1 + 3 * Ky * (Ky + 1);
        if (rem < nGrainsInLayer) {
            break;
        }
        rem -= nGrainsInLayer;
        layer++;
    }
    if (layer >= L) {
        layer = L - 1;
        Ky = K_base;
        rem = 0;
    }

    float r = 0.0f;
    float theta = 0.0f;

    if (rem > 0) {
        const uint32_t k = static_cast<uint32_t>((3.0f + sqrtf(12.0f * static_cast<float>(rem) - 3.0f)) / 6.0f);
        const uint32_t ringK = min(k, Ky);
        const uint32_t startK = 1 + 3 * (ringK - 1) * ringK;
        const uint32_t idx = (rem >= startK) ? (rem - startK) : 0;
        r = static_cast<float>(ringK) * spacing;
        theta = (ringK > 0) ? ((static_cast<float>(idx) / static_cast<float>(6 * ringK)) * 6.2831853f + static_cast<float>(layer) * 0.5235987f) : 0.0f;
    }

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    const float jx = 0.0f;
    const float jy = 0.0f;
    const float jz = 0.0f;
    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * 2.0f * jitterRatio * baseRadius;

    // Apex at top (layer 0 -> height = origin.y + (L-1)*layerH), base at bottom (layer L-1 -> height = origin.y)
    const float px = origin.x + r * cosf(theta) + jx;
    const float py = origin.y + static_cast<float>(L - 1 - layer) * layerH + jy;
    const float pz = origin.z + r * sinf(theta) + jz;
    const float grainR = baseRadius + jr;

    const uint16_t grainMat = (matId == 0xFFFF) ? static_cast<uint16_t>(i % 2) : matId;
    const float effDensity = (grainMat == 1) ? 2400.0f : 1600.0f;

    // Normalized volume-scaled inverse mass
    const float rRatio = baseRadius / (grainR > 0.0f ? grainR : baseRadius);
    const float invMass = (rRatio * rRatio * rRatio) * (1600.0f / effDensity);

    pos[i] = make_float4(px, py, pz, grainR);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, invMass);

    uint8_t red, green, blue;
    if (grainMat == 1) {
        red = 110 + (hx % 25);
        green = 80 + (hy % 20);
        blue = 60 + (hz % 15);
    } else {
        red = 210 + (hx % 40);
        green = 180 + (hy % 35);
        blue = 140 + (hz % 30);
    }
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = grainMat;
    flags[i] = 0;
    grainId[i] = i;
}

// Funnel Scenario Kernel
__global__ void granularSpawnFunnelKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    float4* __restrict__ pred,
    uint32_t* __restrict__ tint,
    uint16_t* __restrict__ materialId,
    uint16_t* __restrict__ flags,
    uint32_t* __restrict__ grainId,
    uint32_t count,
    float3 origin,
    float topRadius,
    float orificeRadius,
    float height,
    float baseRadius,
    float jitterRatio,
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    const float fracY = static_cast<float>(i) / static_cast<float>(count);
    const float localY = fracY * height;
    const float localRadius = orificeRadius + (topRadius - orificeRadius) * fracY;

    const float u = (hx & 0xFFFF) / 65535.0f;
    const float v = (hy & 0xFFFF) / 65535.0f;
    const float rad = localRadius * sqrtf(u);
    const float theta = v * 2.0f * 3.14159265f;

    const float px = origin.x + rad * cosf(theta);
    const float py = origin.y + localY;
    const float pz = origin.z + rad * sinf(theta);

    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * baseRadius;
    const float r = baseRadius + jr;

    const float vol = (4.0f / 3.0f) * 3.14159265f * (r * r * r);
    const float invMass = 1.0f / (vol * 1600.0f);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, -0.2f, 0.0f, invMass);

    const uint8_t red = 210 + (hx % 40);
    const uint8_t green = 180 + (hy % 35);
    const uint8_t blue = 140 + (hz % 30);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = 0;
    grainId[i] = i;
}

// Dune Scenario Kernel
__global__ void granularSpawnDuneKernel(
    float4* __restrict__ pos,
    float4* __restrict__ vel,
    float4* __restrict__ pred,
    uint32_t* __restrict__ tint,
    uint16_t* __restrict__ materialId,
    uint16_t* __restrict__ flags,
    uint32_t* __restrict__ grainId,
    uint32_t count,
    float3 origin,
    float lengthX,
    float widthZ,
    float slopeAngleDeg,
    float baseRadius,
    float jitterRatio,
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    const float fracX = ((hx & 0xFFFF) / 65535.0f);
    const float fracZ = ((hz & 0xFFFF) / 65535.0f);
    const float fracY = ((hy & 0xFFFF) / 65535.0f);

    const float slopeRad = slopeAngleDeg * 3.14159265f / 180.0f;
    const float px = origin.x + (fracX - 0.5f) * lengthX;
    const float pz = origin.z + (fracZ - 0.5f) * widthZ;
    const float baseY = origin.y + fracX * lengthX * sinf(slopeRad);
    const float py = baseY + fracY * 0.15f;

    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * baseRadius;
    const float r = baseRadius + jr;

    const float vol = (4.0f / 3.0f) * 3.14159265f * (r * r * r);
    const float invMass = 1.0f / (vol * 1600.0f);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, invMass);

    const uint8_t red = 210 + (hx % 40);
    const uint8_t green = 180 + (hy % 35);
    const uint8_t blue = 140 + (hz % 30);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = 0;
    grainId[i] = i;
}

// -----------------------------------------------------------------------------
// Phase 11F: Cliff Scenario Kernel (Vertical 90-degree cliff face)
// -----------------------------------------------------------------------------
__global__ void granularSpawnCliffKernel(
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
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t layerSize = sideX * sideZ;
    const uint32_t y = i / layerSize;
    const uint32_t rem = i % layerSize;
    const uint32_t z = rem / sideX;
    const uint32_t x = rem % sideX;

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    // Stagger alternate layers for hexagonal/close-packing stability
    const float staggerX = (y % 2 == 1) ? (spacing * 0.5f) : 0.0f;
    const float staggerZ = (y % 2 == 1) ? (spacing * 0.2887f) : 0.0f;

    const float jx = ((hx & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * spacing;
    const float jz = ((hz & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * spacing;

    const float px = origin.x + static_cast<float>(x) * spacing + staggerX + jx;
    const float py = origin.y + static_cast<float>(y) * spacing * 0.866f + baseRadius;
    const float pz = origin.z + (static_cast<float>(z) - static_cast<float>(sideZ) * 0.5f) * spacing + staggerZ + jz;

    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * baseRadius;
    const float r = baseRadius + jr;

    const float vol = (4.0f / 3.0f) * 3.14159265f * (r * r * r);
    const float invMass = 1.0f / (vol * 1600.0f);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, invMass);

    // Damp sand brown-ochre tinting
    const uint8_t red = 180 + (hx % 30);
    const uint8_t green = 135 + (hy % 25);
    const uint8_t blue = 90 + (hz % 20);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = 0;
    grainId[i] = i;
}

// -----------------------------------------------------------------------------
// Phase 11F: Bonded Block Kernel (Horizontal beam with anchored root)
// -----------------------------------------------------------------------------
__global__ void granularSpawnBondedBlockKernel(
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
    uint32_t sideY,
    uint32_t sideZ,
    float anchorRatio,
    float baseRadius,
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t layerSize = sideX * sideZ;
    const uint32_t y = i / layerSize;
    const uint32_t rem = i % layerSize;
    const uint32_t z = rem / sideX;
    const uint32_t x = rem % sideX;

    const float px = origin.x + static_cast<float>(x) * spacing;
    const float py = origin.y + static_cast<float>(y) * spacing;
    const float pz = origin.z + (static_cast<float>(z) - static_cast<float>(sideZ) * 0.5f) * spacing;

    const float vol = (4.0f / 3.0f) * 3.14159265f * (baseRadius * baseRadius * baseRadius);
    // Left end (x / sideX <= anchorRatio) is anchored (invMass = 0)
    const bool isAnchored = (static_cast<float>(x) / static_cast<float>(sideX) <= anchorRatio);
    const float invMass = isAnchored ? 0.0f : (1.0f / (vol * 1600.0f));

    pos[i] = make_float4(px, py, pz, baseRadius);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, 0.0f, 0.0f, invMass);

    uint8_t red = isAnchored ? 220 : 130;
    uint8_t green = isAnchored ? 50 : 160;
    uint8_t blue = isAnchored ? 50 : 210;
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = isAnchored ? 0x0004 : 0; // bit 2: anchored
    grainId[i] = i;
}

// -----------------------------------------------------------------------------
// Phase 11F: Clumps Scenario Kernel (Distinct cluster color hashing)
// -----------------------------------------------------------------------------
__global__ void granularSpawnClumpsKernel(
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
    float clumpRadius,
    float baseRadius,
    float jitterRatio,
    uint32_t seed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const uint32_t layerSize = sideX * sideZ;
    const uint32_t y = i / layerSize;
    const uint32_t rem = i % layerSize;
    const uint32_t z = rem / sideX;
    const uint32_t x = rem % sideX;

    const uint32_t hx = granularHash(i ^ seed);
    const uint32_t hy = granularHash(hx ^ 0x9e3779b9);
    const uint32_t hz = granularHash(hy ^ 0x85ebca6b);
    const uint32_t hr = granularHash(hz ^ 0xc2b2ae35);

    const float px = origin.x + (static_cast<float>(x) - static_cast<float>(sideX) * 0.5f) * spacing;
    const float py = origin.y + static_cast<float>(y) * spacing;
    const float pz = origin.z + (static_cast<float>(z) - static_cast<float>(sideZ) * 0.5f) * spacing;

    const float jr = ((hr & 0xFFFF) / 65535.0f - 0.5f) * jitterRatio * baseRadius;
    const float r = baseRadius + jr;

    const float vol = (4.0f / 3.0f) * 3.14159265f * (r * r * r);
    const float invMass = 1.0f / (vol * 1600.0f);

    pos[i] = make_float4(px, py, pz, r);
    pred[i] = pos[i];
    vel[i] = make_float4(0.0f, -0.5f, 0.0f, invMass);

    // Color code each clump cluster distinctly
    const float clumpDiameter = 2.0f * clumpRadius;
    const int cx = static_cast<int>(floorf(px / clumpDiameter));
    const int cy = static_cast<int>(floorf(py / clumpDiameter));
    const int cz = static_cast<int>(floorf(pz / clumpDiameter));
    const uint32_t clumpHash = granularHash(seed ^ (cx * 73856093u ^ cy * 19349663u ^ cz * 83492791u));

    const uint8_t red = 80 + (clumpHash & 0x7F);
    const uint8_t green = 80 + ((clumpHash >> 7) & 0x7F);
    const uint8_t blue = 80 + ((clumpHash >> 14) & 0x7F);
    tint[i] = (255u << 24) | (static_cast<uint32_t>(blue) << 16) | (static_cast<uint32_t>(green) << 8) | static_cast<uint32_t>(red);

    materialId[i] = 0;
    flags[i] = 0;
    grainId[i] = i;
}

} // namespace vparticles
