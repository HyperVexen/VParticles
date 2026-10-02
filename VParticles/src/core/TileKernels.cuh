#pragma once

#include "InternalTypes.h"
#include "ParticleMath.cuh"

#include <cuda_runtime.h>

namespace vparticles {

__global__ void beginFrameFromParamsKernel(
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    const GraphParams* params)
{
    if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
            state->compactedCount = 0;
            state->deadCount = 0;
            state->spawnStart = 0;
            state->spawnCount = 0;
            state->freeStart = 0;
            state->requestedSpawn = params->requestedSpawn;
            state->frameIndex = params->frameIndex;
        }
        if (threadIdx.x < kMaxTiles) {
            telemetry->tileAliveCounts[threadIdx.x] = 0;
        }
    }
}

__global__ void migrateTilesPackedKernel(
    PackedPool pool,
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount,
    float posScale,
    float invPosScale)
{
    __shared__ uint32_t sTileCounts[kMaxTiles];
    if (threadIdx.x < kMaxTiles) {
        sTileCounts[threadIdx.x] = 0;
    }
    __syncthreads();

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];
        const uint16_t currentTile = pool.tileId[particleIndex];

        const float currentOriginX = cTiles[currentTile].originX;
        const float currentOriginY = cTiles[currentTile].originY;
        const float currentOriginZ = cTiles[currentTile].originZ;

        const float worldX = decodePosition(pool.posX[particleIndex], currentOriginX, invPosScale);
        const float worldY = decodePosition(pool.posY[particleIndex], currentOriginY, invPosScale);
        const float worldZ = decodePosition(pool.posZ[particleIndex], currentOriginZ, invPosScale);

        uint16_t bestTile = currentTile;
        float bestDistSq = (worldX - currentOriginX) * (worldX - currentOriginX) +
                           (worldY - currentOriginY) * (worldY - currentOriginY) +
                           (worldZ - currentOriginZ) * (worldZ - currentOriginZ);

        for (uint32_t t = 0; t < tileCount; ++t) {
            if (t == currentTile) continue;
            const float dx = worldX - cTiles[t].originX;
            const float dy = worldY - cTiles[t].originY;
            const float dz = worldZ - cTiles[t].originZ;
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestTile = static_cast<uint16_t>(t);
            }
        }

        if (bestTile != currentTile) {
            pool.tileId[particleIndex] = bestTile;
            pool.posX[particleIndex] = encodePosition(worldX, cTiles[bestTile].originX, posScale);
            pool.posY[particleIndex] = encodePosition(worldY, cTiles[bestTile].originY, posScale);
            pool.posZ[particleIndex] = encodePosition(worldZ, cTiles[bestTile].originZ, posScale);
        }

        atomicAdd(&sTileCounts[bestTile], 1u);
    }
    __syncthreads();

    if (threadIdx.x < tileCount) {
        const uint32_t countInTile = sTileCounts[threadIdx.x];
        if (countInTile > 0) {
            atomicAdd(&telemetry->tileAliveCounts[threadIdx.x], countInTile);
        }
    }
}

__global__ void migrateTilesKernel(
    ParticlePool pool,
    DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount)
{
    __shared__ uint32_t sTileCounts[kMaxTiles];
    if (threadIdx.x < kMaxTiles) {
        sTileCounts[threadIdx.x] = 0;
    }
    __syncthreads();

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];
        const uint16_t currentTile = pool.tileId[particleIndex];

        const float worldX = pool.pos[particleIndex].x;
        const float worldY = pool.pos[particleIndex].y;
        const float worldZ = pool.pos[particleIndex].z;

        const float currentOriginX = cTiles[currentTile].originX;
        const float currentOriginY = cTiles[currentTile].originY;
        const float currentOriginZ = cTiles[currentTile].originZ;

        uint16_t bestTile = currentTile;
        float bestDistSq = (worldX - currentOriginX) * (worldX - currentOriginX) +
                           (worldY - currentOriginY) * (worldY - currentOriginY) +
                           (worldZ - currentOriginZ) * (worldZ - currentOriginZ);

        for (uint32_t t = 0; t < tileCount; ++t) {
            if (t == currentTile) continue;
            const float dx = worldX - cTiles[t].originX;
            const float dy = worldY - cTiles[t].originY;
            const float dz = worldZ - cTiles[t].originZ;
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestTile = static_cast<uint16_t>(t);
            }
        }

        if (bestTile != currentTile) {
            pool.tileId[particleIndex] = bestTile;
        }

        atomicAdd(&sTileCounts[bestTile], 1u);
    }
    __syncthreads();

    if (threadIdx.x < tileCount) {
        const uint32_t countInTile = sTileCounts[threadIdx.x];
        if (countInTile > 0) {
            atomicAdd(&telemetry->tileAliveCounts[threadIdx.x], countInTile);
        }
    }
}

__global__ void snapshotTelemetryKernel(
    const DeviceFrameState* state,
    DeviceTelemetry* telemetry,
    uint32_t tileCount)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        telemetry->aliveCount = state->activeCount;
        telemetry->requestedSpawn = state->requestedSpawn;
        telemetry->spawned = state->spawnCount;
        telemetry->dropped = state->requestedSpawn - state->spawnCount;
        telemetry->deadCount = state->deadCount;
        telemetry->frameIndex = state->frameIndex;
        telemetry->activeBufferIndex = state->activeBufferIndex;
        if (tileCount <= 1) {
            telemetry->tileAliveCounts[0] = state->activeCount;
        }
    }
}

} // namespace vparticles
