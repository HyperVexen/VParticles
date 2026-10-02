#pragma once

#include "InternalTypes.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace vparticles {

// CUB's block scan lets each tile reserve one contiguous output range. This
// avoids a CPU readback and avoids one global atomic per surviving particle.
__global__ void compactActiveIndicesKernel(
    DeviceFrameState* state,
    const uint8_t* aliveFlags)
{
    if (state->deadCount == 0) {
        return;
    }

    using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ uint32_t outputStart;

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    uint32_t* compactedIndices = state->scratchActiveIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t tileStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x;
         tileStart < count;
         tileStart += stride) {
        const uint64_t activeIndex = tileStart + threadIdx.x;
        const bool inRange = activeIndex < count;
        const uint32_t isAlive = inRange ? static_cast<uint32_t>(aliveFlags[activeIndex]) : 0u;
        uint32_t localOffset = 0;
        uint32_t selectedInTile = 0;
        BlockScan(scanStorage).ExclusiveSum(isAlive, localOffset, selectedInTile);

        if (threadIdx.x == 0) {
            outputStart = selectedInTile == 0
                ? 0u
                : atomicAdd(&state->compactedCount, selectedInTile);
        }
        __syncthreads();

        if (isAlive != 0) {
            compactedIndices[outputStart + localOffset] = activeIndices[activeIndex];
        }
        __syncthreads();
    }
}

__global__ void finalizeCompactionKernel(DeviceFrameState* state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && state->deadCount != 0) {
        uint32_t* oldActiveIndices = state->activeIndices;
        state->activeIndices = state->scratchActiveIndices;
        state->scratchActiveIndices = oldActiveIndices;
        state->activeCount = state->compactedCount;
        state->activeBufferIndex ^= 1u;
    }
}

} // namespace vparticles
