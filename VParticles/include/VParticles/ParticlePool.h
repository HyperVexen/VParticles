#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vparticles {

struct ParticlePool {
    float4* pos = nullptr;
    float4* vel = nullptr;
    float4* color = nullptr;
    float* age = nullptr;
    float* lifetime = nullptr;
    uint32_t* systemId = nullptr;
    uint32_t* activeIndices = nullptr;
    uint32_t capacity = 0;
    // This is a completed telemetry snapshot, not the in-flight GPU counter.
    // Use ParticleSystem::gpuBuffers() from CUDA work that must consume the
    // current active list without a CPU synchronization point.
    uint32_t aliveCount = 0;
};

// Device-facing view of the shared pool. The active-list pointer and count
// remain GPU-resident because compaction may swap the active-index buffers.
// CUDA consumers should dereference activeIndices and aliveCount on the GPU.
struct GpuParticlePool {
    float4* pos = nullptr;
    float4* vel = nullptr;
    float4* color = nullptr;
    float* age = nullptr;
    float* lifetime = nullptr;
    uint32_t* systemId = nullptr;
    uint32_t* const* activeIndices = nullptr;
    const uint32_t* aliveCount = nullptr;
    uint32_t capacity = 0;
};

} // namespace vparticles
