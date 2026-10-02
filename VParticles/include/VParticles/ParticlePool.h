#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace vparticles {

struct ParticlePool {
    float4* pos = nullptr;
    float4* vel = nullptr;
    float4* color = nullptr;
    float* age = nullptr;
    float* lifetime = nullptr;
    uint32_t* systemId = nullptr;
    uint16_t* tileId = nullptr;
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
    const uint16_t* tileId = nullptr;
    uint32_t* const* activeIndices = nullptr;
    const uint32_t* aliveCount = nullptr;
    uint32_t capacity = 0;
};

// Packed SoA storage: 28 bytes/particle in pool vs 62 bytes for FP32 pool.
// Each field gets its own array for coalesced GPU access.
// Positions are tile-local int16 quantized; velocities, lifetime, and size
// are IEEE fp16; age is normalized uint16; color is packed RGBA8 in uint32.
struct PackedPool {
    int16_t*  posX = nullptr;        // 2 bytes x N
    int16_t*  posY = nullptr;        // 2 bytes x N
    int16_t*  posZ = nullptr;        // 2 bytes x N
    __half*   velX = nullptr;        // 2 bytes x N
    __half*   velY = nullptr;        // 2 bytes x N
    __half*   velZ = nullptr;        // 2 bytes x N
    uint16_t* age  = nullptr;        // 2 bytes x N (normalized: age/lifetime * 65535)
    __half*   lifetime = nullptr;    // 2 bytes x N
    uint32_t* colorPacked = nullptr; // 4 bytes x N (RGBA8)
    __half*   size = nullptr;        // 2 bytes x N
    uint16_t* systemId = nullptr;    // 2 bytes x N
    uint16_t* tileId = nullptr;      // 2 bytes x N
    uint32_t  capacity = 0;
};

} // namespace vparticles
