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
    uint32_t aliveCount = 0;
};

} // namespace vparticles
