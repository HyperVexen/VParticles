#pragma once

#include "VParticles/ParticlePool.h"
#include <cstdint>

namespace vparticles::viewer {

struct RenderVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float size = 1.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
}; // 32 bytes aligned (layout matches GL_FLOAT x 4 for pos+size and GL_FLOAT x 4 for color)

struct DrawArraysIndirectCommand {
    uint32_t count = 0;
    uint32_t instanceCount = 1;
    uint32_t first = 0;
    uint32_t baseInstance = 0;
};

class CudaGLBridge {
public:
    CudaGLBridge();
    ~CudaGLBridge();

    CudaGLBridge(const CudaGLBridge&) = delete;
    CudaGLBridge& operator=(const CudaGLBridge&) = delete;

    bool init(uint32_t vboId, uint32_t indirectBufferId, uint32_t capacity);
    void shutdown();

    // Maps OpenGL buffers, runs gather kernel to flatten live particles and update indirect command, and unmaps
    bool gather(const GpuParticlePool& pool, cudaStream_t stream = nullptr);

    uint32_t capacity() const { return capacity_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    uint32_t capacity_ = 0;
};

} // namespace vparticles::viewer
