#include "CudaGLBridge.h"

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>

#define VP_CUDA_GL_CHECK(call)                                                 \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " -> " << cudaGetErrorString(err__) << std::endl;     \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace vparticles::viewer {

namespace {

__global__ void gatherRenderKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ color,
    uint32_t* const* __restrict__ activeIndicesPtr,
    const uint32_t* __restrict__ aliveCountPtr,
    uint32_t maxCapacity,
    RenderVertex* __restrict__ outVertices,
    DrawArraysIndirectCommand* __restrict__ outCmd)
{
    // Single thread in grid updates the indirect draw command
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t alive = *aliveCountPtr;
        if (alive > maxCapacity) {
            alive = maxCapacity;
        }
        outCmd->count = alive;
        outCmd->instanceCount = 1;
        outCmd->first = 0;
        outCmd->baseInstance = 0;
    }

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t alive = *aliveCountPtr;
    if (idx >= alive || idx >= maxCapacity) {
        return;
    }

    const uint32_t* activeIndices = *activeIndicesPtr;
    uint32_t slot = activeIndices[idx];

    float4 p = pos[slot];
    float4 c = color[slot];

    // p.x, p.y, p.z = position, p.w = particle size
    outVertices[idx] = RenderVertex{
        p.x, p.y, p.z, p.w,
        c.x, c.y, c.z, c.w
    };
}

} // anonymous namespace

struct CudaGLBridge::Impl {
    cudaGraphicsResource_t vboResource = nullptr;
    cudaGraphicsResource_t indirectResource = nullptr;
    bool isRegistered = false;
};

CudaGLBridge::CudaGLBridge()
    : impl_(new Impl())
{}

CudaGLBridge::~CudaGLBridge() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

bool CudaGLBridge::init(uint32_t vboId, uint32_t indirectBufferId, uint32_t capacity) {
    shutdown();
    capacity_ = capacity;

    VP_CUDA_GL_CHECK(cudaGraphicsGLRegisterBuffer(
        &impl_->vboResource,
        vboId,
        cudaGraphicsRegisterFlagsWriteDiscard));

    VP_CUDA_GL_CHECK(cudaGraphicsGLRegisterBuffer(
        &impl_->indirectResource,
        indirectBufferId,
        cudaGraphicsRegisterFlagsWriteDiscard));

    impl_->isRegistered = true;
    return true;
}

void CudaGLBridge::shutdown() {
    if (!impl_ || !impl_->isRegistered) return;

    if (impl_->vboResource) {
        cudaGraphicsUnregisterResource(impl_->vboResource);
        impl_->vboResource = nullptr;
    }
    if (impl_->indirectResource) {
        cudaGraphicsUnregisterResource(impl_->indirectResource);
        impl_->indirectResource = nullptr;
    }
    impl_->isRegistered = false;
    capacity_ = 0;
}

bool CudaGLBridge::gather(const GpuParticlePool& pool, cudaStream_t stream) {
    if (!impl_ || !impl_->isRegistered) return false;
    if (!pool.pos || !pool.color || !pool.activeIndices || !pool.aliveCount) return false;

    cudaGraphicsResource_t resources[2] = {
        impl_->vboResource,
        impl_->indirectResource
    };

    VP_CUDA_GL_CHECK(cudaGraphicsMapResources(2, resources, stream));

    RenderVertex* d_outVertices = nullptr;
    size_t vboBytes = 0;
    VP_CUDA_GL_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&d_outVertices),
        &vboBytes,
        impl_->vboResource));

    DrawArraysIndirectCommand* d_outCmd = nullptr;
    size_t cmdBytes = 0;
    VP_CUDA_GL_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&d_outCmd),
        &cmdBytes,
        impl_->indirectResource));

    constexpr uint32_t kBlockSize = 256;
    const uint32_t numBlocks = (capacity_ + kBlockSize - 1) / kBlockSize;

    gatherRenderKernel<<<numBlocks, kBlockSize, 0, stream>>>(
        pool.pos,
        pool.color,
        pool.activeIndices,
        pool.aliveCount,
        capacity_,
        d_outVertices,
        d_outCmd);

    VP_CUDA_GL_CHECK(cudaGetLastError());
    VP_CUDA_GL_CHECK(cudaGraphicsUnmapResources(2, resources, stream));

    return true;
}

} // namespace vparticles::viewer
