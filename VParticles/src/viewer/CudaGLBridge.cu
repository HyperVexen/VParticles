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

__device__ inline float3 turboColormap(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    // Smooth blue -> cyan -> green -> yellow -> red heatmap
    float r = fminf(fmaxf(1.5f - fabsf(t * 4.0f - 3.0f), 0.0f), 1.0f);
    float g = fminf(fmaxf(1.5f - fabsf(t * 4.0f - 2.0f), 0.0f), 1.0f);
    float b = fminf(fmaxf(1.5f - fabsf(t * 4.0f - 1.0f), 0.0f), 1.0f);
    return make_float3(r, g, b);
}

__global__ void gatherGranularRenderKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    const uint32_t* __restrict__ tint,
    const uint16_t* __restrict__ flags,
    uint32_t aliveCount,
    uint32_t maxCapacity,
    int colorMode,
    float maxSpeed,
    float basePointSize,
    RenderVertex* __restrict__ outVertices,
    DrawArraysIndirectCommand* __restrict__ outCmd)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t alive = aliveCount;
        if (alive > maxCapacity) {
            alive = maxCapacity;
        }
        outCmd->count = alive;
        outCmd->instanceCount = 1;
        outCmd->first = 0;
        outCmd->baseInstance = 0;
    }

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= aliveCount || idx >= maxCapacity) {
        return;
    }

    float4 p = pos[idx];
    float r = p.w > 0.0f ? p.w : 0.01f;
    float renderSize = r * 80.0f * basePointSize;

    float4 col = make_float4(0.85f, 0.72f, 0.50f, 1.0f); // Default quartz sand

    if (colorMode == 0 && tint) {
        // Material / Clump packed RGBA8
        uint32_t t = tint[idx];
        col.x = static_cast<float>(t & 0xFF) / 255.0f;
        col.y = static_cast<float>((t >> 8) & 0xFF) / 255.0f;
        col.z = static_cast<float>((t >> 16) & 0xFF) / 255.0f;
        col.w = 1.0f;
    } else if (colorMode == 1 && vel) {
        // Velocity heatmap
        float4 v = vel[idx];
        float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        float t = speed / fmaxf(maxSpeed, 0.1f);
        float3 rgb = turboColormap(t);
        col = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
    } else if (colorMode == 2 && flags) {
        // Sleeping state: cool slate blue if asleep, radiant amber/gold if awake
        if (flags[idx] & 0x0001) { // kGrainFlagAsleep
            col = make_float4(0.35f, 0.50f, 0.70f, 0.85f);
        } else {
            col = make_float4(1.0f, 0.65f, 0.15f, 1.0f);
        }
    }

    outVertices[idx] = RenderVertex{
        p.x, p.y, p.z, renderSize,
        col.x, col.y, col.z, col.w
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

bool CudaGLBridge::gatherGranular(
    const GranularPool& pool,
    int colorMode,
    float maxSpeed,
    float basePointSize,
    cudaStream_t stream)
{
    if (!impl_ || !impl_->isRegistered) return false;
    if (!pool.pos || pool.aliveCount == 0) return false;

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
    const uint32_t numBlocks = (pool.aliveCount + kBlockSize - 1) / kBlockSize;

    gatherGranularRenderKernel<<<numBlocks, kBlockSize, 0, stream>>>(
        pool.pos,
        pool.vel,
        pool.tint,
        pool.flags,
        pool.aliveCount,
        capacity_,
        colorMode,
        maxSpeed,
        basePointSize,
        d_outVertices,
        d_outCmd);

    VP_CUDA_GL_CHECK(cudaGetLastError());
    VP_CUDA_GL_CHECK(cudaGraphicsUnmapResources(2, resources, stream));

    return true;
}

} // namespace vparticles::viewer
