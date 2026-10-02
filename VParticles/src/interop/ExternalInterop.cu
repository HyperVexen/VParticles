#include "VParticles/ExternalInterop.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <utility>

namespace vparticles::interop {

// ============================================================================
// GPU Gather Kernels
// ============================================================================

__global__ void gatherModernInteropKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ color,
    uint32_t* const* __restrict__ activeIndicesPtr,
    const uint32_t* __restrict__ aliveCountPtr,
    uint32_t maxCapacity,
    InteropVertex* __restrict__ outVertices,
    D3D12DrawArguments* __restrict__ outCmd)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t alive = *aliveCountPtr;
        if (alive > maxCapacity) alive = maxCapacity;
        if (outCmd) {
            outCmd->vertexCountPerInstance = alive;
            outCmd->instanceCount = 1;
            outCmd->startVertexLocation = 0;
            outCmd->startInstanceLocation = 0;
        }
    }

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t alive = *aliveCountPtr;
    if (idx >= alive || idx >= maxCapacity) return;

    const uint32_t* activeIndices = *activeIndicesPtr;
    uint32_t slot = activeIndices[idx];

    float4 p = pos[slot];
    float4 c = color[slot];
    outVertices[idx] = InteropVertex{ p.x, p.y, p.z, p.w, c.x, c.y, c.z, c.w };
}

__global__ void gatherModernInteropPackedKernel(
    const int16_t* __restrict__ posX,
    const int16_t* __restrict__ posY,
    const int16_t* __restrict__ posZ,
    const __half* __restrict__ size,
    const uint32_t* __restrict__ colorPacked,
    const uint32_t* const* __restrict__ activeIndicesPtr,
    const uint32_t* __restrict__ aliveCountPtr,
    float posScale,
    uint32_t maxCapacity,
    InteropVertex* __restrict__ outVertices,
    D3D12DrawArguments* __restrict__ outCmd)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t alive = *aliveCountPtr;
        if (alive > maxCapacity) alive = maxCapacity;
        if (outCmd) {
            outCmd->vertexCountPerInstance = alive;
            outCmd->instanceCount = 1;
            outCmd->startVertexLocation = 0;
            outCmd->startInstanceLocation = 0;
        }
    }

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t alive = *aliveCountPtr;
    if (idx >= alive || idx >= maxCapacity) return;

    const uint32_t* activeIndices = *activeIndicesPtr;
    uint32_t slot = activeIndices[idx];

    float px = (float)posX[slot] * posScale;
    float py = (float)posY[slot] * posScale;
    float pz = (float)posZ[slot] * posScale;
    float psize = __half2float(size[slot]);

    uint32_t cPacked = colorPacked[slot];
    float cr = (float)(cPacked & 0xFF) / 255.0f;
    float cg = (float)((cPacked >> 8) & 0xFF) / 255.0f;
    float cb = (float)((cPacked >> 16) & 0xFF) / 255.0f;
    float ca = (float)((cPacked >> 24) & 0xFF) / 255.0f;

    outVertices[idx] = InteropVertex{ px, py, pz, psize, cr, cg, cb, ca };
}

// ============================================================================
// ExternalMemoryBuffer Implementation
// ============================================================================

ExternalMemoryBuffer::ExternalMemoryBuffer() = default;

ExternalMemoryBuffer::~ExternalMemoryBuffer() {
    release();
}

ExternalMemoryBuffer::ExternalMemoryBuffer(ExternalMemoryBuffer&& other) noexcept
    : extMem_(std::exchange(other.extMem_, nullptr))
    , devPtr_(std::exchange(other.devPtr_, nullptr))
    , sizeBytes_(std::exchange(other.sizeBytes_, 0)) {}

ExternalMemoryBuffer& ExternalMemoryBuffer::operator=(ExternalMemoryBuffer&& other) noexcept {
    if (this != &other) {
        release();
        extMem_ = std::exchange(other.extMem_, nullptr);
        devPtr_ = std::exchange(other.devPtr_, nullptr);
        sizeBytes_ = std::exchange(other.sizeBytes_, 0);
    }
    return *this;
}

bool ExternalMemoryBuffer::import(const ExternalMemoryDesc& desc) {
    release();

    cudaExternalMemoryHandleDesc memDesc = {};
    switch (desc.type) {
    case ExternalMemoryType::D3D12Resource:
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
#if defined(_WIN32)
        memDesc.handle.win32.handle = desc.handle;
#endif
        break;
    case ExternalMemoryType::D3D12Heap:
        memDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
#if defined(_WIN32)
        memDesc.handle.win32.handle = desc.handle;
#endif
        break;
    case ExternalMemoryType::OpaqueWin32:
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
#if defined(_WIN32)
        memDesc.handle.win32.handle = desc.handle;
#endif
        break;
    case ExternalMemoryType::OpaqueFd:
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memDesc.handle.fd = desc.fd;
        break;
    }

    memDesc.size = desc.sizeBytes;
    memDesc.flags = desc.isDedicated ? cudaExternalMemoryDedicated : 0;

    cudaError_t err = cudaImportExternalMemory(&extMem_, &memDesc);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ExternalMemoryBuffer] cudaImportExternalMemory failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaExternalMemoryBufferDesc bufDesc = {};
    bufDesc.offset = 0;
    bufDesc.size = desc.sizeBytes;
    bufDesc.flags = 0;

    err = cudaExternalMemoryGetMappedBuffer(&devPtr_, extMem_, &bufDesc);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ExternalMemoryBuffer] cudaExternalMemoryGetMappedBuffer failed: %s\n", cudaGetErrorString(err));
        cudaDestroyExternalMemory(extMem_);
        extMem_ = nullptr;
        devPtr_ = nullptr;
        return false;
    }

    sizeBytes_ = desc.sizeBytes;
    return true;
}

void ExternalMemoryBuffer::release() {
    if (devPtr_) {
        cudaFree(devPtr_);
        devPtr_ = nullptr;
    }
    if (extMem_) {
        cudaDestroyExternalMemory(extMem_);
        extMem_ = nullptr;
    }
    sizeBytes_ = 0;
}

// ============================================================================
// ExternalTimelineSemaphore Implementation
// ============================================================================

ExternalTimelineSemaphore::ExternalTimelineSemaphore() = default;

ExternalTimelineSemaphore::~ExternalTimelineSemaphore() {
    release();
}

ExternalTimelineSemaphore::ExternalTimelineSemaphore(ExternalTimelineSemaphore&& other) noexcept
    : extSem_(std::exchange(other.extSem_, nullptr)) {}

ExternalTimelineSemaphore& ExternalTimelineSemaphore::operator=(ExternalTimelineSemaphore&& other) noexcept {
    if (this != &other) {
        release();
        extSem_ = std::exchange(other.extSem_, nullptr);
    }
    return *this;
}

bool ExternalTimelineSemaphore::import(const ExternalSemaphoreDesc& desc) {
    release();

    cudaExternalSemaphoreHandleDesc semDesc = {};
    switch (desc.type) {
    case ExternalSemaphoreType::D3D12Fence:
        semDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
#if defined(_WIN32)
        semDesc.handle.win32.handle = desc.handle;
#endif
        break;
    case ExternalSemaphoreType::TimelineSemaphoreWin32:
        semDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
#if defined(_WIN32)
        semDesc.handle.win32.handle = desc.handle;
#endif
        break;
    case ExternalSemaphoreType::TimelineSemaphoreFd:
        semDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        semDesc.handle.fd = desc.fd;
        break;
    }

    semDesc.flags = 0;

    cudaError_t err = cudaImportExternalSemaphore(&extSem_, &semDesc);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ExternalTimelineSemaphore] cudaImportExternalSemaphore failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

void ExternalTimelineSemaphore::release() {
    if (extSem_) {
        cudaDestroyExternalSemaphore(extSem_);
        extSem_ = nullptr;
    }
}

bool ExternalTimelineSemaphore::waitAsync(uint64_t value, cudaStream_t stream) {
    if (!extSem_) return false;

    cudaExternalSemaphoreWaitParams waitParams = {};
    waitParams.flags = 0;
    waitParams.params.fence.value = value;

    cudaError_t err = cudaWaitExternalSemaphoresAsync(&extSem_, &waitParams, 1, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ExternalTimelineSemaphore] cudaWaitExternalSemaphoresAsync failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool ExternalTimelineSemaphore::signalAsync(uint64_t value, cudaStream_t stream) {
    if (!extSem_) return false;

    cudaExternalSemaphoreSignalParams sigParams = {};
    sigParams.flags = 0;
    sigParams.params.fence.value = value;

    cudaError_t err = cudaSignalExternalSemaphoresAsync(&extSem_, &sigParams, 1, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ExternalTimelineSemaphore] cudaSignalExternalSemaphoresAsync failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// ModernInteropBridge Implementation
// ============================================================================

ModernInteropBridge::ModernInteropBridge() = default;

ModernInteropBridge::~ModernInteropBridge() {
    shutdown();
}

bool ModernInteropBridge::init(void* vertexBufferPtr, void* indirectCmdPtr, uint32_t maxCapacity) {
    if (!vertexBufferPtr || maxCapacity == 0) return false;
    vertexBuffer_ = vertexBufferPtr;
    indirectCmd_ = indirectCmdPtr;
    capacity_ = maxCapacity;
    return true;
}

void ModernInteropBridge::shutdown() {
    vertexBuffer_ = nullptr;
    indirectCmd_ = nullptr;
    capacity_ = 0;
}

bool ModernInteropBridge::gather(const GpuParticlePool& pool, cudaStream_t stream) {
    if (!vertexBuffer_ || capacity_ == 0) return false;

    constexpr uint32_t kBlockSize = 256;
    uint32_t blocks = (capacity_ + kBlockSize - 1) / kBlockSize;
    if (blocks == 0) blocks = 1;

    gatherModernInteropKernel<<<blocks, kBlockSize, 0, stream>>>(
        pool.pos,
        pool.color,
        pool.activeIndices,
        pool.aliveCount,
        capacity_,
        static_cast<InteropVertex*>(vertexBuffer_),
        static_cast<D3D12DrawArguments*>(indirectCmd_));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ModernInteropBridge] gatherModernInteropKernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool ModernInteropBridge::gatherPacked(
    const PackedPool& pool,
    const uint32_t* const* activeIndices,
    const uint32_t* aliveCount,
    float posScale,
    cudaStream_t stream)
{
    if (!vertexBuffer_ || capacity_ == 0) return false;

    constexpr uint32_t kBlockSize = 256;
    uint32_t blocks = (capacity_ + kBlockSize - 1) / kBlockSize;
    if (blocks == 0) blocks = 1;

    gatherModernInteropPackedKernel<<<blocks, kBlockSize, 0, stream>>>(
        pool.posX,
        pool.posY,
        pool.posZ,
        pool.size,
        pool.colorPacked,
        activeIndices,
        aliveCount,
        posScale,
        capacity_,
        static_cast<InteropVertex*>(vertexBuffer_),
        static_cast<D3D12DrawArguments*>(indirectCmd_));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[ModernInteropBridge] gatherModernInteropPackedKernel failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace vparticles::interop
