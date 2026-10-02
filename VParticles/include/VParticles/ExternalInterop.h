#pragma once

#include "VParticles/ParticlePool.h"
#include "VParticles/SimulationTypes.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace vparticles::interop {

// 32-byte vertex structure matching standard game engine vertex layouts (e.g., float4 pos+size, float4 color)
struct alignas(16) InteropVertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float size = 1.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
};

// Layout-compatible with D3D12_DRAW_ARGUMENTS
struct alignas(4) D3D12DrawArguments {
    uint32_t vertexCountPerInstance = 0;
    uint32_t instanceCount = 1;
    uint32_t startVertexLocation = 0;
    uint32_t startInstanceLocation = 0;
};

// Layout-compatible with VkDrawIndirectCommand
struct alignas(4) VkDrawIndirectCommand {
    uint32_t vertexCount = 0;
    uint32_t instanceCount = 1;
    uint32_t firstVertex = 0;
    uint32_t firstInstance = 0;
};

enum class ExternalMemoryType {
    D3D12Resource,
    D3D12Heap,
    OpaqueWin32,     // Vulkan on Windows (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
    OpaqueFd         // Vulkan on Linux (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
};

enum class ExternalSemaphoreType {
    D3D12Fence,
    TimelineSemaphoreWin32, // Vulkan timeline semaphore on Windows
    TimelineSemaphoreFd     // Vulkan timeline semaphore on Linux
};

struct ExternalMemoryDesc {
    ExternalMemoryType type = ExternalMemoryType::D3D12Resource;
    uint64_t sizeBytes = 0;
#if defined(_WIN32)
    HANDLE handle = nullptr;
#endif
    int fd = -1;
    bool isDedicated = true;
};

struct ExternalSemaphoreDesc {
    ExternalSemaphoreType type = ExternalSemaphoreType::D3D12Fence;
#if defined(_WIN32)
    HANDLE handle = nullptr;
#endif
    int fd = -1;
};

// RAII wrapper around imported CUDA external memory
class ExternalMemoryBuffer {
public:
    ExternalMemoryBuffer();
    ~ExternalMemoryBuffer();

    ExternalMemoryBuffer(const ExternalMemoryBuffer&) = delete;
    ExternalMemoryBuffer& operator=(const ExternalMemoryBuffer&) = delete;

    ExternalMemoryBuffer(ExternalMemoryBuffer&& other) noexcept;
    ExternalMemoryBuffer& operator=(ExternalMemoryBuffer&& other) noexcept;

    bool import(const ExternalMemoryDesc& desc);
    void release();

    bool isValid() const { return devPtr_ != nullptr; }
    void* devicePointer() const { return devPtr_; }
    uint64_t sizeBytes() const { return sizeBytes_; }

private:
    cudaExternalMemory_t extMem_ = nullptr;
    void* devPtr_ = nullptr;
    uint64_t sizeBytes_ = 0;
};

// RAII wrapper around imported CUDA external timeline semaphore (D3D12 Fence or Vulkan Timeline Semaphore)
class ExternalTimelineSemaphore {
public:
    ExternalTimelineSemaphore();
    ~ExternalTimelineSemaphore();

    ExternalTimelineSemaphore(const ExternalTimelineSemaphore&) = delete;
    ExternalTimelineSemaphore& operator=(const ExternalTimelineSemaphore&) = delete;

    ExternalTimelineSemaphore(ExternalTimelineSemaphore&& other) noexcept;
    ExternalTimelineSemaphore& operator=(ExternalTimelineSemaphore&& other) noexcept;

    bool import(const ExternalSemaphoreDesc& desc);
    void release();

    bool isValid() const { return extSem_ != nullptr; }

    // Enqueue an asynchronous wait on the external semaphore on the specified CUDA stream
    bool waitAsync(uint64_t value, cudaStream_t stream = nullptr);

    // Enqueue an asynchronous signal on the external semaphore on the specified CUDA stream
    bool signalAsync(uint64_t value, cudaStream_t stream = nullptr);

private:
    cudaExternalSemaphore_t extSem_ = nullptr;
};

// High-performance bridge that gathers active particles directly into an external engine vertex buffer
// and populates the indirect draw command without CPU synchronization.
class ModernInteropBridge {
public:
    ModernInteropBridge();
    ~ModernInteropBridge();

    ModernInteropBridge(const ModernInteropBridge&) = delete;
    ModernInteropBridge& operator=(const ModernInteropBridge&) = delete;

    // Initialize the bridge with mapped external memory pointers
    // vertexBufferPtr: destination buffer for InteropVertex array
    // indirectCmdPtr: destination buffer for D3D12DrawArguments / VkDrawIndirectCommand
    // maxCapacity: maximum particles that can fit in vertexBufferPtr
    bool init(void* vertexBufferPtr, void* indirectCmdPtr, uint32_t maxCapacity);
    void shutdown();

    // Gathers active particles and emits indirect draw count into external engine memory
    bool gather(const GpuParticlePool& pool, cudaStream_t stream = nullptr);

    // Alternative gather for packed storage pool
    bool gatherPacked(const PackedPool& pool, const uint32_t* const* activeIndices, const uint32_t* aliveCount, float posScale, cudaStream_t stream = nullptr);

    uint32_t capacity() const { return capacity_; }
    bool isInitialized() const { return vertexBuffer_ != nullptr; }

private:
    void* vertexBuffer_ = nullptr;
    void* indirectCmd_ = nullptr;
    uint32_t capacity_ = 0;
};

} // namespace vparticles::interop
