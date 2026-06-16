// On Windows, windows.h must come before any GL headers so WINGDIAPI is defined.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "InstanceData.h"
#include "Particle.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: reads the particle array, writes InstanceData[] directly into
// the OpenGL VBO that is mapped into CUDA address space.
// No CPU involvement. No PCIe transfer. Pure GPU-to-GPU copy.
// ─────────────────────────────────────────────────────────────────────────────
struct DrawArraysIndirectCommand {
    uint32_t count;
    uint32_t instanceCount;
    uint32_t first;
    uint32_t baseInstance;
};

__global__ void ResetIndirectCmdKernel(DrawArraysIndirectCommand* cmd)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        cmd->count = 4;
        cmd->instanceCount = 0;
        cmd->first = 0;
        cmd->baseInstance = 0;
    }
}

__global__ void FillInstanceBufferKernel(
    const Particle*  __restrict__ particles,
    InstanceData*    __restrict__ instances,
    DrawArraysIndirectCommand* __restrict__ cmd,
    int maxParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= maxParticles) return;

    const Particle& p = particles[i];
    if (p.lifetime > 0.0f)
    {
        int idx = atomicAdd(&cmd->instanceCount, 1);
        instances[idx].x    = p.x;
        instances[idx].y    = p.y;
        instances[idx].z    = p.z;
        instances[idx].size = p.size * 0.5f; // half-size for shader
        instances[idx].r    = p.r;
        instances[idx].g    = p.g;
        instances[idx].b    = p.b;
        instances[idx].a    = p.a;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Called from GpuRenderer::Draw() (in GpuRenderer.cpp via extern declaration)
// ─────────────────────────────────────────────────────────────────────────────
void LaunchFillInstanceKernel(
    const Particle* particles,
    void*           cudaMappedPtr,
    void*           cudaMappedIndirect,
    int             maxParticles,
    cudaStream_t    stream)
{
    if (maxParticles <= 0) return;

    DrawArraysIndirectCommand* cmd = reinterpret_cast<DrawArraysIndirectCommand*>(cudaMappedIndirect);
    
    // Reset the indirect command buffer
    ResetIndirectCmdKernel<<<1, 1, 0, stream>>>(cmd);

    // Fill the instances, incrementing cmd->instanceCount
    InstanceData* instances = reinterpret_cast<InstanceData*>(cudaMappedPtr);
    int blocks = (maxParticles + 255) / 256;
    FillInstanceBufferKernel<<<blocks, 256, 0, stream>>>(particles, instances, cmd, maxParticles);
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA-GL interop helpers — called from GpuRenderer.cpp
// ─────────────────────────────────────────────────────────────────────────────
bool CudaRegisterGLBuffer(void** outResource, unsigned int vboId)
{
    cudaGraphicsResource_t res;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &res, vboId, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[GpuRenderer] cudaGraphicsGLRegisterBuffer failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    *outResource = reinterpret_cast<void*>(res);
    return true;
}

void CudaUnregisterGLBuffer(void* resource)
{
    if (!resource) return;
    cudaGraphicsUnregisterResource(
        reinterpret_cast<cudaGraphicsResource_t>(resource));
}

bool CudaMapGLBuffer(void* resource, void** outPtr, cudaStream_t stream)
{
    cudaGraphicsResource_t res = reinterpret_cast<cudaGraphicsResource_t>(resource);
    cudaError_t err = cudaGraphicsMapResources(1, &res, stream);
    if (err != cudaSuccess) return false;

    size_t numBytes = 0;
    err = cudaGraphicsResourceGetMappedPointer(outPtr, &numBytes, res);
    return err == cudaSuccess;
}

void CudaUnmapGLBuffer(void* resource, cudaStream_t stream)
{
    cudaGraphicsResource_t res = reinterpret_cast<cudaGraphicsResource_t>(resource);
    cudaGraphicsUnmapResources(1, &res, stream);
}
