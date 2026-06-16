#pragma once
#include <cstdint>
#include <SFML/Graphics.hpp>
#include "InstanceData.h"

class ParticleSystem;

// GPU-accelerated renderer using CUDA-OpenGL interop + instanced rendering.
//
// Architecture:
//   1. CUDA physics kernel writes final particle state (ParticleSystem.cu)
//   2. FillInstanceBufferKernel reads particles[], writes InstanceData[] VBO
//      directly in GPU memory — zero CPU readback, zero PCIe transfer
//   3. OpenGL draws N quads via glDrawArraysInstanced in a single draw call
//
// Must call Init() after the sf::RenderWindow has been created (OpenGL context
// must exist before CUDA-GL interop registration).
class GpuRenderer
{
public:
    GpuRenderer() = default;
    ~GpuRenderer();

    // Call once after sf::RenderWindow creation.
    // Returns false if initialization fails (falls back to CPU renderer).
    bool Init(sf::RenderWindow& window, int maxParticles);

    // Main draw call — maps VBO, launches fill kernel, draws, unmaps.
    // viewMatrix: column-major 4x4 view matrix from Camera.
    // projMatrix: column-major 4x4 projection matrix from Camera.
    void Draw(sf::RenderWindow& window, const ParticleSystem& ps, const float* viewMatrix, const float* projMatrix, const struct SimulationSettings& settings);

    bool IsInitialized() const { return m_initialized; }

private:
    bool CompileShaders();
    void SetupVAO();
    void SetupGrid();

    // OpenGL objects
    uint32_t m_vao           = 0;
    uint32_t m_quadVbo       = 0;  // 4-vertex unit quad (shared geometry)
    uint32_t m_instanceVbo   = 0;  // per-instance data written by CUDA kernel
    uint32_t m_shaderProgram = 0;
    int      m_maxParticles  = 0;

    // Viewport Grid Floor rendering objects
    uint32_t m_gridVao           = 0;
    uint32_t m_gridVbo           = 0;
    uint32_t m_gridShaderProgram = 0;
    int      m_gridVertexCount   = 0;

    // CUDA-GL interop handle (opaque — actually cudaGraphicsResource_t)
    void* m_cudaResource = nullptr;
    
    uint32_t m_indirectVbo = 0;
    void* m_cudaIndirectResource = nullptr;

    bool m_initialized = false;
};
