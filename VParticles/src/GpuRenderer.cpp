#include "GpuRenderer.h"
#include "ParticleSystem.h"
#include "InstanceData.h"

// GLEW must be included before any other GL headers
#include <GL/glew.h>
#include <SFML/OpenGL.hpp>

#include <cstdio>
#include <cstring>

// Forward declarations of CUDA helpers defined in GpuRenderer.cu
bool CudaRegisterGLBuffer(void** outResource, unsigned int vboId);
void CudaUnregisterGLBuffer(void* resource);
bool CudaMapGLBuffer(void* resource, void** outPtr, cudaStream_t stream);
void CudaUnmapGLBuffer(void* resource, cudaStream_t stream);
void LaunchFillInstanceKernel(const Particle* particles, void* cudaMappedPtr, void* cudaMappedIndirect, int maxParticles, cudaStream_t stream);

// ─── GLSL Shaders (3D MVP pipeline) ─────────────────────────────────────────

static const char* kVertexShaderSrc = R"glsl(
#version 330 core

// Shared quad geometry (4 vertices, -0.5 to +0.5)
layout(location = 0) in vec2 a_QuadPos;

// Per-instance data written by CUDA kernel
layout(location = 1) in vec3  i_WorldPos;   // x, y, z
layout(location = 2) in float i_HalfSize;
layout(location = 3) in vec4  i_Color;

out vec4 v_Color;

uniform mat4 u_View;
uniform mat4 u_Proj;

void main()
{
    // Transform center to camera space
    vec4 viewPos = u_View * vec4(i_WorldPos, 1.0);

    // Apply the quad offset in view space (this keeps it screen-aligned and square)
    viewPos.xy += a_QuadPos * (i_HalfSize * 2.0);

    // Project to clip space
    gl_Position = u_Proj * viewPos;
    v_Color     = i_Color;
}
)glsl";

static const char* kFragmentShaderSrc = R"glsl(
#version 330 core
in  vec4 v_Color;
out vec4 FragColor;
void main()
{
    FragColor = v_Color;
}
)glsl";

// ─── Helpers ─────────────────────────────────────────────────────────────────

static GLuint CompileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        fprintf(stderr, "[GpuRenderer] Shader compile error:\n%s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// ─── GpuRenderer ─────────────────────────────────────────────────────────────

GpuRenderer::~GpuRenderer()
{
    CudaUnregisterGLBuffer(m_cudaResource);
    CudaUnregisterGLBuffer(m_cudaIndirectResource);
    if (m_instanceVbo)   glDeleteBuffers(1, &m_instanceVbo);
    if (m_indirectVbo)   glDeleteBuffers(1, &m_indirectVbo);
    if (m_quadVbo)       glDeleteBuffers(1, &m_quadVbo);
    if (m_vao)           glDeleteVertexArrays(1, &m_vao);
    if (m_shaderProgram) glDeleteProgram(m_shaderProgram);
}

bool GpuRenderer::Init(sf::RenderWindow& window, int maxParticles)
{
    (void)window.setActive(true);

    // Initialize GLEW (loads all GL extensions)
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        fprintf(stderr, "[GpuRenderer] glewInit failed: %s\n",
                glewGetErrorString(glewErr));
        return false;
    }

    // Compile shaders
    GLuint vert = CompileShader(GL_VERTEX_SHADER, kVertexShaderSrc);
    GLuint frag = CompileShader(GL_FRAGMENT_SHADER, kFragmentShaderSrc);
    if (!vert || !frag) return false;

    m_shaderProgram = glCreateProgram();
    glAttachShader(m_shaderProgram, vert);
    glAttachShader(m_shaderProgram, frag);
    glLinkProgram(m_shaderProgram);
    glDeleteShader(vert);
    glDeleteShader(frag);

    GLint linked = 0;
    glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        char log[512];
        glGetProgramInfoLog(m_shaderProgram, 512, nullptr, log);
        fprintf(stderr, "[GpuRenderer] Shader link error:\n%s\n", log);
        return false;
    }

    // Unit quad: two triangles as a triangle strip
    // Positions are in local space [-0.5, 0.5]
    static const float kQuad[8] = {
        -0.5f,  0.5f,  // top-left
        -0.5f, -0.5f,  // bottom-left
         0.5f,  0.5f,  // top-right
         0.5f, -0.5f,  // bottom-right
    };

    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    // Quad VBO (location 0) — same for every instance
    glGenBuffers(1, &m_quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuad), kQuad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glVertexAttribDivisor(0, 0); // advance per-vertex

    // Instance VBO (locations 1,2,3) — written each frame by CUDA kernel
    m_maxParticles = maxParticles;
    glGenBuffers(1, &m_instanceVbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_instanceVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 (size_t)maxParticles * sizeof(InstanceData),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    const GLsizei stride = sizeof(InstanceData);

    // i_WorldPos  — offset 0,  3 floats (x, y, z)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          (void*)offsetof(InstanceData, x));
    glVertexAttribDivisor(1, 1); // advance per-instance

    // i_HalfSize  — offset 12, 1 float
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride,
                          (void*)offsetof(InstanceData, size));
    glVertexAttribDivisor(2, 1);

    // i_Color     — offset 16, 4 unsigned bytes normalized
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride,
                          (void*)offsetof(InstanceData, r));
    glVertexAttribDivisor(3, 1);

    glBindVertexArray(0);

    // Register instance VBO with CUDA
    if (!CudaRegisterGLBuffer(&m_cudaResource, m_instanceVbo))
        return false;

    // Create Indirect VBO
    glGenBuffers(1, &m_indirectVbo);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_indirectVbo);
    
    struct DrawArraysIndirectCommand {
        uint32_t count;
        uint32_t instanceCount;
        uint32_t first;
        uint32_t baseInstance;
    };
    DrawArraysIndirectCommand cmd = { 4, 0, 0, 0 };
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawArraysIndirectCommand), &cmd, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    if (!CudaRegisterGLBuffer(&m_cudaIndirectResource, m_indirectVbo))
        return false;

    m_initialized = true;
    printf("[GpuRenderer] Initialized — CUDA-GL interop active, max %d particles (3D Zero-Sync)\n",
           maxParticles);
    return true;
}

void GpuRenderer::Draw(sf::RenderWindow& window, const ParticleSystem& ps, const float* viewMatrix, const float* projMatrix)
{
    if (!m_initialized) return;

    // ── Step 1: Map the VBOs into CUDA address space ──
    void* mappedPtr = nullptr;
    void* mappedIndirect = nullptr;
    if (!CudaMapGLBuffer(m_cudaResource, &mappedPtr, ps.m_stream) ||
        !CudaMapGLBuffer(m_cudaIndirectResource, &mappedIndirect, ps.m_stream))
    {
        fprintf(stderr, "[GpuRenderer] Failed to map GL buffers\n");
        return;
    }

    // ── Step 2: GPU kernel fills VBO directly — no CPU readback ──
    LaunchFillInstanceKernel(ps.particles, mappedPtr, mappedIndirect, m_maxParticles, ps.m_stream);

    // ── Step 3: Unmap (gives ownership back to OpenGL) ──
    CudaUnmapGLBuffer(m_cudaResource, ps.m_stream);
    CudaUnmapGLBuffer(m_cudaIndirectResource, ps.m_stream);
    
    // (cudaStreamSynchronize removed to allow CPU/GPU overlap)

    // ── Step 4: Raw OpenGL instanced draw (3D) ──
    (void)window.setActive(true);

    // Let SFML know we're about to use raw GL
    window.pushGLStates();

    glUseProgram(m_shaderProgram);

    // Upload view and projection matrices
    GLint viewLoc = glGetUniformLocation(m_shaderProgram, "u_View");
    GLint projLoc = glGetUniformLocation(m_shaderProgram, "u_Proj");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projMatrix);

    // Enable alpha blending and explicitly disable depth testing for transparent particles
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_indirectVbo);
    glDrawArraysIndirect(GL_TRIANGLE_STRIP, nullptr);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
    glBindVertexArray(0);

    glUseProgram(0);

    // Restore SFML GL state
    window.popGLStates();
}
