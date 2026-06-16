#include "ParticleSystem.h"
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__constant__ SimulationSettings d_Settings;

ParticleSystem::~ParticleSystem()
{
    if (particles) cudaFree(particles);
    if (d_deadIndices) cudaFree(d_deadIndices);
    if (d_deadCount) cudaFree(d_deadCount);
    if (h_stagingBuffer) cudaFreeHost(h_stagingBuffer);
    if (d_randBuffer) cudaFree(d_randBuffer);
    if (m_stream) cudaStreamDestroy(m_stream);
    if (m_frameDone) cudaEventDestroy(m_frameDone);
    if (m_curandGen) curandDestroyGenerator(m_curandGen);

    particles = nullptr;
    d_deadIndices = nullptr;
    d_deadCount = nullptr;
    h_stagingBuffer = nullptr;
    d_randBuffer = nullptr;
    m_stream = nullptr;
    m_frameDone = nullptr;
    m_curandGen = nullptr;
}

__global__ void InitDeadIndicesKernel(int* d_deadIndices, int maxParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < maxParticles)
    {
        d_deadIndices[i] = i; // Slot i is dead and available
    }
}

void ParticleSystem::InitializePool(std::size_t maxP)
{
    if (particles)
    {
        this->~ParticleSystem();
    }
    this->maxParticles = maxP;
    
    cudaMalloc(&particles, maxP * sizeof(Particle));
    cudaMemset(particles, 0, maxP * sizeof(Particle)); // zero out lifetimes
    
    cudaMalloc(&d_deadIndices, maxP * sizeof(int));
    cudaMalloc(&d_deadCount, sizeof(int));
    
    int h_maxP = static_cast<int>(maxP);
    cudaMemcpy(d_deadCount, &h_maxP, sizeof(int), cudaMemcpyHostToDevice);
    h_deadCount = h_maxP;

    cudaHostAlloc(&h_stagingBuffer, maxP * sizeof(Particle), cudaHostAllocDefault);
    cudaMalloc(&d_randBuffer, maxP * 24 * sizeof(float)); 
    
    cudaStreamCreate(&m_stream);
    cudaEventCreate(&m_frameDone);

    curandCreateGenerator(&m_curandGen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(m_curandGen, 1337);

    int numBlocks = (static_cast<int>(maxP) + 255) / 256;
    InitDeadIndicesKernel<<<numBlocks, 256, 0, m_stream>>>(d_deadIndices, h_maxP);
}

void ParticleSystem::Reset(const SimulationSettings& settings)
{
    simulationTime = 0.0f;
    spawnAccumulator = 0.0f;
    burstFired = false;
    nextParticleId = 0;
    curandSetPseudoRandomGeneratorSeed(m_curandGen, settings.randomSeed);
    
    int h_maxP = static_cast<int>(maxParticles);
    cudaMemcpyAsync(d_deadCount, &h_maxP, sizeof(int), cudaMemcpyHostToDevice, m_stream);
    h_deadCount = h_maxP;
    
    cudaMemsetAsync(particles, 0, maxParticles * sizeof(Particle), m_stream);
    
    int numBlocks = (h_maxP + 255) / 256;
    InitDeadIndicesKernel<<<numBlocks, 256, 0, m_stream>>>(d_deadIndices, h_maxP);
}

__device__ float LerpDev(float a, float b, float t)
{
    return a + (b - a) * t;
}

__device__ CudaColor LerpColorDev(const CudaColor& a, const CudaColor& b, float t)
{
    return CudaColor(
        static_cast<std::uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<std::uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<std::uint8_t>(a.b + (b.b - a.b) * t),
        static_cast<std::uint8_t>(a.a + (b.a - a.a) * t)
    );
}

__global__ void SpawnParticlesKernel(Particle* particles, const float* randData, int spawnCount, int* d_deadIndices, int* d_deadCount, uint32_t baseId)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spawnCount) return;

    int deadIdx = atomicSub(d_deadCount, 1) - 1;
    if (deadIdx < 0) {
        atomicAdd(d_deadCount, 1); // undo
        return; // out of slots
    }

    int slot = d_deadIndices[deadIdx];
    int rIdx = idx * 24;
    Particle& p = particles[slot];
    p.id = baseId + idx;

    auto r01 = [&](int offset) { return randData[rIdx + offset]; };
    auto r11 = [&](int offset) { return randData[rIdx + offset] * 2.0f - 1.0f; };

    p.x = d_Settings.emitterX;
    p.y = d_Settings.emitterY;
    p.z = d_Settings.emitterZ;

    float normalX = 0.0f;
    float normalY = 1.0f;
    float normalZ = 0.0f;

    if (d_Settings.shape == EmitterShape::Circle)
    {
        float angle = r01(0) * 2.0f * M_PI;
        float radius = d_Settings.emitRadius;
        if (d_Settings.emissionMode == EmissionMode::Volume)
        {
            radius *= sqrtf(r01(1));
        }
        else
        {
            normalX = cosf(angle);
            normalY = sinf(angle);
            normalZ = 0.0f;
        }
        p.x += cosf(angle) * radius;
        p.y += sinf(angle) * radius;
    }
    else if (d_Settings.shape == EmitterShape::Box)
    {
        if (d_Settings.emissionMode == EmissionMode::Volume)
        {
            p.x += r11(0) * (d_Settings.emitWidth * 0.5f);
            p.y += r11(1) * (d_Settings.emitHeight * 0.5f);
            p.z += r11(12) * (d_Settings.emitDepth * 0.5f);
        }
        else // Surface mode
        {
            float w = d_Settings.emitWidth;
            float h = d_Settings.emitHeight;
            float d = d_Settings.emitDepth;
            float areaX = h * d;
            float areaY = w * d;
            float areaZ = w * h;
            float totalArea = areaX + areaY + areaZ;

            float rVal = r01(0) * totalArea;
            if (rVal < areaX)
            {
                // Left or Right face
                float side = (r01(1) < 0.5f ? -0.5f : 0.5f);
                p.x += side * w;
                p.y += r11(12) * (h * 0.5f);
                p.z += r11(14) * (d * 0.5f);
                normalX = (side < 0.0f ? -1.0f : 1.0f);
                normalY = 0.0f;
                normalZ = 0.0f;
            }
            else if (rVal < areaX + areaY)
            {
                // Bottom or Top face
                float side = (r01(1) < 0.5f ? -0.5f : 0.5f);
                p.x += r11(12) * (w * 0.5f);
                p.y += side * h;
                p.z += r11(14) * (d * 0.5f);
                normalX = 0.0f;
                normalY = (side < 0.0f ? -1.0f : 1.0f);
                normalZ = 0.0f;
            }
            else
            {
                // Back or Front face
                float side = (r01(1) < 0.5f ? -0.5f : 0.5f);
                p.x += r11(12) * (w * 0.5f);
                p.y += r11(14) * (h * 0.5f);
                p.z += side * d;
                normalX = 0.0f;
                normalY = 0.0f;
                normalZ = (side < 0.0f ? -1.0f : 1.0f);
            }
        }
    }
    else if (d_Settings.shape == EmitterShape::Cube)
    {
        float s = d_Settings.emitCubeSize;
        if (d_Settings.emissionMode == EmissionMode::Volume)
        {
            p.x += r11(0) * (s * 0.5f);
            p.y += r11(1) * (s * 0.5f);
            p.z += r11(12) * (s * 0.5f);
        }
        else // Surface mode (6 equal-area faces)
        {
            int face = static_cast<int>(r01(0) * 6.0f);
            if (face == 0)      { p.x += -0.5f * s; p.y += r11(1) * (s * 0.5f); p.z += r11(12) * (s * 0.5f); normalX = -1.0f; normalY = 0.0f; normalZ = 0.0f; }
            else if (face == 1) { p.x +=  0.5f * s; p.y += r11(1) * (s * 0.5f); p.z += r11(12) * (s * 0.5f); normalX =  1.0f; normalY = 0.0f; normalZ = 0.0f; }
            else if (face == 2) { p.y += -0.5f * s; p.x += r11(1) * (s * 0.5f); p.z += r11(12) * (s * 0.5f); normalX = 0.0f; normalY = -1.0f; normalZ = 0.0f; }
            else if (face == 3) { p.y +=  0.5f * s; p.x += r11(1) * (s * 0.5f); p.z += r11(12) * (s * 0.5f); normalX = 0.0f; normalY =  1.0f; normalZ = 0.0f; }
            else if (face == 4) { p.z += -0.5f * s; p.x += r11(1) * (s * 0.5f); p.y += r11(12) * (s * 0.5f); normalX = 0.0f; normalY = 0.0f; normalZ = -1.0f; }
            else                { p.z +=  0.5f * s; p.x += r11(1) * (s * 0.5f); p.y += r11(12) * (s * 0.5f); normalX = 0.0f; normalY = 0.0f; normalZ =  1.0f; }
        }
    }
    else if (d_Settings.shape == EmitterShape::Sphere)
    {
        // Uniform direction vector on sphere
        float zVal = r11(0); // [-1, 1]
        float phi = r01(1) * 2.0f * M_PI;
        float rDir = sqrtf(fmaxf(1.0f - zVal * zVal, 0.0f));
        float dx = rDir * cosf(phi);
        float dy = rDir * sinf(phi);
        float dz = zVal;

        float r = d_Settings.emitRadius;
        if (d_Settings.emissionMode == EmissionMode::Volume)
        {
            // Volumetric density correction: cube root of uniform random
            r *= powf(r01(12), 1.0f / 3.0f);
        }
        else
        {
            normalX = dx;
            normalY = dy;
            normalZ = dz;
        }
        p.x += dx * r;
        p.y += dy * r;
        p.z += dz * r;
    }
    else if (d_Settings.shape == EmitterShape::Grid)
    {
        int cols = d_Settings.gridColumns > 0 ? d_Settings.gridColumns : 1;
        int rows = d_Settings.gridRows > 0 ? d_Settings.gridRows : 1;
        int slices = d_Settings.gridSlices > 0 ? d_Settings.gridSlices : 1;

        int gridIndex = p.id;
        int ix = gridIndex % cols;
        int iy = (gridIndex / cols) % rows;
        int iz = (gridIndex / (cols * rows)) % slices;

        p.x += (ix - (cols - 1) * 0.5f) * d_Settings.gridSpacingX;
        p.y += (iy - (rows - 1) * 0.5f) * d_Settings.gridSpacingY;
        p.z += (iz - (slices - 1) * 0.5f) * d_Settings.gridSpacingZ;
    }

    float vx_std = d_Settings.velocityX + r11(2) * d_Settings.velocityVarianceX;
    float vy_std = d_Settings.velocityY + r11(3) * d_Settings.velocityVarianceY;
    float vz_std = d_Settings.velocityZ + r11(13) * d_Settings.velocityVarianceZ;

    if (d_Settings.emissionMode == EmissionMode::Surface && d_Settings.shape != EmitterShape::Point && d_Settings.shape != EmitterShape::Grid)
    {
        float speed = sqrtf(vx_std * vx_std + vy_std * vy_std + vz_std * vz_std);
        p.vx = normalX * speed;
        p.vy = normalY * speed;
        p.vz = normalZ * speed;
    }
    else
    {
        p.vx = vx_std;
        p.vy = vy_std;
        p.vz = vz_std;
    }

    float lifeVar = r11(4) * d_Settings.lifetimeRandomness * d_Settings.particleLifetime;
    p.maxLifetime = d_Settings.particleLifetime + lifeVar;
    if (p.maxLifetime < 0.05f) p.maxLifetime = 0.05f;
    p.lifetime = p.maxLifetime;

    float massVar = r11(5) * d_Settings.massRandomness * d_Settings.baseMass;
    p.mass = d_Settings.baseMass + massVar;
    if (p.mass < 0.01f) p.mass = 0.01f;

    float sizeVar = r11(6) * d_Settings.sizeRandomness * d_Settings.baseSize;
    p.baseSize = d_Settings.baseSize + sizeVar;
    if (p.baseSize < 0.1f) p.baseSize = 0.1f;
    p.size = p.baseSize;

    p.rotation = d_Settings.initialRotationMin + r01(7) * (d_Settings.initialRotationMax - d_Settings.initialRotationMin);
    p.angularVelocity = d_Settings.angularVelocityMin + r01(8) * (d_Settings.angularVelocityMax - d_Settings.angularVelocityMin);

    CudaColor baseC = d_Settings.colorGradient[0];
    float tintVarR = r11(9) * d_Settings.colorRandomness * 255.0f;
    float tintVarG = r11(10) * d_Settings.colorRandomness * 255.0f;
    float tintVarB = r11(11) * d_Settings.colorRandomness * 255.0f;

    p.baseR = static_cast<std::uint8_t>(fminf(fmaxf(baseC.r + tintVarR, 0.0f), 255.0f));
    p.baseG = static_cast<std::uint8_t>(fminf(fmaxf(baseC.g + tintVarG, 0.0f), 255.0f));
    p.baseB = static_cast<std::uint8_t>(fminf(fmaxf(baseC.b + tintVarB, 0.0f), 255.0f));
    p.baseA = baseC.a;

    p.r = p.baseR;
    p.g = p.baseG;
    p.b = p.baseB;
    p.a = p.baseA;
}

__global__ void UpdateParticlesKernel(Particle* particles, int maxParticles, int* d_deadIndices, int* d_deadCount, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= maxParticles) return;

    Particle& p = particles[i];

    if (p.lifetime <= 0.0f) return; // already dead

    p.lifetime -= dt;
    if (p.lifetime <= 0.0f)
    {
        p.a = 0;
        int deadIdx = atomicAdd(d_deadCount, 1);
        d_deadIndices[deadIdx] = i; // append to free list
        return;
    }

    // Physics
    float forceX = d_Settings.windX;
    float forceY = d_Settings.windY - d_Settings.gravity;
    float forceZ = d_Settings.windZ;
    
    float ax = forceX / p.mass;
    float ay = forceY / p.mass;
    float az = forceZ / p.mass;

    p.vx += ax * dt;
    p.vy += ay * dt;
    p.vz += az * dt;

    p.vx -= p.vx * d_Settings.drag * dt;
    p.vy -= p.vy * d_Settings.drag * dt;
    p.vz -= p.vz * d_Settings.drag * dt;

    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.z += p.vz * dt;

    // Boundary Confinement (Volume mode acts as a boundary box/sphere/circle)
    if (d_Settings.emissionMode == EmissionMode::Volume)
    {
        if (d_Settings.shape == EmitterShape::Box || d_Settings.shape == EmitterShape::Cube)
        {
            float w = (d_Settings.shape == EmitterShape::Cube) ? d_Settings.emitCubeSize : d_Settings.emitWidth;
            float h = (d_Settings.shape == EmitterShape::Cube) ? d_Settings.emitCubeSize : d_Settings.emitHeight;
            float d = (d_Settings.shape == EmitterShape::Cube) ? d_Settings.emitCubeSize : d_Settings.emitDepth;

            float minX = d_Settings.emitterX - w * 0.5f;
            float maxX = d_Settings.emitterX + w * 0.5f;
            float minY = d_Settings.emitterY - h * 0.5f;
            float maxY = d_Settings.emitterY + h * 0.5f;
            float minZ = d_Settings.emitterZ - d * 0.5f;
            float maxZ = d_Settings.emitterZ + d * 0.5f;

            if (p.x < minX) { p.x = minX; p.vx = -p.vx * 0.8f; }
            if (p.x > maxX) { p.x = maxX; p.vx = -p.vx * 0.8f; }
            if (p.y < minY) { p.y = minY; p.vy = -p.vy * 0.8f; }
            if (p.y > maxY) { p.y = maxY; p.vy = -p.vy * 0.8f; }
            if (p.z < minZ) { p.z = minZ; p.vz = -p.vz * 0.8f; }
            if (p.z > maxZ) { p.z = maxZ; p.vz = -p.vz * 0.8f; }
        }
        else if (d_Settings.shape == EmitterShape::Circle)
        {
            float dx = p.x - d_Settings.emitterX;
            float dy = p.y - d_Settings.emitterY;
            float dist = sqrtf(dx * dx + dy * dy);
            float R = d_Settings.emitRadius;
            if (dist > R && dist > 0.0f)
            {
                float nx = dx / dist;
                float ny = dy / dist;
                
                // Reflection: V_new = V - 2(V.N)N
                float dotVal = p.vx * nx + p.vy * ny;
                p.vx = (p.vx - 2.0f * dotVal * nx) * 0.8f;
                p.vy = (p.vy - 2.0f * dotVal * ny) * 0.8f;
                
                p.x = d_Settings.emitterX + nx * R;
                p.y = d_Settings.emitterY + ny * R;
            }
        }
        else if (d_Settings.shape == EmitterShape::Sphere)
        {
            float dx = p.x - d_Settings.emitterX;
            float dy = p.y - d_Settings.emitterY;
            float dz = p.z - d_Settings.emitterZ;
            float dist = sqrtf(dx * dx + dy * dy + dz * dz);
            float R = d_Settings.emitRadius;
            if (dist > R && dist > 0.0f)
            {
                float nx = dx / dist;
                float ny = dy / dist;
                float nz = dz / dist;
                
                // Reflection
                float dotVal = p.vx * nx + p.vy * ny + p.vz * nz;
                p.vx = (p.vx - 2.0f * dotVal * nx) * 0.8f;
                p.vy = (p.vy - 2.0f * dotVal * ny) * 0.8f;
                p.vz = (p.vz - 2.0f * dotVal * nz) * 0.8f;
                
                p.x = d_Settings.emitterX + nx * R;
                p.y = d_Settings.emitterY + ny * R;
                p.z = d_Settings.emitterZ + nz * R;
            }
        }
    }

    p.rotation += p.angularVelocity * dt;

    // Visual Evolution
    float t = 1.0f - (p.lifetime / p.maxLifetime); 

    p.size = p.baseSize * LerpDev(d_Settings.sizeStart, d_Settings.sizeEnd, t);

    CudaColor gradColor;
    if (t < 0.333333f)
        gradColor = LerpColorDev(d_Settings.colorGradient[0], d_Settings.colorGradient[1], t * 3.0f);
    else if (t < 0.666666f)
        gradColor = LerpColorDev(d_Settings.colorGradient[1], d_Settings.colorGradient[2], (t - 0.333333f) * 3.0f);
    else
        gradColor = LerpColorDev(d_Settings.colorGradient[2], d_Settings.colorGradient[3], (t - 0.666666f) * 3.0f);

    p.r = static_cast<std::uint8_t>((p.baseR * gradColor.r) / 255);
    p.g = static_cast<std::uint8_t>((p.baseG * gradColor.g) / 255);
    p.b = static_cast<std::uint8_t>((p.baseB * gradColor.b) / 255);
    p.a = static_cast<std::uint8_t>((p.baseA * gradColor.a) / 255);
}

void ParticleSystem::Update(float dt, const SimulationSettings& settings, SimulationStats& stats)
{
    if (settings.paused) return;

    cudaMemcpyToSymbolAsync(d_Settings, &settings, sizeof(SimulationSettings), 0, cudaMemcpyHostToDevice, m_stream);

    simulationTime += dt;
    stats.spawnedThisFrame = 0;
    stats.killedThisFrame = 0;

    int spawnCount = 0;

    if (simulationTime >= settings.startTime && simulationTime <= settings.endTime)
    {
        if (!burstFired && settings.burstCount > 0)
        {
            spawnCount += settings.burstCount;
            burstFired = true;
        }

        if (settings.spawnRate > 0.0f)
        {
            spawnAccumulator += settings.spawnRate * dt;
            int spawnsThisFrame = static_cast<int>(spawnAccumulator);
            spawnAccumulator -= spawnsThisFrame;
            spawnCount += spawnsThisFrame;
        }
    }

    // Clamp spawn count to available dead slots (h_deadCount is 1 frame behind, which is fine)
    if (spawnCount > h_deadCount)
    {
        spawnCount = h_deadCount;
    }

    if (spawnCount > 0)
    {
        curandGenerateUniform(m_curandGen, d_randBuffer, spawnCount * 24);

        int blockSize = 256;
        int numBlocks = (spawnCount + blockSize - 1) / blockSize;
        SpawnParticlesKernel<<<numBlocks, blockSize, 0, m_stream>>>(particles, d_randBuffer, spawnCount, d_deadIndices, d_deadCount, nextParticleId);
        
        nextParticleId += spawnCount;
        stats.spawnedThisFrame = spawnCount;
    }

    // ZERO-SYNC UPDATE
    int maxP = static_cast<int>(maxParticles);
    int blockSize = 256;
    int numBlocks = (maxP + blockSize - 1) / blockSize;
    
    UpdateParticlesKernel<<<numBlocks, blockSize, 0, m_stream>>>(particles, maxP, d_deadIndices, d_deadCount, dt);

    // Read back dead count asynchronously for NEXT frame
    cudaMemcpyAsync(&h_deadCount, d_deadCount, sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    
    stats.activeParticles = static_cast<uint32_t>(maxParticles - h_deadCount);
}
