#pragma once

#include <vector>
#include "Particle.h"
#include "SimulationSettings.h"

#include <random>
#include "PerformanceStats.h"
#include <cuda_runtime.h>
#include <curand.h>

class ParticleSystem
{
public:

    Particle* particles = nullptr;
    cudaStream_t m_stream = nullptr;
    int* d_deadIndices = nullptr;
    int* d_deadCount = nullptr;
    int h_deadCount = 0;
    
    Particle* h_stagingBuffer = nullptr;
    
    cudaEvent_t m_frameDone = nullptr;
    curandGenerator_t m_curandGen = nullptr;
    float* d_randBuffer = nullptr;
    
    std::size_t maxParticles = 0;

    std::size_t GetActiveCount() const { return maxParticles - h_deadCount; }
    float GetSimulationTime() const { return simulationTime; }
    void SetSimulationTime(float t) { simulationTime = t; }

    ~ParticleSystem();

    void InitializePool(std::size_t maxParticles);

    void Reset(const SimulationSettings& settings);

    bool SpawnParticle(const SimulationSettings& settings);

    void Update(float dt, const SimulationSettings& settings, SimulationStats& stats);

private:
    float simulationTime = 0.0f;
    float spawnAccumulator = 0.0f;
    bool burstFired = false;
    uint32_t nextParticleId = 0;

    std::mt19937 rng;
};
