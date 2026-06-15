#pragma once

#include <vector>
#include "Particle.h"
#include "SimulationSettings.h"

#include <random>
#include "PerformanceStats.h"

class ParticleSystem
{
public:

    std::vector<Particle> particles;
    std::size_t activeCount = 0;

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
