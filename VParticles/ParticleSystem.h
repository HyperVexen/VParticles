#pragma once

#include <vector>
#include "Particle.h"
#include "SimulationSettings.h"

class ParticleSystem
{
public:

    std::vector<Particle> particles;

    void Create(int count);
    void Create(int count, const SimulationSettings& settings);

    void Reset(int count, const SimulationSettings& settings);
    void Resize(int count, const SimulationSettings& settings);

    void SetVelocity(float velocityX, float velocityY);

    void Update(float dt, float windowWidth, float windowHeight);
};
