#pragma once

#include <vector>
#include "Particle.h"

class ParticleSystem
{
public:

    std::vector<Particle> particles;

    void Create(int count);

    void Update(float dt, float windowWidth, float windowHeight);
};