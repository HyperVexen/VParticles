#include "ParticleSystem.h"

void ParticleSystem::Create(int count)
{
    SimulationSettings settings;

    Create(count, settings);
}

void ParticleSystem::Create(int count, const SimulationSettings& settings)
{
    particles.clear();

    for (int i = 0; i < count; i++)
    {
        Particle p;

        p.x = SimulationDefaults::SpawnX;
        p.y = SimulationDefaults::SpawnY;

        p.vx = settings.velocityX;
        p.vy = settings.velocityY;

        particles.push_back(p);
    }
}

void ParticleSystem::Reset(int count, const SimulationSettings& settings)
{
    Create(count, settings);
}

void ParticleSystem::SetVelocity(float velocityX, float velocityY)
{
    for (auto& p : particles)
    {
        p.vx = velocityX;
        p.vy = velocityY;
    }
}

void ParticleSystem::Update(
    float dt,
    float windowWidth,
    float windowHeight)
{
    for (auto& p : particles)
    {
        p.x += p.vx * dt;
        p.y += p.vy * dt;

        if (p.x < 0)
        {
            p.x = 0;
            p.vx *= -1;
        }

        if (p.x > windowWidth)
        {
            p.x = windowWidth;
            p.vx *= -1;
        }

        if (p.y < 0)
        {
            p.y = 0;
            p.vy *= -1;
        }

        if (p.y > windowHeight)
        {
            p.y = windowHeight;
            p.vy *= -1;
        }
    }
}
