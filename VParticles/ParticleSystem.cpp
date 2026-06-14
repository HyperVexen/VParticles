#include "ParticleSystem.h"

void ParticleSystem::Create(int count)
{
    particles.clear();

    for (int i = 0; i < count; i++)
    {
        Particle p;

        p.x = 400.0f;
        p.y = 300.0f;

        p.vx = 100.0f;
        p.vy = 80.0f;

        particles.push_back(p);
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