#pragma once

namespace SimulationDefaults
{
    inline constexpr int ParticleCount = 1000;

    inline constexpr float SpawnX = 400.0f;
    inline constexpr float SpawnY = 300.0f;

    inline constexpr float VelocityX = 100.0f;
    inline constexpr float VelocityY = 80.0f;

    inline constexpr float ParticleRadius = 2.0f;
}

struct SimulationSettings
{
    float velocityX = SimulationDefaults::VelocityX;
    float velocityY = SimulationDefaults::VelocityY;
    int particleCount = SimulationDefaults::ParticleCount;

    bool paused = false;
};
