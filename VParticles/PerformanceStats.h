#pragma once
#include <cstdint>

struct SimulationStats
{
    uint32_t activeParticles = 0;
    uint32_t spawnedThisFrame = 0;
    uint32_t killedThisFrame = 0;

    float updateTimeMs = 0.0f;
    float renderTimeMs = 0.0f;
};

struct PerformanceStats
{
    float fps = 0.0f;
    float averageFps = 0.0f;
    float frameTimeMs = 0.0f;
};
