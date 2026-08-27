#pragma once

#include <cstdint>

namespace vparticles {

struct Float3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Color4 {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
};

struct EmitterDesc {
    Float3 position = {};
    Float3 velocity = {0.0f, 4.0f, 0.0f};
    Float3 velocityVariance = {1.0f, 1.0f, 1.0f};
    Color4 color = {1.0f, 0.55f, 0.18f, 1.0f};
    float spawnRate = 100000.0f;
    float lifetime = 4.0f;
    float lifetimeVariance = 0.25f;
};

struct SimulationSettings {
    Float3 gravity = {0.0f, -9.81f, 0.0f};
    Float3 wind = {};
    float drag = 0.05f;
    uint32_t seed = 0xA341316Cu;
};

struct SimulationStats {
    // Telemetry is delivered asynchronously. These identify the GPU frame that
    // produced this sample and how many later submissions were already queued
    // when the host observed it.
    uint64_t frameIndex = 0;
    uint32_t telemetryLatencyFrames = 0;
    bool valid = false;
    uint32_t capacity = 0;
    uint32_t aliveCount = 0;
    uint32_t requestedSpawn = 0;
    uint32_t spawned = 0;
    uint32_t dropped = 0;
    uint32_t deadCount = 0;
    float spawnMs = 0.0f;
    float simulateMs = 0.0f;
    float compactMs = 0.0f;
    float totalMs = 0.0f;
};

} // namespace vparticles
