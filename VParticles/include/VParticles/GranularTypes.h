#pragma once

#include "VParticles/SimulationTypes.h"
#include <cstdint>

namespace vparticles {

enum class SimulationMode : uint8_t {
    Effects  = 0,
    Granular = 1
};

enum class GranularQuality : uint8_t {
    Draft   = 0,   // 4 substeps x 2 iterations
    Preview = 1,   // 4 substeps x 4 iterations
    Final   = 2    // 8 substeps x 4 iterations
};

constexpr uint32_t kMaxGrainMaterials = 64;

struct GrainMaterial {
    float radius = 0.01f;
    float density = 1600.0f;        // kg/m^3 (sand ~1600)
    float staticFriction = 0.45f;
    float dynamicFriction = 0.40f;
    float restitution = 0.1f;
    float drag = 0.05f;
    float damping = 2.0f;
    float cohesion = 0.0f;
    float repulsion = 1.0f;         // compliance weight
};

struct GranularSettings {
    float fixedDt = 1.0f / 60.0f;
    uint32_t substeps = 4;
    uint32_t iterations = 4;
    float grainRadius = 0.01f;
    float radiusJitter = 0.0f;      // +/- fraction of radius (0.0 to 0.2)
    float skinRatio = 0.20f;        // fraction of radius used as Verlet skin (e.g. 0.20 = 20%)
    GranularQuality quality = GranularQuality::Preview;
    Float3 gravity = {0.0f, -9.81f, 0.0f};
    CollisionPlane groundPlane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.1f, 0.45f, 1};
    CollisionSphere obstacleSphere = {};
    CollisionBox containerBox = {};
    GrainMaterial materials[kMaxGrainMaterials] = {};
    uint32_t materialCount = 1;

    void applyQuality(GranularQuality q) {
        quality = q;
        switch (q) {
            case GranularQuality::Draft:
                substeps = 4;
                iterations = 2;
                break;
            case GranularQuality::Preview:
                substeps = 4;
                iterations = 4;
                break;
            case GranularQuality::Final:
                substeps = 8;
                iterations = 4;
                break;
        }
    }
};

// Section 4.2 Dense FP32 Structure of Arrays for Granular Simulation
struct GranularPool {
    float4* pos = nullptr;          // xyz + radius (16 bytes)
    float4* vel = nullptr;          // xyz + invMass (16 bytes)
    float4* pred = nullptr;         // xyz + packed sleep counter / contact bit (16 bytes)
    uint32_t* tint = nullptr;       // RGBA8 hash-based color variation (4 bytes)
    uint16_t* materialId = nullptr; // 2 bytes
    uint16_t* flags = nullptr;      // 2 bytes (bit 0 = asleep, bit 1 = surface)
    uint32_t* grainId = nullptr;    // stable grain ID across reorders (4 bytes)
    uint32_t capacity = 0;
    uint32_t aliveCount = 0;
};

struct GranularStats {
    uint64_t frameIndex = 0;
    float simTime = 0.0f;
    uint32_t capacity = 0;
    uint32_t aliveCount = 0;
    uint32_t substeps = 0;
    uint32_t iterations = 0;
    float meanHeight = 0.0f;
    float maxHeight = 0.0f;
    float kineticEnergy = 0.0f;
    float stepMs = 0.0f;
    float totalMs = 0.0f;
    uint64_t neighborRebuildCount = 0;
    uint32_t neighborOverflowCount = 0;
    float neighborRebuildMs = 0.0f;
    float neighborAmortizedMs = 0.0f;
};

} // namespace vparticles
