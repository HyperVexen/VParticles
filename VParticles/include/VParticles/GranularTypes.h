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

constexpr uint16_t kGrainFlagAsleep  = 0x0001;
constexpr uint16_t kGrainFlagSurface = 0x0002;

enum class SandScenario : uint8_t {
    None           = 0,
    Pile           = 1,
    ColumnCollapse = 2,
    Funnel         = 3,
    Dune           = 4,
    Cliff          = 5,
    BondedBlock    = 6,
    Clumps         = 7
};

constexpr uint32_t kMaxGrainMaterials = 64;
constexpr uint32_t kMaxBondsPerGrain = 12;

struct GrainBond {
    uint32_t targetGrainId = 0xFFFFFFFF; // Stable grain ID of connected partner (0xFFFFFFFF = empty/broken)
    float restLength = 0.0f;             // Rest distance d0 (meters)
    float compliance = 0.0f;             // XPBD compliance alpha (m/N, 0 = rigid)
    float breakForce = 0.0f;             // Rupture force threshold (N, 0 = unbreakable)
};

struct GrainMaterial {
    float radius = 0.01f;
    float density = 1600.0f;        // kg/m^3 (sand ~1600)
    float staticFriction = 0.58f;   // quartz sand tan(repose ~30-34 deg)
    float dynamicFriction = 0.50f;
    float restitution = 0.05f;
    float drag = 0.05f;
    float damping = 2.0f;
    float cohesion = 0.0f;          // Damp sand cohesion strength
    float repulsion = 1.0f;         // compliance weight
    float cohesionRuptureRatio = 0.15f; // Rupture distance ratio (1 + ratio) * (r_i + r_j)
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
    CollisionPlane groundPlane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.05f, 0.58f, 1};
    CollisionSphere obstacleSphere = {};
    CollisionBox containerBox = {};
    GrainMaterial materials[kMaxGrainMaterials] = {};
    uint32_t materialCount = 1;

    // Phase 11E: Sleeping & Scenarios
    bool enableSleeping = true;
    float sleepSpeedThreshold = 0.02f;   // speed below which sleep counter increments
    uint32_t sleepFrames = 10;           // consecutive frames below threshold to fall asleep
    SandScenario scenario = SandScenario::None;

    // Phase 11F: Constraints, Cohesion, Bonds & Clumping
    float defaultCohesion = 0.0f;        // Damp sand surface tension / capillary cohesion
    float bondCompliance = 0.0f;         // Glue bond XPBD compliance (m/N, 0 = rigid)
    float bondBreakForce = 0.0f;         // Glue bond tensile rupture threshold (N, 0 = unbreakable)
    float clumpRadius = 0.05f;           // Clump cluster radius (m)

    // Phase 11G: Dynamic Kinematic Moving Colliders & Scale Testing
    int kinematicColliderEnabled = 0;    // 1 = active moving kinematic collider
    int kinematicColliderType = 0;       // 0 = linear oscillating plow, 1 = circular rotating churner
    Float3 kinematicOrigin = {0.0f, 0.12f, 0.0f};
    Float3 kinematicVelocity = {0.0f, 0.0f, 0.0f};
    float kinematicRadius = 0.08f;       // Radius of moving plow / churner
    float kinematicSpeed = 1.5f;         // Linear / angular frequency (rad/s)
    float kinematicAmplitude = 0.35f;    // Sweep oscillation distance (m)

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
    uint32_t asleepCount = 0;
    uint64_t neighborRebuildCount = 0;
    uint32_t neighborOverflowCount = 0;
    float neighborRebuildMs = 0.0f;
    float neighborAmortizedMs = 0.0f;
    uint32_t activeBondCount = 0;
    uint32_t brokenBondCount = 0;
};

} // namespace vparticles
