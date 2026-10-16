#pragma once

#include "VParticles/GranularTypes.h"
#include "VParticles/NeighborTypes.h"
#include <cstdint>

namespace vparticles {

struct NeighborValidationResult {
    bool passed = false;

    // Gate 11C-1: Ground-truth 10K neighbor pair match
    bool pairMatchPassed = false;
    uint32_t grainCount10K = 10000;
    uint64_t totalPairsFound = 0;
    uint64_t totalPairsGroundTruth = 0;
    uint32_t falsePositives = 0;
    uint32_t falseNegatives = 0;
    uint32_t maxNeighborsFound = 0;
    float avgNeighborsFound = 0.0f;
    uint32_t overflowCount10K = 0;

    // Gate 11C-2: 1M grain latency & amortization
    bool latencyPassed = false;
    uint32_t grainCount1M = 1000000;
    float keygenMs = 0.0f;
    float sortMs = 0.0f;
    float boundsMs = 0.0f;
    float reorderMs = 0.0f;
    float verletBuildMs = 0.0f;
    float totalRebuildMs = 0.0f;
    float dispCheckMs = 0.0f;
    float amortizedRebuildMs = 0.0f;

    // Gate 11C-3: Determinism across 5 runs
    bool determinismPassed = false;
    uint64_t runHashes[5] = {};

    // Gate 11C-4: Overflow counter tracking
    bool overflowHandlingPassed = false;
    uint32_t syntheticOverflowDetected = 0;
};

NeighborValidationResult runNeighborValidationSuite(uint32_t capacity = 1000000, float skinRatio = 0.20f);

struct SolverValidationResult {
    bool passed = false;

    // Gate 11D-1: Stacking & Non-Penetration (100K grains)
    bool penetrationPassed = false;
    uint32_t grainCount100K = 100000;
    float maxPenetration = 0.0f;       // must be < 10% of radius (e.g. < 0.001m for r=0.01m)
    float maxPenetrationRatio = 0.0f;  // maxPenetration / radius (must be < 0.10)
    float meanPenetration = 0.0f;

    // Gate 11D-2: Kinetic Energy Decay
    bool energyDecayPassed = false;
    float peakKE = 0.0f;
    float finalKE = 0.0f;
    float keRatio = 0.0f;              // finalKE / peakKE (must be < 0.05)

    // Gate 11D-3: 250K Grain Stability & NaN Safety
    bool stabilityPassed = false;
    uint32_t nanCount = 0;
    uint32_t infCount = 0;
    float maxVelocity = 0.0f;

    // Gate 11D-4: Solver Latency Budget
    bool latencyPassed = false;
    float solvePassMs250K = 0.0f;      // budget: < 1.0 ms
    float solvePassMs1M = 0.0f;        // budget: < 3.5 ms
};

SolverValidationResult runSolverValidationSuite(uint32_t capacity = 1000000);

struct MaterialValidationResult {
    bool passed = false;

    // Gate 11E-1: Sand Pile Angle of Repose (50K - 100K grains)
    bool angleOfReposePassed = false;
    float measuredAngleDeg = 0.0f;     // Target: 30.0 deg <= angle <= 35.5 deg (quartz sand)
    float peakHeight = 0.0f;
    float baseRadius = 0.0f;

    // Gate 11E-2: Polydisperse Stacking & Mixture (+/- 20% radius jitter + mass scaling)
    bool polydispersePassed = false;
    uint32_t polydisperseGrains = 100000;
    float polydisperseMaxPen = 0.0f;   // Target: < 10% radius
    float polydisperseMaxPenRatio = 0.0f;

    // Gate 11E-3: Grain Sleeping & Wake-Up Optimization
    bool sleepingPassed = false;
    float sleepingFraction = 0.0f;     // Target: > 80% asleep in settled pile
    float wakeUpLatencyMs = 0.0f;      // Step latency during sleep
    bool wakeUpVerified = false;       // Wakes up on external perturbation

    // Gate 11E-4: Sand Scenarios Suite
    bool scenariosPassed = false;
    uint32_t scenariosTested = 0;      // 4 scenarios (pile, column_collapse, funnel, dune)
};

MaterialValidationResult runMaterialValidationSuite(uint32_t capacity = 1000000);

struct ConstraintValidationResult {
    bool passed = false;

    // Gate 11F-1: Damp-Sand Cliff Stability & Load-induced Crumble
    bool cliffPassed = false;
    float dryCliffApexDropRatio = 0.0f;    // In dry sand, apex drops > 30% (avalanche to repose angle)
    float dampCliffApexDropRatio = 0.0f;   // In damp sand, apex retains > 95% height (vertical face holds)
    bool loadCrumbleVerified = false;      // Under overburden load, cliff shears and crumbles into debris

    // Gate 11F-2: Bonded Block Tensile Break Force Parity Across Substeps
    bool breakForcePassed = false;
    float breakForce4Substeps = 0.0f;      // Load at fracture with 4 substeps
    float breakForce8Substeps = 0.0f;      // Load at fracture with 8 substeps
    float breakForceErrorRatio = 0.0f;     // |F4 - F8| / F4 (Target: <= 15.0%)

    // Gate 11F-3: Bond Count Integrity Across 1000 Reorders
    bool reorderIntegrityPassed = false;
    uint32_t initialBonds = 0;
    uint32_t finalBonds = 0;
    uint32_t reorderIterations = 1000;
    uint32_t corruptedBonds = 0;

    // Gate 11F-4: Clump Generation & Impact Crumbling
    bool clumpsPassed = false;
    uint32_t totalClumpsSpawned = 0;
    uint32_t initialIntraClumpBonds = 0;
    uint32_t postImpactActiveBonds = 0;
    float bondRuptureFraction = 0.0f;      // Fraction of bonds broken on high-energy impact (> 20%)
};

ConstraintValidationResult runConstraintValidationSuite(uint32_t capacity = 1000000);

struct ScaleValidationResult {
    bool passed = false;

    // Gate 11G-1: Interactive viewer simulation latency budget (< 8.33 ms @ 100K -> 120+ FPS, < 16.6 ms @ 250K -> 60+ FPS)
    bool viewerLatencyPassed = false;
    float simMs100K = 0.0f;           // Budget: < 8.33 ms (allows 120+ FPS)
    float simMs250K = 0.0f;           // Budget: < 16.6 ms (allows 60+ FPS)
    float fpsEquivalent250K = 0.0f;
    float simMs1M = 0.0f;             // Viewport 1M scaling playback latency
    float fpsEquivalent1M = 0.0f;

    // Gate 11G-2: Multi-million awake grain scaling (1M to 5M awake grains stress test)
    bool awakeScalingPassed = false;
    uint32_t awakeGrainsTested = 0;
    float simMsAwake = 0.0f;          // Multi-million awake grains stable execution
    uint32_t awakeNanCount = 0;
    size_t memoryAllocatedMB = 0;

    // Gate 11G-3: Kinematic moving collider plowing test
    bool kinematicColliderPassed = false;
    float kinematicPlowSpeed = 2.5f;   // 2.5 m/s plow sphere
    uint32_t plowedGrainsWoken = 0;    // Verified dynamic wake-up along plow trench
    bool nonTunnelingVerified = false; // Zero tunneling through moving obstacle
    uint32_t kinematicNanCount = 0;

    // Gate 11G-4: Scenario hot-switching and memory stability across all 7 presets
    bool scenarioSwitchingPassed = false;
    uint32_t scenariosVerified = 0;   // 7 scenarios: pile, column_collapse, funnel, dune, cliff, bonded_block, clumps
};

ScaleValidationResult runScaleValidationSuite(uint32_t capacity = 1000000);

} // namespace vparticles
