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

} // namespace vparticles
