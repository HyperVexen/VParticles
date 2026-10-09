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

} // namespace vparticles
