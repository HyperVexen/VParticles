#pragma once

#include "VParticles/GranularTypes.h"
#include <cuda_runtime.h>
#include <memory>

namespace vparticles {

class GranularBondManager {
public:
    explicit GranularBondManager(uint32_t capacity);
    ~GranularBondManager();

    GranularBondManager(const GranularBondManager&) = delete;
    GranularBondManager& operator=(const GranularBondManager&) = delete;

    void reset();

    // Create lattice bonds connecting all adjacent grains within maxDist
    void createLatticeBonds(
        const GranularPool& pool,
        float maxDist,
        float compliance,
        float breakForce,
        cudaStream_t stream);

    // Create clump bonds connecting grains that belong to the same clump cluster
    void createClumpBonds(
        const GranularPool& pool,
        float clumpRadius,
        float maxDist,
        float compliance,
        float breakForce,
        uint32_t seed,
        cudaStream_t stream);

    // Reorder bond arrays when GranularPool is reordered by spatial sort
    void reorder(
        const uint32_t* sortedIndices,
        const uint32_t* sortedGrainId,
        uint32_t aliveCount,
        cudaStream_t stream);

    // Solve compliant XPBD distance constraints
    void solveBonds(
        GranularPool& pool,
        float substepDt,
        cudaStream_t stream);

    // Telemetry & metrics
    uint32_t getActiveBondCount(uint32_t aliveCount, cudaStream_t stream);
    uint32_t getBrokenBondCount(cudaStream_t stream);

    const GrainBond* bonds() const;
    const uint32_t* bondCounts() const;
    const uint32_t* grainIdToIndex() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vparticles
