#pragma once

#include "VParticles/GranularTypes.h"
#include "VParticles/NeighborTypes.h"
#include <cuda_runtime.h>
#include <memory>

namespace vparticles {

class GranularNeighborSearch {
public:
    explicit GranularNeighborSearch(uint32_t capacity);
    ~GranularNeighborSearch();

    GranularNeighborSearch(const GranularNeighborSearch&) = delete;
    GranularNeighborSearch& operator=(const GranularNeighborSearch&) = delete;

    // Full neighbor rebuild (keys -> sort -> bounds -> reorder -> verlet list -> color partitioning)
    void build(GranularPool& pool, const GranularSettings& settings, cudaStream_t stream);

    // Fast O(N) check: returns true if any grain moved > skin/2
    bool checkRebuildTrigger(const GranularPool& pool, const GranularSettings& settings, cudaStream_t stream);

    // Reset reference positions to current positions
    void resetReferencePositions(const GranularPool& pool, cudaStream_t stream);

    // Accessors
    const NeighborData& neighborData() const;
    const NeighborStats& stats() const;
    const GridParams& gridParams() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vparticles
