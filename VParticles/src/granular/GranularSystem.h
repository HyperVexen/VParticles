#pragma once

#include "VParticles/GranularTypes.h"
#include "VParticles/ContactSolver.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace vparticles {

class GranularSystem {
public:
    explicit GranularSystem(uint32_t capacity, bool enableCudaGraphs = true);
    ~GranularSystem();

    GranularSystem(const GranularSystem&) = delete;
    GranularSystem& operator=(const GranularSystem&) = delete;

    void setSettings(const GranularSettings& settings);
    const GranularSettings& settings() const;

    const GranularPool& pool() const;
    const GranularStats& stats() const;

    void setContactSolver(std::unique_ptr<IContactSolver> solver);
    const IContactSolver* contactSolver() const;

    const NeighborData& neighborData() const;
    const NeighborStats& neighborStats() const;
    void rebuildNeighbors();

    void spawnLattice(
        const Float3& origin,
        uint32_t count,
        float spacing,
        float jitterRatio = 0.0f,
        const Float3& initialVel = {0.0f, 0.0f, 0.0f},
        uint32_t seed = 11011);

    void update(float dt);
    void reset();
    void synchronize();

private:
    void stepFixed(float fixedDt);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vparticles
