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

    class GranularBondManager* bondManager() const;

    void spawnLattice(
        const Float3& origin,
        uint32_t count,
        float spacing,
        float jitterRatio = 0.0f,
        const Float3& initialVel = {0.0f, 0.0f, 0.0f},
        uint32_t seed = 11011,
        uint16_t matId = 0);

    void spawnColumnCollapse(
        const Float3& origin,
        uint32_t count,
        float spacing,
        float jitterRatio = 0.0f,
        uint32_t seed = 22022,
        uint16_t matId = 0);

    void spawnPouringPile(
        const Float3& origin,
        uint32_t count,
        float nozzleRadius = 0.08f,
        float height = 0.6f,
        float jitterRatio = 0.0f,
        uint32_t seed = 33033,
        uint16_t matId = 0);

    void spawnFunnel(
        const Float3& origin,
        uint32_t count,
        float topRadius = 0.25f,
        float orificeRadius = 0.04f,
        float height = 0.5f,
        float jitterRatio = 0.0f,
        uint32_t seed = 44044);

    void spawnDune(
        const Float3& origin,
        uint32_t count,
        float lengthX = 1.0f,
        float widthZ = 0.8f,
        float slopeAngleDeg = 25.0f,
        float jitterRatio = 0.0f,
        uint32_t seed = 55055);

    void spawnCliff(
        const Float3& origin,
        uint32_t count,
        float spacing = 0.02f,
        float jitterRatio = 0.0f,
        uint32_t seed = 66066);

    void spawnBondedBlock(
        const Float3& origin,
        uint32_t count,
        float spacing = 0.02f,
        float anchorRatio = 0.15f,
        float compliance = 1e-5f,
        float breakForce = 50.0f,
        uint32_t seed = 77077);

    void spawnClumps(
        const Float3& origin,
        uint32_t count,
        float spacing = 0.02f,
        float clumpRadius = 0.05f,
        float compliance = 1e-5f,
        float breakForce = 50.0f,
        float jitterRatio = 0.0f,
        uint32_t seed = 88088);

    void update(float dt);
    void reset();
    void synchronize();

private:
    void stepFixed(float fixedDt);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vparticles
