#pragma once

#include "VParticles/ParticlePool.h"
#include "VParticles/SimulationTypes.h"

#include <cstdint>
#include <vector>

namespace vparticles {

class ParticleSystem {
public:
    explicit ParticleSystem(
        uint32_t capacity,
        StorageMode mode = StorageMode::FP32,
        bool enableCudaGraphs = true);
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    uint32_t addEmitter(const EmitterDesc& desc);
    void clearEmitters();
    void setEmitter(uint32_t index, const EmitterDesc& desc);
    uint32_t emitterCount() const;
    void queueBurst(uint32_t systemId, uint32_t count);
    void update(float dt);
    void reset();

    void setSettings(const SimulationSettings& settings);
    const SimulationSettings& settings() const;
    StorageMode storageMode() const;

    uint32_t addRecipe(const EffectRecipe& recipe);
    void setRecipe(uint32_t recipeId, const EffectRecipe& recipe);
    const EffectRecipe& recipe(uint32_t recipeId) const;
    uint32_t recipeCount() const;

    // Explicit boundary for callers that need an exact host-visible snapshot.
    // Normal update() submission never waits for the GPU.
    void synchronize();

    const ParticlePool& buffers() const;
    const GpuParticlePool& gpuBuffers() const;
    const SimulationStats& stats() const;
    const std::vector<TileStats>& tileStats() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace vparticles
