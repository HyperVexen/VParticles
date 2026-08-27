#pragma once

#include "VParticles/ParticlePool.h"
#include "VParticles/SimulationTypes.h"

#include <cstdint>

namespace vparticles {

class ParticleSystem {
public:
    explicit ParticleSystem(uint32_t capacity);
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    uint32_t addEmitter(const EmitterDesc& desc);
    void queueBurst(uint32_t systemId, uint32_t count);
    void update(float dt);
    void reset();

    const ParticlePool& buffers() const;
    const SimulationStats& stats() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace vparticles
