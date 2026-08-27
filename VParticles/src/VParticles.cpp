#include "VParticles/ParticleSystem.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

uint32_t parseUintArg(const char* value, uint32_t fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value ? static_cast<uint32_t>(parsed) : fallback;
}

float parseFloatArg(const char* value, float fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    return end != value ? parsed : fallback;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        uint32_t capacity = 1'000'000;
        uint32_t frames = 240;
        uint32_t emitterCount = 1;
        float spawnRate = 250'000.0f;
        float dt = 1.0f / 60.0f;

        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--capacity" && index + 1 < argc) {
                capacity = parseUintArg(argv[++index], capacity);
            } else if (arg == "--frames" && index + 1 < argc) {
                frames = parseUintArg(argv[++index], frames);
            } else if (arg == "--emitters" && index + 1 < argc) {
                emitterCount = std::max(1u, parseUintArg(argv[++index], emitterCount));
            } else if (arg == "--spawn-rate" && index + 1 < argc) {
                spawnRate = parseFloatArg(argv[++index], spawnRate);
            } else if (arg == "--dt" && index + 1 < argc) {
                dt = parseFloatArg(argv[++index], dt);
            }
        }

        vparticles::ParticleSystem system(capacity);
        for (uint32_t systemId = 0; systemId < emitterCount; ++systemId) {
            vparticles::EmitterDesc emitter;
            emitter.spawnRate = spawnRate / static_cast<float>(emitterCount);
            emitter.lifetime = 4.0f;
            emitter.lifetimeVariance = 0.2f;
            emitter.velocity = {0.0f, 8.0f, 0.0f};
            emitter.velocityVariance = {2.0f, 2.0f, 2.0f};
            system.addEmitter(emitter);
        }

        std::cout << "VParticles compute benchmark\n"
                  << "capacity=" << capacity
                  << " frames=" << frames
                  << " emitters=" << emitterCount
                  << " spawnRate=" << spawnRate
                  << " dt=" << dt << "\n\n";

        for (uint32_t frame = 0; frame < frames; ++frame) {
            system.update(dt);
            const vparticles::SimulationStats& stats = system.stats();

            if (frame % 30 == 0 || frame + 1 == frames) {
                std::cout << "frame " << std::setw(4) << frame
                          << " alive=" << std::setw(9) << stats.aliveCount
                          << " spawned=" << std::setw(7) << stats.spawned
                          << " dropped=" << std::setw(7) << stats.dropped
                          << " dead=" << std::setw(7) << stats.deadCount
                          << " spawnMs=" << std::fixed << std::setprecision(3) << stats.spawnMs
                          << " simMs=" << stats.simulateMs
                          << " compactMs=" << stats.compactMs
                          << " totalMs=" << stats.totalMs
                          << '\n';
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VParticles failed: " << error.what() << '\n';
        return 1;
    }
}
