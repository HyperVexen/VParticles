#pragma once

#include "ParticleSystem.h"
#include "GpuRenderer.h"
#include "SimulationSettings.h"
#include <SFML/Graphics.hpp>
#include <vector>
#include <string>

class BenchmarkRunner
{
public:
    static void Run(sf::RenderWindow& window, ParticleSystem& particleSystem, GpuRenderer& gpuRenderer, SimulationSettings& settings);

private:
    static void RunTest(sf::RenderWindow& window, ParticleSystem& particleSystem, GpuRenderer& gpuRenderer, SimulationSettings& settings, std::size_t targetParticles, const std::string& modeName);
};
