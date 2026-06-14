#include <SFML/Graphics.hpp>
#include <imgui-SFML.h>
#include <optional>

#include "ParticleRenderer.h"
#include "ParticleSystem.h"
#include "PerformanceStats.h"
#include "SimulationGui.h"
#include "SimulationSettings.h"

int main()
{
    sf::RenderWindow window(
        sf::VideoMode({ 1280, 720 }),
        "VParticles"
    );

    if (!ImGui::SFML::Init(window))
    {
        return 1;
    }

    SimulationSettings settings;
    PerformanceStats stats;

    ParticleSystem particleSystem;
    ParticleRenderer particleRenderer;
    SimulationGui simulationGui;

    particleSystem.Create(SimulationDefaults::ParticleCount, settings);

    sf::Clock clock;

    while (window.isOpen())
    {
        sf::Time frameTime = clock.restart();
        float dt = frameTime.asSeconds();

        stats.frameTimeMs = dt * 1000.0f;
        stats.fps = dt > 0.0f ? 1.0f / dt : 0.0f;

        while (const std::optional event = window.pollEvent())
        {
            ImGui::SFML::ProcessEvent(window, *event);

            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        ImGui::SFML::Update(window, frameTime);

        SimulationGuiResult guiResult = simulationGui.Draw(settings, stats);

        if (guiResult.resetRequested)
        {
            settings = SimulationSettings();
            particleSystem.Reset(SimulationDefaults::ParticleCount, settings);
        }
        else if (guiResult.velocityChanged)
        {
            particleSystem.SetVelocity(
                settings.velocityX,
                settings.velocityY
            );
        }

        auto size = window.getSize();

        if (!settings.paused)
        {
            particleSystem.Update(
                dt,
                static_cast<float>(size.x),
                static_cast<float>(size.y)
            );
        }

        window.clear();

        particleRenderer.Draw(window, particleSystem);
        ImGui::SFML::Render(window);

        window.display();
    }

    ImGui::SFML::Shutdown(window);

    return 0;
}
