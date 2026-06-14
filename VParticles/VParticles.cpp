#include <SFML/Graphics.hpp>
#include <imgui-SFML.h>
#include <optional>
#include <iostream>
#include <string>

#include "ParticleRenderer.h"
#include "ParticleSystem.h"
#include "PerformanceStats.h"
#include "SimulationGui.h"
#include "SimulationSettings.h"

void RunConsoleMode(ParticleSystem& particleSystem)
{
    sf::Clock clock;
    float timeAccumulator = 0.0f;
    int frameCount = 0;

    std::cout << "Starting console simulation mode..." << std::endl;

    while (true)
    {
        sf::Time frameTime = clock.restart();
        float dt = frameTime.asSeconds();

        particleSystem.Update(dt, 1280.0f, 720.0f);

        timeAccumulator += dt;
        frameCount++;

        if (timeAccumulator >= 1.0f)
        {
            float avgFps = static_cast<float>(frameCount) / timeAccumulator;
            std::cout << "Average FPS: " << avgFps 
                      << " | Particles: " << particleSystem.particles.size() 
                      << std::endl;
            timeAccumulator = 0.0f;
            frameCount = 0;
        }
    }
}

int main(int argc, char** argv)
{
    bool startInConsoleMode = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--console")
        {
            startInConsoleMode = true;
        }
    }

    SimulationSettings settings;
    PerformanceStats stats;

    ParticleSystem particleSystem;
    ParticleRenderer particleRenderer;
    SimulationGui simulationGui;

    particleSystem.Create(SimulationDefaults::ParticleCount, settings);

    if (startInConsoleMode)
    {
        RunConsoleMode(particleSystem);
        return 0;
    }
    sf::RenderWindow window(
        sf::VideoMode({ 1280, 720 }),
        "VParticles"
    );

    if (!ImGui::SFML::Init(window))
    {
        return 1;
    }

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;

    constexpr float uiFontSize = 22.0f;

    ImFontConfig fontConfig;
    fontConfig.SizePixels = uiFontSize;

    ImFont* uiFont = io.Fonts->AddFontFromFileTTF(
        "C:\\Windows\\Fonts\\segoeui.ttf",
        uiFontSize,
        &fontConfig,
        io.Fonts->GetGlyphRangesDefault()
    );

    if (uiFont == nullptr)
    {
        uiFont = io.Fonts->AddFontDefaultVector(&fontConfig);
    }

    ImFontBaked* bakedFont = uiFont->GetFontBaked(uiFontSize);
    const ImWchar* ranges = io.Fonts->GetGlyphRangesDefault();

    for (; ranges[0] != 0; ranges += 2)
    {
        for (unsigned int codepoint = ranges[0]; codepoint <= ranges[1]; ++codepoint)
        {
            bakedFont->FindGlyph(static_cast<ImWchar>(codepoint));
        }
    }

    uiFont->Flags |= ImFontFlags_LockBakedSizes;

    ImGuiStyle& style = ImGui::GetStyle();
    style.FontScaleMain = 1.0f;
    style.ScaleAllSizes(1.25f);
    style.Colors[ImGuiCol_Text] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.04f, 0.045f, 0.055f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.12f, 0.22f, 0.36f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.18f, 0.33f, 0.54f, 1.0f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.13f, 0.22f, 0.34f, 1.0f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.19f, 0.32f, 0.48f, 1.0f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.39f, 0.59f, 1.0f);

    // Removed local variables moved to top of main

    sf::Clock clock;
    float timeAccumulator = 0.0f;
    int frameCount = 0;

    while (window.isOpen())
    {
        sf::Time frameTime = clock.restart();
        float dt = frameTime.asSeconds();

        stats.frameTimeMs = dt * 1000.0f;
        stats.fps = dt > 0.0f ? 1.0f / dt : 0.0f;

        timeAccumulator += dt;
        frameCount++;
        if (timeAccumulator >= 0.5f) {
            stats.averageFps = static_cast<float>(frameCount) / timeAccumulator;
            timeAccumulator = 0.0f;
            frameCount = 0;
        }

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

        if (guiResult.switchToConsoleRequested)
        {
            window.close();
            RunConsoleMode(particleSystem);
            return 0;
        }

        if (guiResult.resetRequested)
        {
            settings = SimulationSettings();
            particleSystem.Reset(settings.particleCount, settings);
        }
        else 
        {
            if (guiResult.particleCountChanged)
            {
                particleSystem.Resize(settings.particleCount, settings);
            }
            if (guiResult.velocityChanged)
            {
                particleSystem.SetVelocity(
                    settings.velocityX,
                    settings.velocityY
                );
            }
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
