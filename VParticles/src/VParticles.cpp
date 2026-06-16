#include <SFML/Graphics.hpp>
#include <imgui-SFML.h>
#include <optional>
#include <iostream>
#include <string>

#include "GpuMonitor.h"
#include "GpuRenderer.h"
#include "Camera.h"

#include "ParticleSystem.h"
#include "PerformanceStats.h"
#include "SimulationGui.h"
#include "SimulationSettings.h"
#include "Benchmark.h"

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

        SimulationStats simStats;
        SimulationSettings settings;
        particleSystem.Update(dt, settings, simStats);

        timeAccumulator += dt;
        frameCount++;

        if (timeAccumulator >= 1.0f)
        {
            float avgFps = static_cast<float>(frameCount) / timeAccumulator;
            std::cout << "Average FPS: " << avgFps 
                      << " | Particles: " << particleSystem.GetActiveCount() 
                      << std::endl;
            timeAccumulator = 0.0f;
            frameCount = 0;
        }
    }
}

int main(int argc, char** argv)
{
    bool startInConsoleMode = false;
    bool runBenchmark = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--console")
        {
            startInConsoleMode = true;
        }
        else if (std::string(argv[i]) == "--benchmark")
        {
            runBenchmark = true;
        }
    }

    SimulationSettings settings;
    PerformanceStats stats;

    SimulationStats simStats;

    ParticleSystem particleSystem;

    GpuRenderer gpuRenderer;           // GPU-path renderer (CUDA-GL interop)
    SimulationGui simulationGui;
    GpuMonitor gpuMonitor;

    particleSystem.InitializePool(1000000);
    particleSystem.Reset(settings);

    if (startInConsoleMode)
    {
        RunConsoleMode(particleSystem);
        return 0;
    }
    // Request a depth buffer for 3D rendering
    sf::ContextSettings glSettings;
    glSettings.depthBits = 24;
    glSettings.majorVersion = 3;
    glSettings.minorVersion = 3;

    sf::RenderWindow window(
        sf::VideoMode({ 1280, 720 }),
        "VParticles",
        sf::Style::Default,
        sf::State::Windowed,
        glSettings
    );
    window.setVerticalSyncEnabled(false);   // Uncap FPS
    window.setFramerateLimit(0);            // No frame limit

    if (!ImGui::SFML::Init(window))
    {
        return 1;
    }

    // Init GPU renderer after window (OpenGL context) is created
    gpuRenderer.Init(window, 1000000);

    if (runBenchmark)
    {
        BenchmarkRunner::Run(window, particleSystem, gpuRenderer, settings);
        return 0;
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
    
    // Blender-style Rounding
    style.WindowRounding = 8.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    
    // Blender-style Colors
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_Border]                 = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.17f, 0.17f, 0.17f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.23f, 0.23f, 0.23f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.28f, 0.28f, 0.28f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_Separator]              = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);

    // Removed local variables moved to top of main

    // Camera state
    Camera camera;
    bool isDragging = false;
    sf::Vector2i lastMousePos;

    sf::Clock clock;
    float timeAccumulator = 0.0f;
    int frameCount = 0;
    float gpuPollAccumulator = 0.0f;

    // Initial GPU poll so stats aren't zero on first frame
    gpuMonitor.Poll(stats);

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

        // Poll GPU metrics once per second (NVML is not free)
        gpuPollAccumulator += dt;
        if (gpuPollAccumulator >= 1.0f) {
            gpuMonitor.Poll(stats);
            gpuPollAccumulator = 0.0f;
        }

        while (const std::optional event = window.pollEvent())
        {
            ImGui::SFML::ProcessEvent(window, *event);

            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
            else if (const auto* resized = event->getIf<sf::Event::Resized>())
            {
                sf::FloatRect visibleArea({0.f, 0.f}, {static_cast<float>(resized->size.x), static_cast<float>(resized->size.y)});
                window.setView(sf::View(visibleArea));
            }
            else if (const auto* pressed = event->getIf<sf::Event::MouseButtonPressed>())
            {
                // Only start dragging if middle mouse button and not over ImGui
                if (pressed->button == sf::Mouse::Button::Middle && !ImGui::GetIO().WantCaptureMouse)
                {
                    isDragging = true;
                    lastMousePos = pressed->position;
                }
            }
            else if (event->is<sf::Event::MouseButtonReleased>())
            {
                isDragging = false;
            }
            else if (const auto* moved = event->getIf<sf::Event::MouseMoved>())
            {
                if (isDragging)
                {
                    sf::Vector2i delta = moved->position - lastMousePos;
                    lastMousePos = moved->position;

                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::RShift))
                    {
                        camera.Pan(static_cast<float>(delta.x), static_cast<float>(delta.y));
                    }
                    else
                    {
                        camera.yaw   += static_cast<float>(delta.x) * 0.005f;
                        camera.pitch += static_cast<float>(delta.y) * 0.005f;
                        camera.ClampPitch();
                    }
                }
            }
            else if (const auto* scrolled = event->getIf<sf::Event::MouseWheelScrolled>())
            {
                if (!ImGui::GetIO().WantCaptureMouse)
                {
                    camera.distance -= scrolled->delta * 50.0f;
                    if (camera.distance < 10.0f) camera.distance = 10.0f;
                    if (camera.distance > 20000.0f) camera.distance = 20000.0f;
                }
            }
        }

        ImGui::SFML::Update(window, frameTime);

        SimulationGuiResult guiResult = simulationGui.Draw(settings, stats, simStats);

        if (guiResult.switchToConsoleRequested)
        {
            window.close();
            RunConsoleMode(particleSystem);
            return 0;
        }

        if (guiResult.resetRequested)
        {
            particleSystem.Reset(settings);
        }

        sf::Clock updateClock;
        if (!settings.paused)
        {
            particleSystem.Update(dt, settings, simStats);
        }
        simStats.updateTimeMs = updateClock.getElapsedTime().asSeconds() * 1000.0f;

        window.clear();

        sf::Clock renderClock;
        // GPU mode: CUDA kernel fills VBO, OpenGL instanced draw — zero CPU readback
        if (gpuRenderer.IsInitialized())
        {
            float aspect = static_cast<float>(window.getSize().x) / static_cast<float>(window.getSize().y);
            Mat4 viewMat = camera.GetViewMatrix();
            Mat4 projMat = camera.GetProjectionMatrix(aspect);
            gpuRenderer.Draw(window, particleSystem, viewMat.Ptr(), projMat.Ptr(), settings);
        }
        simStats.renderTimeMs = renderClock.getElapsedTime().asSeconds() * 1000.0f;
        ImGui::SFML::Render(window);

        window.display();
    }

    ImGui::SFML::Shutdown(window);

    return 0;
}
