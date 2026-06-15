#include "SimulationGui.h"

#include <iomanip>
#include <imgui.h>
#include <sstream>
#include <string>

static void ApplyPreset(SimulationSettings& settings, ArtistPreset preset)
{
    settings.activePreset = preset;
    // reset some defaults first
    settings.shape = EmitterShape::Point;
    settings.burstCount = 0;
    settings.spawnRate = 100.0f;
    settings.gravity = 0.0f;
    settings.drag = 0.0f;
    settings.windX = 0.0f;
    settings.windY = 0.0f;
    settings.velocityX = 0.0f;
    settings.velocityY = 0.0f;
    settings.velocityVarianceX = 0.0f;
    settings.velocityVarianceY = 0.0f;
    settings.sizeStart = 1.0f;
    settings.sizeEnd = 1.0f;
    settings.colorGradient = { sf::Color::White, sf::Color::White, sf::Color::White, sf::Color::White };
    
    switch (preset)
    {
    case ArtistPreset::Rain:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 100.0f;
        settings.emitterX = 640.0f;
        settings.emitterY = 0.0f;
        settings.spawnRate = 1000.0f;
        settings.gravity = 1500.0f;
        settings.drag = 0.1f;
        settings.baseSize = 2.0f;
        settings.velocityY = 500.0f;
        settings.particleLifetime = 1.5f;
        settings.colorGradient = { sf::Color(200, 200, 255, 150), sf::Color(200, 200, 255, 150), sf::Color(200, 200, 255, 150), sf::Color(200, 200, 255, 0) };
        break;
    case ArtistPreset::Snow:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 100.0f;
        settings.emitterX = 640.0f;
        settings.emitterY = 0.0f;
        settings.spawnRate = 300.0f;
        settings.gravity = 50.0f;
        settings.drag = 0.5f;
        settings.baseSize = 4.0f;
        settings.sizeRandomness = 0.5f;
        settings.windX = 100.0f;
        settings.velocityVarianceX = 50.0f;
        settings.particleLifetime = 5.0f;
        settings.colorGradient = { sf::Color::White, sf::Color::White, sf::Color::White, sf::Color(255, 255, 255, 0) };
        break;
    case ArtistPreset::Smoke:
        settings.shape = EmitterShape::Point;
        settings.emitterX = 640.0f;
        settings.emitterY = 600.0f;
        settings.spawnRate = 50.0f;
        settings.gravity = -100.0f; // rises
        settings.drag = 1.0f;
        settings.baseSize = 20.0f;
        settings.sizeStart = 0.5f;
        settings.sizeEnd = 5.0f;
        settings.particleLifetime = 4.0f;
        settings.velocityVarianceX = 30.0f;
        settings.velocityVarianceY = 30.0f;
        settings.colorGradient = { sf::Color(100, 100, 100, 200), sf::Color(80, 80, 80, 150), sf::Color(50, 50, 50, 50), sf::Color(0, 0, 0, 0) };
        break;
    case ArtistPreset::Dust:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 720.0f;
        settings.emitterX = 640.0f;
        settings.emitterY = 360.0f;
        settings.spawnRate = 200.0f;
        settings.gravity = 5.0f;
        settings.drag = 2.0f;
        settings.baseSize = 2.0f;
        settings.particleLifetime = 8.0f;
        settings.velocityVarianceX = 20.0f;
        settings.velocityVarianceY = 20.0f;
        settings.colorGradient = { sf::Color(200, 180, 150, 0), sf::Color(200, 180, 150, 100), sf::Color(200, 180, 150, 100), sf::Color(200, 180, 150, 0) };
        break;
    case ArtistPreset::Sparks:
        settings.shape = EmitterShape::Point;
        settings.emitterX = 640.0f;
        settings.emitterY = 600.0f;
        settings.burstCount = 500;
        settings.spawnRate = 0.0f;
        settings.gravity = 800.0f;
        settings.drag = 0.5f;
        settings.baseSize = 4.0f;
        settings.sizeEnd = 0.1f;
        settings.velocityY = -600.0f;
        settings.velocityVarianceX = 400.0f;
        settings.velocityVarianceY = 200.0f;
        settings.particleLifetime = 1.5f;
        settings.colorGradient = { sf::Color::White, sf::Color::Yellow, sf::Color::Red, sf::Color(50, 0, 0, 0) };
        break;
    case ArtistPreset::Custom:
    default:
        break;
    }
}

SimulationGuiResult SimulationGui::Draw(
    SimulationSettings& settings,
    const PerformanceStats& stats,
    const SimulationStats& simStats) const
{
    SimulationGuiResult result;

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(350.0f, 720.0f), ImGuiCond_Once);

    // Blender-like window padding
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 6.0f));

    ImGui::Begin("Properties", nullptr, ImGuiWindowFlags_NoCollapse);

    auto BeginPropertiesTable = [](const char* id) -> bool {
        return ImGui::BeginTable(id, 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoBordersInBody);
    };

    auto EndPropertiesTable = []() {
        ImGui::EndTable();
    };

    auto DrawPropertyLabel = [](const char* label) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
        ImGui::PushItemWidth(-1.0f); // Stretch to fill column
    };

    auto EndPropertyControl = []() {
        ImGui::PopItemWidth();
    };

    if (ImGui::Button("Switch to Console Bench", ImVec2(-1, 0)))
    {
        result.switchToConsoleRequested = true;
    }

    if (BeginPropertiesTable("GlobalControls"))
    {
        DrawPropertyLabel("Preset");
        const char* presetNames[] = { "Custom", "Rain", "Snow", "Smoke", "Dust", "Sparks" };
        int currentPreset = static_cast<int>(settings.activePreset);
        if (ImGui::Combo("##Preset", &currentPreset, presetNames, 6))
        {
            ApplyPreset(settings, static_cast<ArtistPreset>(currentPreset));
            result.resetRequested = true;
        }
        EndPropertyControl();

        DrawPropertyLabel("Action");
        if (ImGui::Button("Reset / Fire Burst", ImVec2(-1, 0)))
            result.resetRequested = true;
        EndPropertyControl();

        DrawPropertyLabel("Playback");
        ImGui::Checkbox("Paused", &settings.paused);
        EndPropertyControl();
        
        EndPropertiesTable();
    }

    if (ImGui::CollapsingHeader("Timeline & Determinism", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (BeginPropertiesTable("Timeline"))
        {
            DrawPropertyLabel("Start Time");
            ImGui::InputFloat("##Start Time", &settings.startTime);
            EndPropertyControl();

            DrawPropertyLabel("End Time");
            ImGui::InputFloat("##End Time", &settings.endTime);
            EndPropertyControl();

            DrawPropertyLabel("Random Seed");
            ImGui::InputScalar("##Random Seed", ImGuiDataType_U32, &settings.randomSeed);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Emission & Transform", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (BeginPropertiesTable("Emission"))
        {
            DrawPropertyLabel("Emitter X");
            ImGui::SliderFloat("##Emitter X", &settings.emitterX, 0.0f, 1280.0f);
            EndPropertyControl();

            DrawPropertyLabel("Emitter Y");
            ImGui::SliderFloat("##Emitter Y", &settings.emitterY, 0.0f, 720.0f);
            EndPropertyControl();
            
            DrawPropertyLabel("Shape");
            const char* shapeNames[] = { "Point", "Circle", "Box" };
            int currentShape = static_cast<int>(settings.shape);
            if (ImGui::Combo("##Shape", &currentShape, shapeNames, 3))
                settings.shape = static_cast<EmitterShape>(currentShape);
            EndPropertyControl();
                
            if (settings.shape == EmitterShape::Circle)
            {
                DrawPropertyLabel("Radius");
                ImGui::SliderFloat("##Radius", &settings.emitRadius, 0.0f, 500.0f);
                EndPropertyControl();
            }
            else if (settings.shape == EmitterShape::Box)
            {
                DrawPropertyLabel("Width");
                ImGui::SliderFloat("##Width", &settings.emitWidth, 0.0f, 1280.0f);
                EndPropertyControl();
                DrawPropertyLabel("Height");
                ImGui::SliderFloat("##Height", &settings.emitHeight, 0.0f, 1280.0f);
                EndPropertyControl();
            }

            DrawPropertyLabel("Spawn Rate");
            ImGui::InputFloat("##Spawn Rate", &settings.spawnRate);
            EndPropertyControl();

            DrawPropertyLabel("Burst Count");
            ImGui::InputInt("##Burst Count", &settings.burstCount);
            EndPropertyControl();

            DrawPropertyLabel("Max Particles");
            ImGui::InputInt("##Max Particles", &settings.particleCount);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Physics", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (BeginPropertiesTable("PhysicsTbl"))
        {
            DrawPropertyLabel("Gravity");
            ImGui::SliderFloat("##Gravity", &settings.gravity, -1000.0f, 2000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Wind X");
            ImGui::SliderFloat("##Wind X", &settings.windX, -1000.0f, 1000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Wind Y");
            ImGui::SliderFloat("##Wind Y", &settings.windY, -1000.0f, 1000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Drag");
            ImGui::SliderFloat("##Drag", &settings.drag, 0.0f, 10.0f);
            EndPropertyControl();

            DrawPropertyLabel("Base Mass");
            ImGui::SliderFloat("##Base Mass", &settings.baseMass, 0.1f, 10.0f);
            EndPropertyControl();

            DrawPropertyLabel("Mass Random");
            ImGui::SliderFloat("##Mass Random", &settings.massRandomness, 0.0f, 1.0f);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Velocity"))
    {
        if (BeginPropertiesTable("VelocityTbl"))
        {
            DrawPropertyLabel("Initial Vel X");
            ImGui::SliderFloat("##VelX", &settings.velocityX, -1000.0f, 1000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Initial Vel Y");
            ImGui::SliderFloat("##VelY", &settings.velocityY, -1000.0f, 1000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Variance X");
            ImGui::SliderFloat("##VarX", &settings.velocityVarianceX, 0.0f, 1000.0f);
            EndPropertyControl();

            DrawPropertyLabel("Variance Y");
            ImGui::SliderFloat("##VarY", &settings.velocityVarianceY, 0.0f, 1000.0f);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Lifetime & Size"))
    {
        if (BeginPropertiesTable("LifetimeTbl"))
        {
            DrawPropertyLabel("Lifetime");
            ImGui::SliderFloat("##Lifetime", &settings.particleLifetime, 0.1f, 20.0f);
            EndPropertyControl();

            DrawPropertyLabel("Life Random");
            ImGui::SliderFloat("##LifeRandom", &settings.lifetimeRandomness, 0.0f, 1.0f);
            EndPropertyControl();

            DrawPropertyLabel("Base Size");
            ImGui::SliderFloat("##Base Size", &settings.baseSize, 0.1f, 100.0f);
            EndPropertyControl();

            DrawPropertyLabel("Size Random");
            ImGui::SliderFloat("##SizeRandom", &settings.sizeRandomness, 0.0f, 1.0f);
            EndPropertyControl();

            DrawPropertyLabel("Start Mult");
            ImGui::SliderFloat("##StartMult", &settings.sizeStart, 0.0f, 10.0f);
            EndPropertyControl();

            DrawPropertyLabel("End Mult");
            ImGui::SliderFloat("##EndMult", &settings.sizeEnd, 0.0f, 10.0f);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Rotation"))
    {
        if (BeginPropertiesTable("RotationTbl"))
        {
            DrawPropertyLabel("Init Rot Min");
            ImGui::SliderFloat("##InitRotMin", &settings.initialRotationMin, -3.14f, 3.14f);
            EndPropertyControl();

            DrawPropertyLabel("Init Rot Max");
            ImGui::SliderFloat("##InitRotMax", &settings.initialRotationMax, -3.14f, 3.14f);
            EndPropertyControl();

            DrawPropertyLabel("Ang Vel Min");
            ImGui::SliderFloat("##AngVelMin", &settings.angularVelocityMin, -10.0f, 10.0f);
            EndPropertyControl();

            DrawPropertyLabel("Ang Vel Max");
            ImGui::SliderFloat("##AngVelMax", &settings.angularVelocityMax, -10.0f, 10.0f);
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    if (ImGui::CollapsingHeader("Color"))
    {
        if (BeginPropertiesTable("ColorTbl"))
        {
            DrawPropertyLabel("Color Random");
            ImGui::SliderFloat("##ColorRandom", &settings.colorRandomness, 0.0f, 1.0f);
            EndPropertyControl();
            
            float c0[4] = { settings.colorGradient[0].r/255.f, settings.colorGradient[0].g/255.f, settings.colorGradient[0].b/255.f, settings.colorGradient[0].a/255.f };
            DrawPropertyLabel("Stop 1");
            if (ImGui::ColorEdit4("##Stop1", c0, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                settings.colorGradient[0] = sf::Color(static_cast<std::uint8_t>(c0[0]*255), static_cast<std::uint8_t>(c0[1]*255), static_cast<std::uint8_t>(c0[2]*255), static_cast<std::uint8_t>(c0[3]*255));
            EndPropertyControl();
                
            float c1[4] = { settings.colorGradient[1].r/255.f, settings.colorGradient[1].g/255.f, settings.colorGradient[1].b/255.f, settings.colorGradient[1].a/255.f };
            DrawPropertyLabel("Stop 2");
            if (ImGui::ColorEdit4("##Stop2", c1, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                settings.colorGradient[1] = sf::Color(static_cast<std::uint8_t>(c1[0]*255), static_cast<std::uint8_t>(c1[1]*255), static_cast<std::uint8_t>(c1[2]*255), static_cast<std::uint8_t>(c1[3]*255));
            EndPropertyControl();

            float c2[4] = { settings.colorGradient[2].r/255.f, settings.colorGradient[2].g/255.f, settings.colorGradient[2].b/255.f, settings.colorGradient[2].a/255.f };
            DrawPropertyLabel("Stop 3");
            if (ImGui::ColorEdit4("##Stop3", c2, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                settings.colorGradient[2] = sf::Color(static_cast<std::uint8_t>(c2[0]*255), static_cast<std::uint8_t>(c2[1]*255), static_cast<std::uint8_t>(c2[2]*255), static_cast<std::uint8_t>(c2[3]*255));
            EndPropertyControl();

            float c3[4] = { settings.colorGradient[3].r/255.f, settings.colorGradient[3].g/255.f, settings.colorGradient[3].b/255.f, settings.colorGradient[3].a/255.f };
            DrawPropertyLabel("Stop 4");
            if (ImGui::ColorEdit4("##Stop4", c3, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar))
                settings.colorGradient[3] = sf::Color(static_cast<std::uint8_t>(c3[0]*255), static_cast<std::uint8_t>(c3[1]*255), static_cast<std::uint8_t>(c3[2]*255), static_cast<std::uint8_t>(c3[3]*255));
            EndPropertyControl();
            
            EndPropertiesTable();
        }
    }

    ImGui::Separator();
    
    if (ImGui::CollapsingHeader("Performance Stats", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (BeginPropertiesTable("StatsTbl"))
        {
            DrawPropertyLabel("Active Ptc");
            ImGui::Text("%u", simStats.activeParticles);
            EndPropertyControl();
            
            DrawPropertyLabel("Spawn/Kill (Fr)");
            ImGui::Text("+%u / -%u", simStats.spawnedThisFrame, simStats.killedThisFrame);
            EndPropertyControl();

            DrawPropertyLabel("Update Time");
            ImGui::Text("%.2f ms", simStats.updateTimeMs);
            EndPropertyControl();

            DrawPropertyLabel("Render Time");
            ImGui::Text("%.2f ms", simStats.renderTimeMs);
            EndPropertyControl();

            DrawPropertyLabel("FPS");
            ImGui::Text("%.1f", stats.fps);
            EndPropertyControl();

            DrawPropertyLabel("Average FPS");
            ImGui::Text("%.1f", stats.averageFps);
            EndPropertyControl();

            EndPropertiesTable();
        }
    }

    ImGui::End();
    ImGui::PopStyleVar(2);

    return result;
}
