#include "SimulationGui.h"

#include <iomanip>
#include <imgui.h>
#include <sstream>
#include <string>

SimulationGuiResult SimulationGui::Draw(
    SimulationSettings& settings,
    const PerformanceStats& stats) const
{
    SimulationGuiResult result;

    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(560.0f, 330.0f), ImGuiCond_Always);

    ImGui::Begin("Simulation Controls");

    ImGui::SetNextItemWidth(340.0f);
    result.velocityChanged |= ImGui::SliderFloat(
        "Velocity X",
        &settings.velocityX,
        -500.0f,
        500.0f
    );

    ImGui::SetNextItemWidth(340.0f);
    result.velocityChanged |= ImGui::SliderFloat(
        "Velocity Y",
        &settings.velocityY,
        -500.0f,
        500.0f
    );

    ImGui::Checkbox("Paused", &settings.paused);

    result.resetRequested = ImGui::Button("Reset");

    std::ostringstream fpsValue;
    fpsValue << std::fixed << std::setprecision(1) << stats.fps;

    std::ostringstream frameTimeValue;
    frameTimeValue << std::fixed << std::setprecision(2) << stats.frameTimeMs
        << " ms";

    ImGui::Separator();

    const std::string fpsValueText = fpsValue.str();
    const std::string frameTimeValueText = frameTimeValue.str();

    ImGui::TextUnformatted("FPS");
    ImGui::SameLine(240.0f);
    ImGui::TextUnformatted(fpsValueText.c_str());

    ImGui::TextUnformatted("Frame Time");
    ImGui::SameLine(240.0f);
    ImGui::TextUnformatted(frameTimeValueText.c_str());

    ImGui::End();

    return result;
}
