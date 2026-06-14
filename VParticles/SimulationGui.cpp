#include "SimulationGui.h"

#include <imgui.h>

SimulationGuiResult SimulationGui::Draw(
    SimulationSettings& settings,
    const PerformanceStats& stats) const
{
    SimulationGuiResult result;

    ImGui::Begin("Simulation Controls");

    result.velocityChanged |= ImGui::SliderFloat(
        "Velocity X",
        &settings.velocityX,
        -500.0f,
        500.0f
    );

    result.velocityChanged |= ImGui::SliderFloat(
        "Velocity Y",
        &settings.velocityY,
        -500.0f,
        500.0f
    );

    ImGui::Checkbox("Paused", &settings.paused);

    result.resetRequested = ImGui::Button("Reset");

    ImGui::Separator();
    ImGui::Text("FPS: %.1f", stats.fps);
    ImGui::Text("Frame Time: %.2f ms", stats.frameTimeMs);

    ImGui::End();

    return result;
}
