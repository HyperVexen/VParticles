#pragma once

#include "VParticles/GranularTypes.h"
#include "VParticles/SimulationTypes.h"
#include "Math3D.h"
#include <imgui.h>
#include <ImGuizmo.h>
#include <string>
#include <vector>

namespace vparticles::viewer {

enum class SelectedEntityType : uint8_t {
    None = 0,
    SimulationEngine,
    Emitter,
    ColliderGroundPlane,
    ColliderMovingPlow,
    ColliderObstacleSphere,
    ColliderContainerBox,
    MaterialProperties,
    EnvironmentForces
};

struct RollingMetrics {
    static constexpr int kHistorySize = 120;
    float simMs[kHistorySize] = {0.0f};
    float gatherMs[kHistorySize] = {0.0f};
    float renderMs[kHistorySize] = {0.0f};
    float totalMs[kHistorySize] = {0.0f};
    float fps[kHistorySize] = {0.0f};
    int cursor = 0;

    void record(float sMs, float gMs, float rMs, float tMs, float currentFps) {
        simMs[cursor]    = sMs;
        gatherMs[cursor] = gMs;
        renderMs[cursor] = rMs;
        totalMs[cursor]  = tMs;
        fps[cursor]      = currentFps;
        cursor = (cursor + 1) % kHistorySize;
    }
};

struct StudioState {
    // --- Selection State ---
    SelectedEntityType selectedType = SelectedEntityType::ColliderMovingPlow;
    int selectedIndex = 0;

    // --- Gizmo Configuration ---
    ImGuizmo::OPERATION currentGizmoOperation = ImGuizmo::TRANSLATE;
    ImGuizmo::MODE currentGizmoMode = ImGuizmo::WORLD;
    bool useSnap = false;
    float snapTranslation = 0.1f;
    float snapRotation = 15.0f;
    float snapScale = 0.1f;

    // --- Timeline & Transport State ---
    bool isPlaying = true;
    uint32_t currentFrame = 0;
    uint32_t maxFrames = 300;
    int targetFpsMode = 3; // 0: 24, 1: 30, 2: 60, 3: 120, 4: Free/Unlocked
    bool isLooping = true;
    int stepRequest = 0;   // +1 step forward, -1 step backward
    bool resetRequested = false;

    // --- Viewport Display & Shading ---
    int sandColorMode = 0; // 0: Material Tint, 1: Velocity Heatmap, 2: Sleeping State
    float pointScale = 1.0f;
    bool showInfiniteGrid = true;
    bool showStatsOverlay = true;
    bool showGizmo = true;
    bool isFlyModeActive = false;

    // --- Active Scenario ---
    int currentScenarioIndex = 1; // 0: pile, 1: column_collapse, 2: funnel, 3: dune, 4: cliff, 5: bonded_block, 6: clumps
    bool scenarioSwitchRequested = false;

    // --- Live Granular Settings ---
    GranularSettings granularSettings = {};
    uint32_t capacity = 250000;
    float fixedDt = 1.0f / 60.0f;

    // --- Emitter Configuration (Gizmo-bound) ---
    Float3 emitterPosition = {0.0f, 0.60f, 0.0f};
    Float3 emitterRotation = {0.0f, 0.0f, 0.0f};
    Float3 emitterScale = {1.0f, 1.0f, 1.0f};
    float emitterSpawnRate = 50000.0f;
    int emitterShapeType = 0; // 0: Point, 1: Sphere, 2: Box, 3: Cylinder
    bool emitterActive = true;

    // --- Rolling Performance Metrics ---
    RollingMetrics metrics = {};
    float lastSimTimeMs = 0.0f;
    float lastGatherTimeMs = 0.0f;
    float lastRenderTimeMs = 0.0f;
    float lastFrameTimeMs = 0.0f;
    float currentFps = 0.0f;

    // --- Live Simulation Telemetry ---
    uint32_t liveAliveGrains = 0;
    uint32_t liveAsleepGrains = 0;
    uint32_t liveActiveBonds = 0;
    uint32_t liveBrokenBonds = 0;
    float liveMeanHeight = 0.0f;
    float liveMaxHeight = 0.0f;
    float liveKineticEnergy = 0.0f;
    size_t liveVboMemoryBytes = 0;
};

} // namespace vparticles::viewer
