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
    static constexpr int kHistorySize = 180;
    float simMs[kHistorySize] = {0.0f};
    float gatherMs[kHistorySize] = {0.0f};
    float renderMs[kHistorySize] = {0.0f};
    float totalMs[kHistorySize] = {0.0f};
    float fps[kHistorySize] = {0.0f};
    int cursor = 0;
    int count = 0;

    void record(float sMs, float gMs, float rMs, float tMs, float currentFps) {
        simMs[cursor]    = sMs;
        gatherMs[cursor] = gMs;
        renderMs[cursor] = rMs;
        totalMs[cursor]  = tMs;
        fps[cursor]      = currentFps;
        cursor = (cursor + 1) % kHistorySize;
        if (count < kHistorySize) count++;
    }

    void reset() {
        for (int i = 0; i < kHistorySize; ++i) {
            simMs[i] = gatherMs[i] = renderMs[i] = totalMs[i] = fps[i] = 0.0f;
        }
        cursor = 0;
        count = 0;
    }
};

struct StudioState {
    // --- Selection State ---
    SelectedEntityType selectedType = SelectedEntityType::ColliderObstacleSphere;
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
    uint32_t maxFrames = 1200;
    int targetFpsMode = 3; // 0: 24, 1: 30, 2: 60, 3: 120, 4: Free/Unlocked
    bool isLooping = false; // Real-time continuous simulation by default
    int stepRequest = 0;   // +1 step forward, -1 step backward
    bool resetRequested = false;
    bool wakeGrainsRequested = false;
    float timeScale = 1.0f;

    // --- Viewport Display & Shading ---
    int sandColorMode = 0; // 0: Material Tint, 1: Velocity Heatmap, 2: Sleeping State, 3: Height Elevation
    float velocityRangeMax = 3.0f; // Max velocity in m/s for colormap
    float displayScale = 1.0f;     // Visual scale multiplier (independent of physical grain radius)
    float minPointSize = 1.5f;     // Screen clamp min pixels
    float maxPointSize = 64.0f;    // Screen clamp max pixels
    int renderStyle = 0;           // 0: Shaded Sphere, 1: Flat Disc, 2: Wire Ring

    bool showInfiniteGrid = true;
    bool showStatsOverlay = true;
    bool showColormapLegend = true;
    bool showGizmo = true;
    bool isFlyModeActive = false;
    bool isMaximizedViewport = false;

    // --- Visibility Filters ---
    bool showParticles = true;
    bool showColliders = true;
    bool showEmitters = true;
    char outlinerFilter[64] = "";

    // --- Active Scenario ---
    int currentScenarioIndex = 1; // 0: pile, 1: column_collapse, 2: funnel, 3: dune, 4: cliff, 5: bonded_block, 6: clumps
    bool scenarioSwitchRequested = false;

    // --- Live Granular Settings ---
    GranularSettings granularSettings = {};
    uint32_t capacity = 500000;
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
    bool exportBenchmarkRequested = false;

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
