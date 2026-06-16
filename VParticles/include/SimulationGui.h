#pragma once
#include <vector>
#include <string>
#include <imgui.h>
#include "PerformanceStats.h"
#include "SimulationSettings.h"
#include "UndoSystem.h"

enum class PinType
{
    Flow,     // Emitter connection (Yellow)
    Forces,   // Physics forces connection (Green)
    Render,   // Render settings connection (Purple)
    Color,    // Color stops connection (Teal/Cyan)
    Velocity, // Velocity settings connection (Magenta)
    Size      // Size/Lifetime settings connection (Orange)
};

struct NodePin
{
    std::string name;
    PinType type;
    bool isOutput;
    ImVec2 screenPos; // Cached during drawing
};

struct NodeLink
{
    int fromNodeId;
    int toNodeId;
    int toPinIdx; // Inputs are ordered
};

struct Node
{
    int id;
    std::string name;
    ImVec2 pos;      // Canvas local coordinate
    ImVec2 size;     // Node boundary
    std::vector<NodePin> inputs;
    std::vector<NodePin> outputs;
};

struct SimulationGuiResult
{
    bool resetRequested = false;
    bool velocityChanged = false;
    bool particleCountChanged = false;
    bool switchToConsoleRequested = false;
    bool exitRequested = false;
    bool fullscreenToggleRequested = false;
};

class SimulationGui
{
public:
    SimulationGui();

    SimulationGuiResult Draw(
        SimulationSettings& settings,
        UndoSystem& undoSystem,
        const PerformanceStats& stats,
        const SimulationStats& simStats,
        class ParticleSystem& particleSystem,
        struct Camera& camera);

    float GetLeftPanelWidth() const { return m_showOutliner ? m_leftPanelWidth : 0.0f; }
    float GetRightPanelWidth() const { return m_showDetails ? m_rightPanelWidth : 0.0f; }
    float GetBottomPanelHeight() const { return (m_showNodeGraph || m_showDiagnostics) ? m_bottomPanelHeight : 0.0f; }

private:
    void RestoreDefaultLinks();

    bool m_wasAnyItemActive = false;
    SimulationSettings m_settingsBeforeEdit;

    // Selected Node: 0=Emitter, 1=Forces, 2=Config, 3=ColorGradient, 4=Renderer, 5=Velocity, 6=Size, 7=Camera
    int m_selectedNode = 0;

    // Resizable panel dimensions
    float m_leftPanelWidth = 300.0f;
    float m_rightPanelWidth = 350.0f;
    float m_bottomPanelHeight = 320.0f;
    float m_timelineWidth = 320.0f;

    // Node graph connection states
    bool m_connEmitter = true;
    bool m_connForces = true;
    bool m_connColor = true;
    bool m_connVelocity = true;
    bool m_connSize = true;
    bool m_prevConnEmitter = true;
    bool m_prevConnForces = true;
    bool m_prevConnColor = true;
    bool m_prevConnVelocity = true;
    bool m_prevConnSize = true;

    // Performance History
    float m_fpsHistory[100] = {0};
    float m_updateTimeHistory[100] = {0};
    float m_renderTimeHistory[100] = {0};
    int m_historyOffset = 0;

    // Layout visibility toggles
    bool m_showOutliner = true;
    bool m_showPresets = true;
    bool m_showNodeGraph = true;
    bool m_showDiagnostics = true;
    bool m_showDetails = true;
    bool m_showCameraSettings = false;

    // Canvas scrolling and zoom
    ImVec2 m_canvasScroll;
    float m_canvasZoom = 1.0f;

    // Active drag connection state
    bool m_dragLinkActive = false;
    int m_dragStartNodeId = -1;
    int m_dragStartPinIdx = -1;
    bool m_dragStartIsOutput = false;

    // Node systems
    std::vector<Node> m_nodes;
    std::vector<NodeLink> m_links;

    // Timeline duration limit & loop controls
    float m_timelineDuration = 5.0f; // in seconds
    bool m_loopTimeline = true;
};
