#include "SimulationGui.h"
#include "ParticleSystem.h"
#include "Camera.h"

#include <iomanip>
#include <imgui.h>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

SimulationGui::SimulationGui()
    : m_canvasScroll(0.0f, 0.0f)
{
    // Node 0: Emitter Node
    Node emitter;
    emitter.id = 0;
    emitter.name = "Emitter Node";
    emitter.pos = ImVec2(40.0f, 40.0f);
    emitter.size = ImVec2(200.0f, 95.0f);
    emitter.outputs.push_back({ "Flow Out", PinType::Flow, true, ImVec2(0,0) });
    m_nodes.push_back(emitter);

    // Node 1: Forces Node
    Node forces;
    forces.id = 1;
    forces.name = "Forces Node";
    forces.pos = ImVec2(40.0f, 180.0f);
    forces.size = ImVec2(200.0f, 115.0f);
    forces.outputs.push_back({ "Forces Out", PinType::Forces, true, ImVec2(0,0) });
    m_nodes.push_back(forces);

    // Node 2: Config Node
    Node config;
    config.id = 2;
    config.name = "Config Node";
    config.pos = ImVec2(340.0f, 110.0f);
    config.size = ImVec2(200.0f, 115.0f);
    config.inputs.push_back({ "Flow In", PinType::Flow, false, ImVec2(0,0) });
    config.inputs.push_back({ "Forces In", PinType::Forces, false, ImVec2(0,0) });
    config.inputs.push_back({ "Velocity In", PinType::Velocity, false, ImVec2(0,0) });
    config.outputs.push_back({ "Render Out", PinType::Render, true, ImVec2(0,0) });
    m_nodes.push_back(config);

    // Node 3: Color Ramp
    Node colorRamp;
    colorRamp.id = 3;
    colorRamp.name = "Color Ramp";
    colorRamp.pos = ImVec2(40.0f, 460.0f);
    colorRamp.size = ImVec2(200.0f, 105.0f);
    colorRamp.outputs.push_back({ "Color Out", PinType::Color, true, ImVec2(0,0) });
    m_nodes.push_back(colorRamp);

    // Node 4: Renderer Node
    Node renderer;
    renderer.id = 4;
    renderer.name = "Renderer Node";
    renderer.pos = ImVec2(640.0f, 220.0f);
    renderer.size = ImVec2(200.0f, 95.0f);
    renderer.inputs.push_back({ "Render In", PinType::Render, false, ImVec2(0,0) });
    renderer.inputs.push_back({ "Color In", PinType::Color, false, ImVec2(0,0) });
    renderer.inputs.push_back({ "Size In", PinType::Size, false, ImVec2(0,0) });
    m_nodes.push_back(renderer);

    // Node 5: Velocity Node
    Node velocity;
    velocity.id = 5;
    velocity.name = "Velocity Node";
    velocity.pos = ImVec2(40.0f, 320.0f);
    velocity.size = ImVec2(200.0f, 95.0f);
    velocity.outputs.push_back({ "Velocity Out", PinType::Velocity, true, ImVec2(0,0) });
    m_nodes.push_back(velocity);

    // Node 6: Size/Lifetime Node
    Node sizeNode;
    sizeNode.id = 6;
    sizeNode.name = "Size & Life Node";
    sizeNode.pos = ImVec2(340.0f, 460.0f);
    sizeNode.size = ImVec2(200.0f, 95.0f);
    sizeNode.outputs.push_back({ "Size Out", PinType::Size, true, ImVec2(0,0) });
    m_nodes.push_back(sizeNode);

    // Default links (all active by default)
    m_links.push_back({ 0, 2, 0 }); // Emitter -> Config Flow In
    m_links.push_back({ 1, 2, 1 }); // Forces -> Config Forces In
    m_links.push_back({ 5, 2, 2 }); // Velocity -> Config Velocity In
    m_links.push_back({ 2, 4, 0 }); // Config -> Renderer Render In
    m_links.push_back({ 3, 4, 1 }); // Color -> Renderer Color In
    m_links.push_back({ 6, 4, 2 }); // Size -> Renderer Size In
}

void SimulationGui::RestoreDefaultLinks()
{
    m_links.clear();
    m_links.push_back({ 0, 2, 0 });
    m_links.push_back({ 1, 2, 1 });
    m_links.push_back({ 5, 2, 2 });
    m_links.push_back({ 2, 4, 0 });
    m_links.push_back({ 3, 4, 1 });
    m_links.push_back({ 6, 4, 2 });
}

// Static backup variables for node bypassing logic
static float backupSpawnRate = 100.0f;
static int backupBurstCount = 0;
static float backupGravity = 0.0f;
static float backupWindX = 0.0f;
static float backupWindY = 0.0f;
static float backupWindZ = 0.0f;
static float backupDrag = 0.0f;
static float backupVelocityX = 0.0f;
static float backupVelocityY = 0.0f;
static float backupVelocityZ = 0.0f;
static float backupVelocityVarX = 0.0f;
static float backupVelocityVarY = 0.0f;
static float backupVelocityVarZ = 0.0f;
static float backupBaseSize = 4.0f;
static float backupSizeStart = 1.0f;
static float backupSizeEnd = 1.0f;
static CudaColor backupGradient[4] = {
    CudaColor(255, 255, 255, 255),
    CudaColor(255, 255, 255, 255),
    CudaColor(255, 255, 255, 255),
    CudaColor(255, 255, 255, 255)
};

static void SetGradient(SimulationSettings& settings, CudaColor c0, CudaColor c1, CudaColor c2, CudaColor c3)
{
    settings.colorGradient[0] = c0;
    settings.colorGradient[1] = c1;
    settings.colorGradient[2] = c2;
    settings.colorGradient[3] = c3;
}

static void ApplyPreset(SimulationSettings& settings, ArtistPreset preset)
{
    settings.activePreset = preset;
    // reset some defaults first
    settings.shape = EmitterShape::Point;
    settings.emissionMode = EmissionMode::Volume;
    settings.emitterX = 640.0f;
    settings.emitterY = 360.0f;
    settings.emitterZ = 0.0f;
    settings.emitRadius = 50.0f;
    settings.emitWidth = 100.0f;
    settings.emitHeight = 100.0f;
    settings.emitDepth = 100.0f;
    settings.emitCubeSize = 100.0f;
    settings.gridColumns = 10;
    settings.gridRows = 10;
    settings.gridSlices = 1;
    settings.gridSpacingX = 20.0f;
    settings.gridSpacingY = 20.0f;
    settings.gridSpacingZ = 20.0f;
    settings.showVisualGrid = true;
    
    settings.burstCount = 0;
    settings.spawnRate = 100.0f;
    settings.gravity = 0.0f;
    settings.drag = 0.0f;
    settings.windX = 0.0f;
    settings.windY = 0.0f;
    settings.windZ = 0.0f;
    settings.velocityX = 0.0f;
    settings.velocityY = 0.0f;
    settings.velocityZ = 0.0f;
    settings.velocityVarianceX = 0.0f;
    settings.velocityVarianceY = 0.0f;
    settings.velocityVarianceZ = 0.0f;
    settings.sizeStart = 1.0f;
    settings.sizeEnd = 1.0f;
    SetGradient(settings, CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 255));
    
    switch (preset)
    {
    case ArtistPreset::Rain:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 100.0f;
        settings.spawnRate = 1000.0f;
        settings.gravity = 1500.0f;
        settings.drag = 0.1f;
        settings.baseSize = 2.0f;
        settings.velocityY = 500.0f;
        settings.particleLifetime = 1.5f;
        SetGradient(settings, CudaColor(200, 200, 255, 150), CudaColor(200, 200, 255, 150), CudaColor(200, 200, 255, 150), CudaColor(200, 200, 255, 0));
        break;
    case ArtistPreset::Snow:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 100.0f;
        settings.spawnRate = 300.0f;
        settings.gravity = 50.0f;
        settings.drag = 0.5f;
        settings.baseSize = 4.0f;
        settings.sizeRandomness = 0.5f;
        settings.windX = 100.0f;
        settings.velocityVarianceX = 50.0f;
        settings.particleLifetime = 5.0f;
        SetGradient(settings, CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 255), CudaColor(255, 255, 255, 0));
        break;
    case ArtistPreset::Smoke:
        settings.shape = EmitterShape::Point;
        settings.spawnRate = 50.0f;
        settings.gravity = -100.0f; // rises
        settings.drag = 1.0f;
        settings.baseSize = 20.0f;
        settings.sizeStart = 0.5f;
        settings.sizeEnd = 5.0f;
        settings.particleLifetime = 4.0f;
        settings.velocityVarianceX = 30.0f;
        settings.velocityVarianceY = 30.0f;
        SetGradient(settings, CudaColor(100, 100, 100, 200), CudaColor(80, 80, 80, 150), CudaColor(50, 50, 50, 50), CudaColor(0, 0, 0, 0));
        break;
    case ArtistPreset::Dust:
        settings.shape = EmitterShape::Box;
        settings.emitWidth = 1280.0f;
        settings.emitHeight = 720.0f;
        settings.spawnRate = 200.0f;
        settings.gravity = 5.0f;
        settings.drag = 2.0f;
        settings.baseSize = 2.0f;
        settings.particleLifetime = 8.0f;
        settings.velocityVarianceX = 20.0f;
        settings.velocityVarianceY = 20.0f;
        SetGradient(settings, CudaColor(200, 180, 150, 0), CudaColor(200, 180, 150, 100), CudaColor(200, 180, 150, 100), CudaColor(200, 180, 150, 0));
        break;
    case ArtistPreset::Sparks:
        settings.shape = EmitterShape::Point;
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
        SetGradient(settings, CudaColor(255, 255, 255, 255), CudaColor(255, 255, 0, 255), CudaColor(255, 0, 0, 255), CudaColor(50, 0, 0, 0));
        break;
    case ArtistPreset::Custom:
    default:
        break;
    }
}

SimulationGuiResult SimulationGui::Draw(
    SimulationSettings& settings,
    UndoSystem& undoSystem,
    const PerformanceStats& stats,
    const SimulationStats& simStats,
    ParticleSystem& particleSystem,
    Camera& camera)
{
    SimulationSettings startOfFrameSettings = settings;
    SimulationGuiResult result;

    // Map visual links to connection toggle flags
    m_connEmitter = false;
    m_connForces = false;
    m_connColor = false;
    m_connVelocity = false;
    m_connSize = false;
    for (const auto& link : m_links)
    {
        if (link.fromNodeId == 0 && link.toNodeId == 2 && link.toPinIdx == 0) m_connEmitter = true;
        if (link.fromNodeId == 1 && link.toNodeId == 2 && link.toPinIdx == 1) m_connForces = true;
        if (link.fromNodeId == 5 && link.toNodeId == 2 && link.toPinIdx == 2) m_connVelocity = true;
        if (link.fromNodeId == 3 && link.toNodeId == 4 && link.toPinIdx == 1) m_connColor = true;
        if (link.fromNodeId == 6 && link.toNodeId == 4 && link.toPinIdx == 2) m_connSize = true;
    }

    // ── 1. Backup or Apply Node Graph Connection Overrides ──
    // Handle Emitter transition
    if (m_connEmitter && !m_prevConnEmitter)
    {
        settings.spawnRate = backupSpawnRate;
        settings.burstCount = backupBurstCount;
    }
    else if (m_connEmitter)
    {
        backupSpawnRate = settings.spawnRate;
        backupBurstCount = settings.burstCount;
    }
    else
    {
        settings.spawnRate = 0.0f;
        settings.burstCount = 0;
    }

    // Handle Forces transition
    if (m_connForces && !m_prevConnForces)
    {
        settings.gravity = backupGravity;
        settings.windX = backupWindX;
        settings.windY = backupWindY;
        settings.windZ = backupWindZ;
        settings.drag = backupDrag;
    }
    else if (m_connForces)
    {
        backupGravity = settings.gravity;
        backupWindX = settings.windX;
        backupWindY = settings.windY;
        backupWindZ = settings.windZ;
        backupDrag = settings.drag;
    }
    else
    {
        settings.gravity = 0.0f;
        settings.windX = 0.0f;
        settings.windY = 0.0f;
        settings.windZ = 0.0f;
        settings.drag = 0.0f;
    }

    // Handle Color transition
    if (m_connColor && !m_prevConnColor)
    {
        for (int i = 0; i < 4; ++i) settings.colorGradient[i] = backupGradient[i];
    }
    else if (m_connColor)
    {
        for (int i = 0; i < 4; ++i) backupGradient[i] = settings.colorGradient[i];
    }
    else
    {
        for (int i = 0; i < 4; ++i) settings.colorGradient[i] = CudaColor(255, 255, 255, 255); // Solid White
    }

    // Handle Velocity transition
    if (m_connVelocity && !m_prevConnVelocity)
    {
        settings.velocityX = backupVelocityX;
        settings.velocityY = backupVelocityY;
        settings.velocityZ = backupVelocityZ;
        settings.velocityVarianceX = backupVelocityVarX;
        settings.velocityVarianceY = backupVelocityVarY;
        settings.velocityVarianceZ = backupVelocityVarZ;
    }
    else if (m_connVelocity)
    {
        backupVelocityX = settings.velocityX;
        backupVelocityY = settings.velocityY;
        backupVelocityZ = settings.velocityZ;
        backupVelocityVarX = settings.velocityVarianceX;
        backupVelocityVarY = settings.velocityVarianceY;
        backupVelocityVarZ = settings.velocityVarianceZ;
    }
    else
    {
        settings.velocityX = 0.0f;
        settings.velocityY = 0.0f;
        settings.velocityZ = 0.0f;
        settings.velocityVarianceX = 0.0f;
        settings.velocityVarianceY = 0.0f;
        settings.velocityVarianceZ = 0.0f;
    }

    // Handle Size transition
    if (m_connSize && !m_prevConnSize)
    {
        settings.baseSize = backupBaseSize;
        settings.sizeStart = backupSizeStart;
        settings.sizeEnd = backupSizeEnd;
    }
    else if (m_connSize)
    {
        backupBaseSize = settings.baseSize;
        backupSizeStart = settings.sizeStart;
        backupSizeEnd = settings.sizeEnd;
    }
    else
    {
        settings.baseSize = 4.0f;
        settings.sizeStart = 1.0f;
        settings.sizeEnd = 1.0f;
    }

    // ── Update Rolling Buffers for Diagnostics ──
    m_fpsHistory[m_historyOffset] = stats.fps;
    m_updateTimeHistory[m_historyOffset] = simStats.updateTimeMs;
    m_renderTimeHistory[m_historyOffset] = simStats.renderTimeMs;
    m_historyOffset = (m_historyOffset + 1) % 100;

    ImVec2 displaySize = ImGui::GetIO().DisplaySize;
    float menuBarHeight = 24.0f;

    // Calculate effective panel layout dimensions dynamically
    float leftWidth = m_showOutliner ? m_leftPanelWidth : 0.0f;
    float rightWidth = m_showDetails ? m_rightPanelWidth : 0.0f;
    float bottomHeight = (m_showNodeGraph || m_showDiagnostics) ? m_bottomPanelHeight : 0.0f;
    float timelineWidth = m_showDiagnostics ? (m_showNodeGraph ? m_timelineWidth : (displaySize.x - leftWidth - rightWidth)) : 0.0f;

    // ── 2. Top Menu Bar ──
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Reset / Fire Burst", "R"))
            {
                result.resetRequested = true;
            }
            if (ImGui::MenuItem("Switch to Console Bench"))
            {
                result.switchToConsoleRequested = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4"))
            {
                result.exitRequested = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "Ctrl+Z", nullptr, undoSystem.CanUndo()))
            {
                undoSystem.Undo(settings);
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", nullptr, undoSystem.CanRedo()))
            {
                undoSystem.Redo(settings);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Workspace Explorer", nullptr, &m_showOutliner);
            ImGui::MenuItem("Node Graph Editor", nullptr, &m_showNodeGraph);
            ImGui::MenuItem("Details Inspector", nullptr, &m_showDetails);
            ImGui::MenuItem("Timeline Panel", nullptr, &m_showDiagnostics);
            ImGui::Separator();
            if (ImGui::MenuItem("Toggle Fullscreen", "F11"))
            {
                result.fullscreenToggleRequested = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Artist Presets"))
        {
            const char* presetNames[] = { "Custom", "Rain", "Snow", "Smoke", "Dust", "Sparks" };
            for (int i = 0; i < 6; ++i)
            {
                if (ImGui::MenuItem(presetNames[i], nullptr, settings.activePreset == static_cast<ArtistPreset>(i)))
                {
                    undoSystem.PushUndo(settings);
                    ApplyPreset(settings, static_cast<ArtistPreset>(i));
                    RestoreDefaultLinks();
                    result.resetRequested = true;
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    // Styles for docked layout windows
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4.0f, 6.0f));

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
        ImGui::PushItemWidth(-1.0f);
    };

    auto EndPropertyControl = []() {
        ImGui::PopItemWidth();
    };

    // ── 3. Left Panel: Outliner & Content Browser presets (Resizable Width) ──
    if (m_showOutliner)
    {
        ImGui::SetNextWindowPos(ImVec2(0.0f, menuBarHeight));
        ImGui::SetNextWindowSize(ImVec2(m_leftPanelWidth, displaySize.y - menuBarHeight));
        
        ImGui::Begin("Workspace Explorer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::TextDisabled("WORKSPACE OUTLINER");
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::TreeNodeEx("ParticleSystem", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Selectable("Emitter Component", m_selectedNode == 0)) m_selectedNode = 0;
            if (ImGui::Selectable("Physics Forces", m_selectedNode == 1)) m_selectedNode = 1;
            if (ImGui::Selectable("Particle Config", m_selectedNode == 2)) m_selectedNode = 2;
            if (ImGui::Selectable("Color Ramp stops", m_selectedNode == 3)) m_selectedNode = 3;
            if (ImGui::Selectable("Instanced Renderer", m_selectedNode == 4)) m_selectedNode = 4;
            if (ImGui::Selectable("Initial Velocity", m_selectedNode == 5)) m_selectedNode = 5;
            if (ImGui::Selectable("Size & Lifetime", m_selectedNode == 6)) m_selectedNode = 6;
            if (ImGui::Selectable("Camera Target", m_selectedNode == 7)) m_selectedNode = 7;
            
            ImGui::TreePop();
        }

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextDisabled("PRESETS CONTENT BROWSER");
        ImGui::Separator();
        ImGui::Spacing();

        const char* presetNames[] = { "Custom Preset", "✨ Rain Storm", "❄️ Gentle Snow", "🔥 Volumetric Smoke", "🏜️ Sand Dust", "⚡ Electrical Sparks" };
        const char* presetDescriptions[] = {
            "Custom settings.",
            "Downward gravity, blue rain particles.",
            "Slow wind-swept white snow.",
            "Rising dark grey smoke gradient.",
            "Wide boundary box sand particles.",
            "High velocity burst spark particles."
        };

        for (int i = 0; i < 6; ++i)
        {
            ImGui::PushID(i);
            bool isCurrent = (settings.activePreset == static_cast<ArtistPreset>(i));
            
            if (isCurrent)
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 0.8f, 1.0f, 1.0f));
            
            ImGui::BeginChild(presetNames[i], ImVec2(-1.0f, 58.0f), ImGuiChildFlags_Borders);
            ImGui::Text("%s", presetNames[i]);
            ImGui::TextDisabled("%s", presetDescriptions[i]);
            
            ImGui::EndChild();
            
            if (isCurrent)
                ImGui::PopStyleColor();
            
            // Drag and drop source
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
            {
                ImGui::SetDragDropPayload("PRESET_DRAG", &i, sizeof(int));
                ImGui::Text("Apply %s", presetNames[i]);
                ImGui::EndDragDropSource();
            }

            // Click to apply
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
            {
                undoSystem.PushUndo(settings);
                ApplyPreset(settings, static_cast<ArtistPreset>(i));
                RestoreDefaultLinks();
                result.resetRequested = true;
            }
            ImGui::PopID();
        }

        ImGui::End();
    }

    // ── 4. Right Panel: Details Inspector & Diagnostics (Resizable Width) ──
    if (m_showDetails)
    {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - m_rightPanelWidth, menuBarHeight));
        ImGui::SetNextWindowSize(ImVec2(m_rightPanelWidth, displaySize.y - menuBarHeight));
        
        ImGui::Begin("Details Inspector", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::TextDisabled("DETAILS INSPECTOR");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::BeginChild("DetailsScrollRegion", ImVec2(-1.0f, displaySize.y - menuBarHeight - 240.0f), ImGuiChildFlags_None);

        if (m_selectedNode == 0) // Emitter
        {
            ImGui::Text("📐 Emitter Properties");
            ImGui::Spacing();
            if (BeginPropertiesTable("EmitterDetails"))
            {
                DrawPropertyLabel("Position X");
                ImGui::SliderFloat("##Emitter X", &settings.emitterX, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Position Y");
                ImGui::SliderFloat("##Emitter Y", &settings.emitterY, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Position Z");
                ImGui::SliderFloat("##Emitter Z", &settings.emitterZ, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Emitter Shape");
                const char* shapeNames[] = { "Point", "Circle", "Box", "Sphere", "Cube", "Grid" };
                int currentShape = static_cast<int>(settings.shape);
                if (ImGui::Combo("##Shape", &currentShape, shapeNames, 6))
                    settings.shape = static_cast<EmitterShape>(currentShape);
                EndPropertyControl();

                if (settings.shape != EmitterShape::Point && settings.shape != EmitterShape::Grid)
                {
                    DrawPropertyLabel("Emission Mode");
                    const char* modeNames[] = { "Volume (Inside)", "Surface (Boundary)" };
                    int currentMode = static_cast<int>(settings.emissionMode);
                    if (ImGui::Combo("##EmissionMode", &currentMode, modeNames, 2))
                        settings.emissionMode = static_cast<EmissionMode>(currentMode);
                    EndPropertyControl();
                }

                if (settings.shape == EmitterShape::Circle || settings.shape == EmitterShape::Sphere)
                {
                    DrawPropertyLabel("Radius");
                    ImGui::SliderFloat("##Radius", &settings.emitRadius, 0.0f, 500.0f);
                    EndPropertyControl();
                }
                else if (settings.shape == EmitterShape::Box)
                {
                    DrawPropertyLabel("Width");
                    ImGui::SliderFloat("##Width", &settings.emitWidth, 0.0f, 2000.0f);
                    EndPropertyControl();
                    DrawPropertyLabel("Height");
                    ImGui::SliderFloat("##Height", &settings.emitHeight, 0.0f, 2000.0f);
                    EndPropertyControl();
                    DrawPropertyLabel("Depth");
                    ImGui::SliderFloat("##Depth", &settings.emitDepth, 0.0f, 2000.0f);
                    EndPropertyControl();
                }
                else if (settings.shape == EmitterShape::Cube)
                {
                    DrawPropertyLabel("Edge Size");
                    ImGui::SliderFloat("##CubeSize", &settings.emitCubeSize, 0.0f, 2000.0f);
                    EndPropertyControl();
                }
                else if (settings.shape == EmitterShape::Grid)
                {
                    DrawPropertyLabel("Grid Columns");
                    ImGui::SliderInt("##GridCols", &settings.gridColumns, 1, 100);
                    EndPropertyControl();
                    DrawPropertyLabel("Grid Rows");
                    ImGui::SliderInt("##GridRows", &settings.gridRows, 1, 100);
                    EndPropertyControl();
                    DrawPropertyLabel("Grid Slices");
                    ImGui::SliderInt("##GridSlices", &settings.gridSlices, 1, 100);
                    EndPropertyControl();

                    DrawPropertyLabel("Spacing X");
                    ImGui::SliderFloat("##SpacingX", &settings.gridSpacingX, 0.1f, 200.0f);
                    EndPropertyControl();
                    DrawPropertyLabel("Spacing Y");
                    ImGui::SliderFloat("##SpacingY", &settings.gridSpacingY, 0.1f, 200.0f);
                    EndPropertyControl();
                    DrawPropertyLabel("Spacing Z");
                    ImGui::SliderFloat("##SpacingZ", &settings.gridSpacingZ, 0.1f, 200.0f);
                    EndPropertyControl();
                }

                DrawPropertyLabel("Spawn Rate");
                ImGui::SliderFloat("##Spawn Rate", m_connEmitter ? &settings.spawnRate : &backupSpawnRate, 0.0f, 5000.0f, "%.0f");
                EndPropertyControl();

                DrawPropertyLabel("Burst Count");
                ImGui::InputInt("##Burst Count", m_connEmitter ? &settings.burstCount : &backupBurstCount);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 1) // Physics
        {
            ImGui::Text("Physics Forces");
            ImGui::Spacing();
            if (BeginPropertiesTable("PhysicsDetails"))
            {
                DrawPropertyLabel("Gravity");
                ImGui::SliderFloat("##Gravity", m_connForces ? &settings.gravity : &backupGravity, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Wind X");
                ImGui::SliderFloat("##Wind X", m_connForces ? &settings.windX : &backupWindX, -1000.0f, 1000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Wind Y");
                ImGui::SliderFloat("##Wind Y", m_connForces ? &settings.windY : &backupWindY, -1000.0f, 1000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Wind Z");
                ImGui::SliderFloat("##Wind Z", m_connForces ? &settings.windZ : &backupWindZ, -1000.0f, 1000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Drag coefficient");
                ImGui::SliderFloat("##Drag", m_connForces ? &settings.drag : &backupDrag, 0.0f, 10.0f);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 2) // ParticleSettings
        {
            ImGui::Text("Particle Configuration");
            ImGui::Spacing();
            if (BeginPropertiesTable("ParticleDetails"))
            {
                DrawPropertyLabel("Max Particles");
                ImGui::InputInt("##Max Particles", &settings.particleCount);
                EndPropertyControl();

                DrawPropertyLabel("Base Mass");
                ImGui::SliderFloat("##Base Mass", &settings.baseMass, 0.1f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("Mass Randomness");
                ImGui::SliderFloat("##Mass Random", &settings.massRandomness, 0.0f, 1.0f);
                EndPropertyControl();

                DrawPropertyLabel("Lifetime (sec)");
                ImGui::SliderFloat("##Lifetime", &settings.particleLifetime, 0.1f, 20.0f);
                EndPropertyControl();

                DrawPropertyLabel("Life Randomness");
                ImGui::SliderFloat("##LifeRandom", &settings.lifetimeRandomness, 0.0f, 1.0f);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 3) // ColorGradient
        {
            ImGui::Text("Color Ramp stops");
            ImGui::Spacing();
            if (BeginPropertiesTable("ColorDetails"))
            {
                DrawPropertyLabel("Color Jitter");
                ImGui::SliderFloat("##ColorRandom", &settings.colorRandomness, 0.0f, 1.0f);
                EndPropertyControl();

                for (int stop = 0; stop < 4; ++stop)
                {
                    float c[4] = {
                        (m_connColor ? settings.colorGradient[stop].r : backupGradient[stop].r) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].g : backupGradient[stop].g) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].b : backupGradient[stop].b) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].a : backupGradient[stop].a) / 255.f
                    };
                    DrawPropertyLabel(std::string("Stop Color " + std::to_string(stop + 1)).c_str());
                    if (ImGui::ColorEdit4(std::string("##Stop" + std::to_string(stop)).c_str(), c, ImGuiColorEditFlags_AlphaBar))
                    {
                        CudaColor newColor(static_cast<std::uint8_t>(c[0]*255), static_cast<std::uint8_t>(c[1]*255), static_cast<std::uint8_t>(c[2]*255), static_cast<std::uint8_t>(c[3]*255));
                        if (m_connColor) settings.colorGradient[stop] = newColor;
                        else backupGradient[stop] = newColor;
                    }
                    EndPropertyControl();
                }
                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 4) // Renderer
        {
            ImGui::Text("Instanced Renderer");
            ImGui::Spacing();
            if (BeginPropertiesTable("RenderDetails"))
            {
                DrawPropertyLabel("Base Size");
                ImGui::SliderFloat("##Base Size", &settings.baseSize, 0.1f, 100.0f);
                EndPropertyControl();

                DrawPropertyLabel("Size Randomness");
                ImGui::SliderFloat("##SizeRandom", &settings.sizeRandomness, 0.0f, 1.0f);
                EndPropertyControl();

                DrawPropertyLabel("Start Scale");
                ImGui::SliderFloat("##StartMult", &settings.sizeStart, 0.0f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("End Scale");
                ImGui::SliderFloat("##EndMult", &settings.sizeEnd, 0.0f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("Min Rotation");
                ImGui::SliderFloat("##InitRotMin", &settings.initialRotationMin, -3.14f, 3.14f);
                EndPropertyControl();

                DrawPropertyLabel("Max Rotation");
                ImGui::SliderFloat("##InitRotMax", &settings.initialRotationMax, -3.14f, 3.14f);
                EndPropertyControl();

                DrawPropertyLabel("Min Ang Velocity");
                ImGui::SliderFloat("##AngVelMin", &settings.angularVelocityMin, -10.0f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("Max Ang Velocity");
                ImGui::SliderFloat("##AngVelMax", &settings.angularVelocityMax, -10.0f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("Viewport Grid");
                ImGui::Checkbox("##ShowGrid", &settings.showVisualGrid);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 5) // Velocity
        {
            ImGui::Text("Initial Velocity");
            ImGui::Spacing();
            if (BeginPropertiesTable("VelocityDetails"))
            {
                DrawPropertyLabel("Velocity X");
                ImGui::SliderFloat("##Vel X", m_connVelocity ? &settings.velocityX : &backupVelocityX, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Velocity Y");
                ImGui::SliderFloat("##Vel Y", m_connVelocity ? &settings.velocityY : &backupVelocityY, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Velocity Z");
                ImGui::SliderFloat("##Vel Z", m_connVelocity ? &settings.velocityZ : &backupVelocityZ, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Var X");
                ImGui::SliderFloat("##Var X", m_connVelocity ? &settings.velocityVarianceX : &backupVelocityVarX, 0.0f, 1000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Var Y");
                ImGui::SliderFloat("##Var Y", m_connVelocity ? &settings.velocityVarianceY : &backupVelocityVarY, 0.0f, 1000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Var Z");
                ImGui::SliderFloat("##Var Z", m_connVelocity ? &settings.velocityVarianceZ : &backupVelocityVarZ, 0.0f, 1000.0f);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 6) // Size
        {
            ImGui::Text("Size & Lifetime Scale");
            ImGui::Spacing();
            if (BeginPropertiesTable("SizeDetails"))
            {
                DrawPropertyLabel("Base Size");
                ImGui::SliderFloat("##Base Size", m_connSize ? &settings.baseSize : &backupBaseSize, 0.1f, 100.0f);
                EndPropertyControl();

                DrawPropertyLabel("Start Scale");
                ImGui::SliderFloat("##StartMult", m_connSize ? &settings.sizeStart : &backupSizeStart, 0.0f, 10.0f);
                EndPropertyControl();

                DrawPropertyLabel("End Scale");
                ImGui::SliderFloat("##EndMult", m_connSize ? &settings.sizeEnd : &backupSizeEnd, 0.0f, 10.0f);
                EndPropertyControl();

                EndPropertiesTable();
            }
        }
        else if (m_selectedNode == 7) // Camera
        {
            ImGui::Text("Camera & Target");
            ImGui::Spacing();
            if (BeginPropertiesTable("CameraDetails"))
            {
                DrawPropertyLabel("Target X");
                ImGui::SliderFloat("##CamTarX", &camera.targetX, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Target Y");
                ImGui::SliderFloat("##CamTarY", &camera.targetY, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Target Z");
                ImGui::SliderFloat("##CamTarZ", &camera.targetZ, -2000.0f, 2000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Yaw Angle");
                ImGui::SliderAngle("##CamYaw", &camera.yaw, -180.0f, 180.0f);
                EndPropertyControl();

                DrawPropertyLabel("Pitch Angle");
                ImGui::SliderAngle("##CamPitch", &camera.pitch, -89.0f, 89.0f);
                EndPropertyControl();

                DrawPropertyLabel("Orbit Distance");
                ImGui::SliderFloat("##CamDist", &camera.distance, 10.0f, 10000.0f);
                EndPropertyControl();

                DrawPropertyLabel("Reset Camera");
                if (ImGui::Button("Reset View", ImVec2(-1.0f, 0.0f)))
                {
                    camera.targetX = 640.0f;
                    camera.targetY = 360.0f;
                    camera.targetZ = 0.0f;
                    camera.yaw = 0.0f;
                    camera.pitch = 0.0f;
                    camera.distance = 800.0f;
                }
                EndPropertyControl();

                EndPropertiesTable();
            }
        }

        ImGui::EndChild(); // details scroll region

        ImGui::Separator();
        ImGui::Spacing();

        // Diagnostics Panel inside Right Inspector
        if (m_showDiagnostics)
        {
            ImGui::TextDisabled("DIAGNOSTICS & GPU METRICS");
            ImGui::Spacing();

            ImGui::Columns(2, "InspectorPerfGrid", false);
            ImGui::SetColumnWidth(0, 150.0f);
            ImGui::Text("FPS: %.1f", stats.fps);
            ImGui::Text("Avg FPS: %.1f", stats.averageFps);
            ImGui::Text("Active: %u", simStats.activeParticles);
            
            ImGui::NextColumn();
            ImGui::Text("Update: %.2fms", simStats.updateTimeMs);
            ImGui::Text("Render: %.2fms", simStats.renderTimeMs);

            ImGui::Columns(1);
            ImGui::Spacing();

            // Real-time mini charts
            ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0.0f, 0.85f, 1.0f, 1.0f));
            ImGui::PlotLines("##InspectorFPS", m_fpsHistory, 100, m_historyOffset, nullptr, 0.0f, 300.0f, ImVec2(-1.0f, 35.0f));
            ImGui::PopStyleColor();

            ImGui::Spacing();

            // GPU bars
            float util = static_cast<float>(stats.gpuUtilPercent) / 100.0f;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.0f, 0.8f, 1.0f, 1.0f));
            
            char utilL[32]; snprintf(utilL, sizeof(utilL), "GPU Util: %u%%", stats.gpuUtilPercent);
            ImGui::ProgressBar(util, ImVec2(-1.0f, 14.0f), utilL);

            float vramFrac = (stats.gpuMemTotalMb > 0) ? static_cast<float>(stats.gpuMemUsedMb) / static_cast<float>(stats.gpuMemTotalMb) : 0.0f;
            char vramL[32]; snprintf(vramL, sizeof(vramL), "VRAM: %u/%u MB", stats.gpuMemUsedMb, stats.gpuMemTotalMb);
            ImGui::ProgressBar(vramFrac, ImVec2(-1.0f, 14.0f), vramL);

            float tempFrac = stats.gpuTempC / 100.0f;
            char tempL[32]; snprintf(tempL, sizeof(tempL), "Temp: %.0f C", stats.gpuTempC);
            ImGui::ProgressBar(tempFrac, ImVec2(-1.0f, 14.0f), tempL);

            ImGui::PopStyleColor();
        }

        ImGui::End();
    }

    // ── 5. Bottom-Left Panel: Resizable Node Graph Panel (Opaque) ──
    if (m_showNodeGraph)
    {
        ImGui::SetNextWindowPos(ImVec2(leftWidth, displaySize.y - bottomHeight));
        ImGui::SetNextWindowSize(ImVec2(displaySize.x - leftWidth - rightWidth - timelineWidth, bottomHeight));
        
        ImGui::Begin("Node Graph Editor", &m_showNodeGraph, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::TextDisabled("PIPELINE NODE GRAPH EDITOR (MMB DRAG TO PAN, CLICK PIN TO CONNECT, RIGHT-CLICK TO DISCONNECT)");
        ImGui::Separator();
        
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImVec2 offset = ImGui::GetWindowPos();
        ImVec2 winSize = ImGui::GetWindowSize();
        ImGuiIO& io = ImGui::GetIO();

        // Handle Canvas Zoom (Mouse Wheel)
        if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0f)
        {
            float zoomDelta = io.MouseWheel * 0.1f;
            float prevZoom = m_canvasZoom;
            m_canvasZoom = std::max(0.3f, std::min(m_canvasZoom + zoomDelta, 2.0f));

            // Adjust scroll so we zoom towards mouse cursor
            ImVec2 mouseLocal = ImVec2(io.MousePos.x - offset.x, io.MousePos.y - offset.y);
            m_canvasScroll.x = mouseLocal.x - (mouseLocal.x - m_canvasScroll.x) * (m_canvasZoom / prevZoom);
            m_canvasScroll.y = mouseLocal.y - (mouseLocal.y - m_canvasScroll.y) * (m_canvasZoom / prevZoom);
        }

        // Handle Middle Mouse panning
        if (ImGui::IsWindowHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Middle))
        {
            m_canvasScroll.x += io.MouseDelta.x;
            m_canvasScroll.y += io.MouseDelta.y;
        }

        // Render Canvas Grid (infinitely scrolling & scaled)
        float grid_spacing = 30.0f * m_canvasZoom;
        float gridStartX = std::fmod(m_canvasScroll.x, grid_spacing);
        if (gridStartX < 0) gridStartX += grid_spacing;
        float gridStartY = std::fmod(m_canvasScroll.y, grid_spacing);
        if (gridStartY < 0) gridStartY += grid_spacing;
        
        for (float x = gridStartX; x < winSize.x; x += grid_spacing)
            drawList->AddLine(ImVec2(offset.x + x, offset.y), ImVec2(offset.x + x, offset.y + winSize.y), IM_COL32(40, 43, 46, 255));
        for (float y = gridStartY; y < winSize.y; y += grid_spacing)
            drawList->AddLine(ImVec2(offset.x, offset.y + y), ImVec2(offset.x + winSize.x, offset.y + y), IM_COL32(40, 43, 46, 255));

        // Pre-compute pin screen positions based on current canvas scrolling and zoom
        for (auto& node : m_nodes)
        {
            ImVec2 nodeScreenPos = ImVec2(offset.x + m_canvasScroll.x + node.pos.x * m_canvasZoom, offset.y + m_canvasScroll.y + node.pos.y * m_canvasZoom);
            for (size_t j = 0; j < node.inputs.size(); ++j)
            {
                node.inputs[j].screenPos = ImVec2(nodeScreenPos.x, nodeScreenPos.y + ((j == 0) ? 35.0f : (35.0f + 30.0f * j)) * m_canvasZoom);
            }
            for (size_t j = 0; j < node.outputs.size(); ++j)
            {
                float y = 45.0f;
                if (node.id == 2) y = 50.0f;
                node.outputs[j].screenPos = ImVec2(nodeScreenPos.x + node.size.x * m_canvasZoom, nodeScreenPos.y + y * m_canvasZoom);
            }
        }

        // Draw connections (bezier curves)
        int linkToDelete = -1;
        int linkIdx = 0;
        for (const auto& link : m_links)
        {
            const Node* fromNode = nullptr;
            const Node* toNode = nullptr;
            for (const auto& n : m_nodes)
            {
                if (n.id == link.fromNodeId) fromNode = &n;
                if (n.id == link.toNodeId) toNode = &n;
            }

            if (fromNode && toNode && !fromNode->outputs.empty() && toNode->inputs.size() > static_cast<size_t>(link.toPinIdx))
            {
                ImVec2 start = fromNode->outputs[0].screenPos;
                ImVec2 end = toNode->inputs[link.toPinIdx].screenPos;

                // Pulse animation for active links
                float timeSec = static_cast<float>(ImGui::GetTime());
                int alpha = static_cast<int>(200 + 55 * std::sin(timeSec * 4.0f + linkIdx));
                ImU32 color = IM_COL32(255, 255, 255, alpha);
                switch (fromNode->outputs[0].type)
                {
                    case PinType::Flow:   color = IM_COL32(255, 200, 50, alpha); break;
                    case PinType::Forces: color = IM_COL32(50, 220, 100, alpha); break;
                    case PinType::Render: color = IM_COL32(180, 80, 255, alpha); break;
                    case PinType::Color:  color = IM_COL32(0, 210, 255, alpha); break;
                    case PinType::Velocity: color = IM_COL32(255, 0, 255, alpha); break;
                    case PinType::Size:   color = IM_COL32(255, 140, 0, alpha); break;
                }

                // Draw Bezier
                float cpOffset = 60.0f * m_canvasZoom;
                drawList->AddBezierCubic(start, ImVec2(start.x + cpOffset, start.y), ImVec2(end.x - cpOffset, end.y), end, color, 2.5f * m_canvasZoom);

                // Midpoint selection handle
                ImVec2 midPoint = ImVec2((start.x + end.x) * 0.5f, (start.y + end.y) * 0.5f);
                ImGui::SetCursorScreenPos(ImVec2(midPoint.x - 6.0f, midPoint.y - 6.0f));
                ImGui::PushID(2000 + linkIdx);
                ImGui::InvisibleButton("##link_mid", ImVec2(12.0f, 12.0f));
                bool handleHovered = ImGui::IsItemHovered();
                
                ImU32 circleColor = handleHovered ? IM_COL32(255, 50, 50, 255) : color;
                float radius = (handleHovered ? 6.0f : 4.0f) * m_canvasZoom;
                drawList->AddCircleFilled(midPoint, radius, circleColor);
                drawList->AddCircle(midPoint, radius, IM_COL32(0, 0, 0, 255), 0, 1.0f * m_canvasZoom);

                if (handleHovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
                {
                    ImGui::OpenPopup("DeleteLinkPopup");
                }
                if (ImGui::BeginPopup("DeleteLinkPopup"))
                {
                    if (ImGui::MenuItem("Delete Connection"))
                    {
                        linkToDelete = linkIdx;
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopID();
            }
            linkIdx++;
        }

        if (linkToDelete != -1)
        {
            m_links.erase(m_links.begin() + linkToDelete);
        }

        // Draw drag connection preview
        if (m_dragLinkActive)
        {
            const Node* startNode = nullptr;
            for (const auto& n : m_nodes)
            {
                if (n.id == m_dragStartNodeId) startNode = &n;
            }

            if (startNode)
            {
                ImVec2 startPos = ImVec2(0.0f, 0.0f);
                PinType dragType = PinType::Flow;
                if (m_dragStartIsOutput)
                {
                    if (startNode->outputs.size() > static_cast<size_t>(m_dragStartPinIdx))
                    {
                        startPos = startNode->outputs[m_dragStartPinIdx].screenPos;
                        dragType = startNode->outputs[m_dragStartPinIdx].type;
                    }
                }
                else
                {
                    if (startNode->inputs.size() > static_cast<size_t>(m_dragStartPinIdx))
                    {
                        startPos = startNode->inputs[m_dragStartPinIdx].screenPos;
                        dragType = startNode->inputs[m_dragStartPinIdx].type;
                    }
                }

                ImU32 color = IM_COL32(255, 255, 255, 180);
                switch (dragType)
                {
                    case PinType::Flow:   color = IM_COL32(255, 200, 50, 180); break;
                    case PinType::Forces: color = IM_COL32(50, 220, 100, 180); break;
                    case PinType::Render: color = IM_COL32(180, 80, 255, 180); break;
                    case PinType::Color:  color = IM_COL32(0, 210, 255, 180); break;
                    case PinType::Velocity: color = IM_COL32(255, 0, 255, 180); break;
                    case PinType::Size:   color = IM_COL32(255, 140, 0, 180); break;
                }

                ImVec2 mousePos = io.MousePos;
                float cpOffset = 60.0f * m_canvasZoom;
                if (m_dragStartIsOutput)
                {
                    drawList->AddBezierCubic(startPos, ImVec2(startPos.x + cpOffset, startPos.y), ImVec2(mousePos.x - cpOffset, mousePos.y), mousePos, color, 2.5f * m_canvasZoom);
                }
                else
                {
                    drawList->AddBezierCubic(mousePos, ImVec2(mousePos.x + cpOffset, mousePos.y), ImVec2(startPos.x - cpOffset, startPos.y), startPos, color, 2.5f * m_canvasZoom);
                }
            }
        }

        // Nodes drawing
        ImU32 headerColors[7] = {
            IM_COL32(0, 110, 150, 255),   // Emitter
            IM_COL32(0, 120, 70, 255),    // Forces
            IM_COL32(100, 45, 140, 255),  // Config
            IM_COL32(150, 70, 35, 255),   // Color
            IM_COL32(0, 100, 100, 255),   // Render
            IM_COL32(140, 0, 140, 255),   // Velocity
            IM_COL32(160, 80, 0, 255)     // Size
        };

        bool hoveredSocketFound = false;
        int hoveredSocketNodeId = -1;
        int hoveredSocketPinIdx = -1;
        bool hoveredSocketIsOutput = false;

        for (auto& node : m_nodes)
        {
            ImVec2 nodeScreenPos = ImVec2(offset.x + m_canvasScroll.x + node.pos.x * m_canvasZoom, offset.y + m_canvasScroll.y + node.pos.y * m_canvasZoom);
            ImVec2 nodeScreenSize = ImVec2(node.size.x * m_canvasZoom, node.size.y * m_canvasZoom);
            bool isSelected = (m_selectedNode == node.id);
            ImU32 borderColor = isSelected ? IM_COL32(0, 210, 255, 255) : IM_COL32(50, 50, 60, 255);
            float borderThickness = isSelected ? 2.5f : 1.0f;

            // Shadow
            ImVec2 shadowPos = ImVec2(nodeScreenPos.x + 5.0f * m_canvasZoom, nodeScreenPos.y + 5.0f * m_canvasZoom);
            drawList->AddRectFilled(shadowPos, ImVec2(shadowPos.x + nodeScreenSize.x, shadowPos.y + nodeScreenSize.y), IM_COL32(0, 0, 0, 150), 6.0f * m_canvasZoom);

            // Box
            drawList->AddRectFilled(nodeScreenPos, ImVec2(nodeScreenPos.x + nodeScreenSize.x, nodeScreenPos.y + nodeScreenSize.y), IM_COL32(20, 20, 24, 255), 6.0f * m_canvasZoom);
            drawList->AddRectFilled(nodeScreenPos, ImVec2(nodeScreenPos.x + nodeScreenSize.x, nodeScreenPos.y + 24.0f * m_canvasZoom), headerColors[node.id], 6.0f * m_canvasZoom, ImDrawFlags_RoundCornersTop);
            drawList->AddRect(nodeScreenPos, ImVec2(nodeScreenPos.x + nodeScreenSize.x, nodeScreenPos.y + nodeScreenSize.y), borderColor, 6.0f * m_canvasZoom, 0, borderThickness);

            // Status dot
            bool isBypassed = false;
            if (node.id == 0 && !m_connEmitter) isBypassed = true;
            if (node.id == 1 && !m_connForces) isBypassed = true;
            if (node.id == 3 && !m_connColor) isBypassed = true;
            if (node.id == 5 && !m_connVelocity) isBypassed = true;
            if (node.id == 6 && !m_connSize) isBypassed = true;
            
            ImU32 statusColor = isBypassed ? IM_COL32(200, 50, 50, 255) : IM_COL32(50, 200, 50, 255);
            ImVec2 statusPos = ImVec2(nodeScreenPos.x + nodeScreenSize.x - 12.0f * m_canvasZoom, nodeScreenPos.y + 12.0f * m_canvasZoom);
            drawList->AddCircleFilled(statusPos, 4.0f * m_canvasZoom, statusColor);

            // Title
            ImGui::SetWindowFontScale(m_canvasZoom);
            drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 4.0f * m_canvasZoom), IM_COL32(230, 230, 230, 255), node.name.c_str(), nullptr, 0.0f, nullptr);
            ImGui::SetWindowFontScale(1.0f); // Reset for widgets

            // Draw interactive widgets inside node body
            ImGui::SetWindowFontScale(m_canvasZoom);
            ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 30.0f * m_canvasZoom));
            ImGui::PushID(node.id);
            ImGui::PushItemWidth(184.0f * m_canvasZoom);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4.0f * m_canvasZoom, 3.0f * m_canvasZoom));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f * m_canvasZoom, 4.0f * m_canvasZoom));

            if (node.id == 0)
            {
                ImGui::Text("Shape: %s", (settings.shape == EmitterShape::Point) ? "Point" :
                            (settings.shape == EmitterShape::Circle) ? "Circle" :
                            (settings.shape == EmitterShape::Box) ? "Box" :
                            (settings.shape == EmitterShape::Sphere) ? "Sphere" :
                            (settings.shape == EmitterShape::Cube) ? "Cube" : "Grid");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_rate", m_connEmitter ? &settings.spawnRate : &backupSpawnRate, 0.0f, 5000.0f, "Rate: %.0f");
            }
            else if (node.id == 1)
            {
                ImGui::Text("Wind: %.0f, %.0f, %.0f", m_connForces ? settings.windX : backupWindX, m_connForces ? settings.windY : backupWindY, m_connForces ? settings.windZ : backupWindZ);
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_grav", m_connForces ? &settings.gravity : &backupGravity, -2000.0f, 2000.0f, "Grav: %.0f");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 78.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_drag", m_connForces ? &settings.drag : &backupDrag, 0.0f, 10.0f, "Drag: %.2f");
            }
            else if (node.id == 2)
            {
                ImGui::SliderFloat("##nd_life", &settings.particleLifetime, 0.1f, 20.0f, "Life: %.1fs");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_mass", &settings.baseMass, 0.1f, 10.0f, "Mass: %.1f");
            }
            else if (node.id == 3)
            {
                ImGui::TextUnformatted("Color Stops Edit:");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                for (int stop = 0; stop < 4; ++stop)
                {
                    ImVec4 sColor = ImVec4(
                        (m_connColor ? settings.colorGradient[stop].r : backupGradient[stop].r) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].g : backupGradient[stop].g) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].b : backupGradient[stop].b) / 255.f,
                        (m_connColor ? settings.colorGradient[stop].a : backupGradient[stop].a) / 255.f
                    );
                    ImGui::PushID(stop);
                    if (ImGui::ColorButton("##cs_btn", sColor, ImGuiColorEditFlags_None, ImVec2(32.0f * m_canvasZoom, 18.0f * m_canvasZoom)))
                    {
                        ImGui::OpenPopup("ColorPickerPopup");
                    }
                    if (ImGui::BeginPopup("ColorPickerPopup"))
                    {
                        float c[4] = { sColor.x, sColor.y, sColor.z, sColor.w };
                        if (ImGui::ColorPicker4("Color", c, ImGuiColorEditFlags_AlphaBar))
                        {
                            CudaColor newColor(static_cast<uint8_t>(c[0]*255), static_cast<uint8_t>(c[1]*255), static_cast<uint8_t>(c[2]*255), static_cast<uint8_t>(c[3]*255));
                            if (m_connColor) settings.colorGradient[stop] = newColor;
                            else backupGradient[stop] = newColor;
                        }
                        ImGui::EndPopup();
                    }
                    ImGui::PopID();
                    if (stop < 3)
                        ImGui::SameLine();
                }

                // Draw Horizontal Gradient Preview Bar
                CudaColor cc0 = m_connColor ? settings.colorGradient[0] : backupGradient[0];
                CudaColor cc1 = m_connColor ? settings.colorGradient[1] : backupGradient[1];
                CudaColor cc2 = m_connColor ? settings.colorGradient[2] : backupGradient[2];
                CudaColor cc3 = m_connColor ? settings.colorGradient[3] : backupGradient[3];

                ImU32 col0 = IM_COL32(cc0.r, cc0.g, cc0.b, cc0.a);
                ImU32 col1 = IM_COL32(cc1.r, cc1.g, cc1.b, cc1.a);
                ImU32 col2 = IM_COL32(cc2.r, cc2.g, cc2.b, cc2.a);
                ImU32 col3 = IM_COL32(cc3.r, cc3.g, cc3.b, cc3.a);

                float barY = nodeScreenPos.y + 78.0f * m_canvasZoom;
                float barH = 10.0f * m_canvasZoom;
                float segW = (184.0f / 3.0f) * m_canvasZoom;
                float barX = nodeScreenPos.x + 8.0f * m_canvasZoom;

                drawList->AddRectFilledMultiColor(ImVec2(barX, barY), ImVec2(barX + segW, barY + barH), col0, col1, col1, col0);
                drawList->AddRectFilledMultiColor(ImVec2(barX + segW, barY), ImVec2(barX + 2.0f * segW, barY + barH), col1, col2, col2, col1);
                drawList->AddRectFilledMultiColor(ImVec2(barX + 2.0f * segW, barY), ImVec2(barX + 184.0f * m_canvasZoom, barY + barH), col2, col3, col3, col2);
            }
            else if (node.id == 4)
            {
                ImGui::Text("Active: %u", simStats.activeParticles);
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::Checkbox("Show Floor Grid", &settings.showVisualGrid);
            }
            else if (node.id == 5)
            {
                ImGui::SliderFloat("##nd_velY", m_connVelocity ? &settings.velocityY : &backupVelocityY, -2000.0f, 2000.0f, "Vel Y: %.0f");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_vVarX", m_connVelocity ? &settings.velocityVarianceX : &backupVelocityVarX, 0.0f, 1000.0f, "Var X: %.0f");
            }
            else if (node.id == 6)
            {
                ImGui::SliderFloat("##nd_bsize", m_connSize ? &settings.baseSize : &backupBaseSize, 0.1f, 50.0f, "Base: %.1f");
                ImGui::SetCursorScreenPos(ImVec2(nodeScreenPos.x + 8.0f * m_canvasZoom, nodeScreenPos.y + 54.0f * m_canvasZoom));
                ImGui::SliderFloat("##nd_sstart", m_connSize ? &settings.sizeStart : &backupSizeStart, 0.0f, 10.0f, "Start M: %.1f");
            }
            ImGui::PopStyleVar(2);
            ImGui::PopItemWidth();
            ImGui::PopID();
            ImGui::SetWindowFontScale(1.0f); // Reset after widgets

            // Sockets logic
            // Draw & handle input pins
            for (size_t j = 0; j < node.inputs.size(); ++j)
            {
                ImVec2 socketPos = node.inputs[j].screenPos;
                ImU32 socketColor = IM_COL32(30, 30, 35, 255);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);
                switch (node.inputs[j].type)
                {
                    case PinType::Flow:   outlineColor = IM_COL32(255, 200, 50, 255); break;
                    case PinType::Forces: outlineColor = IM_COL32(50, 220, 100, 255); break;
                    case PinType::Render: outlineColor = IM_COL32(180, 80, 255, 255); break;
                    case PinType::Color:  outlineColor = IM_COL32(0, 210, 255, 255); break;
                    case PinType::Velocity: outlineColor = IM_COL32(255, 0, 255, 255); break;
                    case PinType::Size:   outlineColor = IM_COL32(255, 140, 0, 255); break;
                }

                bool isConnected = false;
                for (const auto& l : m_links)
                {
                    if (l.toNodeId == node.id && l.toPinIdx == static_cast<int>(j))
                        isConnected = true;
                }
                if (isConnected) socketColor = outlineColor;

                drawList->AddCircleFilled(socketPos, 5.0f * m_canvasZoom, socketColor);
                drawList->AddCircle(socketPos, 5.0f * m_canvasZoom, outlineColor, 0, 1.5f * m_canvasZoom);

                // Draw pin label (outside node, to the left)
                ImGui::SetWindowFontScale(m_canvasZoom * 0.8f);
                float labelWidth = ImGui::CalcTextSize(node.inputs[j].name.c_str()).x;
                drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(socketPos.x - labelWidth - 8.0f * m_canvasZoom, socketPos.y - 6.0f * m_canvasZoom), IM_COL32(200, 200, 200, 255), node.inputs[j].name.c_str(), nullptr, 0.0f, nullptr);
                ImGui::SetWindowFontScale(1.0f);

                // Invisible button to capture socket clicks
                ImGui::SetCursorScreenPos(ImVec2(socketPos.x - 7.0f * m_canvasZoom, socketPos.y - 7.0f * m_canvasZoom));
                ImGui::PushID(3000 + node.id * 10 + j);
                ImGui::InvisibleButton("##socket_in", ImVec2(14.0f * m_canvasZoom, 14.0f * m_canvasZoom));
                
                if (ImGui::IsItemHovered())
                {
                    hoveredSocketFound = true;
                    hoveredSocketNodeId = node.id;
                    hoveredSocketPinIdx = static_cast<int>(j);
                    hoveredSocketIsOutput = false;
                    ImGui::SetTooltip("%s", node.inputs[j].name.c_str());
                }

                // Disconnect existing link by dragging it away
                if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0) && isConnected && !m_dragLinkActive)
                {
                    for (auto it = m_links.begin(); it != m_links.end(); ++it)
                    {
                        if (it->toNodeId == node.id && it->toPinIdx == static_cast<int>(j))
                        {
                            m_dragLinkActive = true;
                            m_dragStartNodeId = it->fromNodeId;
                            m_dragStartPinIdx = 0; // index of output
                            m_dragStartIsOutput = true;
                            m_links.erase(it);
                            break;
                        }
                    }
                }
                ImGui::PopID();
            }

            // Draw & handle output pins
            for (size_t j = 0; j < node.outputs.size(); ++j)
            {
                ImVec2 socketPos = node.outputs[j].screenPos;
                ImU32 socketColor = IM_COL32(30, 30, 35, 255);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);
                switch (node.outputs[j].type)
                {
                    case PinType::Flow:   outlineColor = IM_COL32(255, 200, 50, 255); break;
                    case PinType::Forces: outlineColor = IM_COL32(50, 220, 100, 255); break;
                    case PinType::Render: outlineColor = IM_COL32(180, 80, 255, 255); break;
                    case PinType::Color:  outlineColor = IM_COL32(0, 210, 255, 255); break;
                    case PinType::Velocity: outlineColor = IM_COL32(255, 0, 255, 255); break;
                    case PinType::Size:   outlineColor = IM_COL32(255, 140, 0, 255); break;
                }

                bool isConnected = false;
                for (const auto& l : m_links)
                {
                    if (l.fromNodeId == node.id) isConnected = true;
                }
                if (isConnected) socketColor = outlineColor;

                drawList->AddCircleFilled(socketPos, 5.0f * m_canvasZoom, socketColor);
                drawList->AddCircle(socketPos, 5.0f * m_canvasZoom, outlineColor, 0, 1.5f * m_canvasZoom);

                // Draw pin label (outside node, to the right)
                ImGui::SetWindowFontScale(m_canvasZoom * 0.8f);
                drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(socketPos.x + 8.0f * m_canvasZoom, socketPos.y - 6.0f * m_canvasZoom), IM_COL32(200, 200, 200, 255), node.outputs[j].name.c_str(), nullptr, 0.0f, nullptr);
                ImGui::SetWindowFontScale(1.0f);

                // Invisible button to capture socket clicks
                ImGui::SetCursorScreenPos(ImVec2(socketPos.x - 7.0f * m_canvasZoom, socketPos.y - 7.0f * m_canvasZoom));
                ImGui::PushID(4000 + node.id * 10 + j);
                ImGui::InvisibleButton("##socket_out", ImVec2(14.0f * m_canvasZoom, 14.0f * m_canvasZoom));
                
                if (ImGui::IsItemHovered())
                {
                    hoveredSocketFound = true;
                    hoveredSocketNodeId = node.id;
                    hoveredSocketPinIdx = static_cast<int>(j);
                    hoveredSocketIsOutput = true;
                    ImGui::SetTooltip("%s", node.outputs[j].name.c_str());
                }

                if (ImGui::IsItemActive() && ImGui::IsMouseClicked(0))
                {
                    m_dragLinkActive = true;
                    m_dragStartNodeId = node.id;
                    m_dragStartPinIdx = static_cast<int>(j);
                    m_dragStartIsOutput = true;
                }
                ImGui::PopID();
            }

            // Node header drag trigger
            ImGui::SetCursorScreenPos(nodeScreenPos);
            std::string invisibleButtonId = "##nodebtn_" + std::to_string(node.id);
            ImGui::InvisibleButton(invisibleButtonId.c_str(), ImVec2(nodeScreenSize.x, 24.0f * m_canvasZoom));
            if (ImGui::IsItemActive())
            {
                node.pos.x += io.MouseDelta.x / m_canvasZoom;
                node.pos.y += io.MouseDelta.y / m_canvasZoom;
            }
            if (ImGui::IsItemClicked())
            {
                m_selectedNode = node.id;
            }
        }

        // Handle link creation when mouse is released
        if (m_dragLinkActive && !ImGui::IsMouseDown(0))
        {
            if (hoveredSocketFound && !hoveredSocketIsOutput && m_dragStartIsOutput)
            {
                if (m_dragStartNodeId != hoveredSocketNodeId)
                {
                    const Node* fromNode = nullptr;
                    const Node* toNode = nullptr;
                    for (const auto& n : m_nodes)
                    {
                        if (n.id == m_dragStartNodeId) fromNode = &n;
                        if (n.id == hoveredSocketNodeId) toNode = &n;
                    }

                    if (fromNode && toNode && fromNode->outputs.size() > static_cast<size_t>(m_dragStartPinIdx) && toNode->inputs.size() > static_cast<size_t>(hoveredSocketPinIdx))
                    {
                        PinType outType = fromNode->outputs[m_dragStartPinIdx].type;
                        PinType inType = toNode->inputs[hoveredSocketPinIdx].type;

                        if (outType == inType)
                        {
                            // Remove existing link on this input socket
                            for (auto it = m_links.begin(); it != m_links.end(); )
                            {
                                if (it->toNodeId == hoveredSocketNodeId && it->toPinIdx == hoveredSocketPinIdx)
                                {
                                    it = m_links.erase(it);
                                }
                                else
                                {
                                    ++it;
                                }
                            }

                            // Create connection
                            m_links.push_back({ m_dragStartNodeId, hoveredSocketNodeId, hoveredSocketPinIdx });
                        }
                    }
                }
            }
            m_dragLinkActive = false;
        }

        // Preset drag & drop target zones in the Node Graph
        if (ImGui::BeginDragDropTarget())
        {
            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("PRESET_DRAG"))
            {
                int pVal = *(const int*)payload->Data;
                undoSystem.PushUndo(settings);
                ApplyPreset(settings, static_cast<ArtistPreset>(pVal));
                RestoreDefaultLinks();
                result.resetRequested = true;
            }
            ImGui::EndDragDropTarget();
        }

        // Right-click on empty canvas opens context menu
        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup) && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !hoveredSocketFound)
        {
            bool hoveringNode = false;
            for (const auto& node : m_nodes)
            {
                ImVec2 nodeScreenPos = ImVec2(offset.x + m_canvasScroll.x + node.pos.x * m_canvasZoom, offset.y + m_canvasScroll.y + node.pos.y * m_canvasZoom);
                if (io.MousePos.x >= nodeScreenPos.x && io.MousePos.x <= nodeScreenPos.x + node.size.x * m_canvasZoom &&
                    io.MousePos.y >= nodeScreenPos.y && io.MousePos.y <= nodeScreenPos.y + node.size.y * m_canvasZoom)
                {
                    hoveringNode = true;
                }
            }
            if (!hoveringNode)
            {
                ImGui::OpenPopup("CanvasContextMenu");
            }
        }

        if (ImGui::BeginPopup("CanvasContextMenu"))
        {
            if (ImGui::MenuItem("Reset Canvas View"))
            {
                m_canvasScroll = ImVec2(0.0f, 0.0f);
            }
            if (ImGui::MenuItem("Reset Node Positions"))
            {
                m_nodes[0].pos = ImVec2(40.0f, 40.0f);
                m_nodes[1].pos = ImVec2(40.0f, 180.0f);
                m_nodes[2].pos = ImVec2(300.0f, 110.0f);
                m_nodes[3].pos = ImVec2(40.0f, 460.0f);
                m_nodes[4].pos = ImVec2(560.0f, 220.0f);
                m_nodes[5].pos = ImVec2(40.0f, 320.0f);
                m_nodes[6].pos = ImVec2(300.0f, 460.0f);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Restore Default Links"))
            {
                RestoreDefaultLinks();
            }
            if (ImGui::MenuItem("Delete All Links"))
            {
                m_links.clear();
            }
            ImGui::EndPopup();
        }

        // Draw Minimap
        float minimapSize = 120.0f;
        ImVec2 minimapPos = ImVec2(offset.x + winSize.x - minimapSize - 20.0f, offset.y + winSize.y - minimapSize - 20.0f);
        drawList->AddRectFilled(minimapPos, ImVec2(minimapPos.x + minimapSize, minimapPos.y + minimapSize), IM_COL32(10, 10, 12, 200), 4.0f);
        drawList->AddRect(minimapPos, ImVec2(minimapPos.x + minimapSize, minimapPos.y + minimapSize), IM_COL32(100, 100, 100, 150), 4.0f);
        
        // Find bounding box of all nodes
        ImVec2 minPos(9999.0f, 9999.0f), maxPos(-9999.0f, -9999.0f);
        for (const auto& node : m_nodes)
        {
            minPos.x = std::min(minPos.x, node.pos.x);
            minPos.y = std::min(minPos.y, node.pos.y);
            maxPos.x = std::max(maxPos.x, node.pos.x + node.size.x);
            maxPos.y = std::max(maxPos.y, node.pos.y + node.size.y);
        }
        
        // Add padding
        minPos.x -= 200.0f; minPos.y -= 200.0f;
        maxPos.x += 200.0f; maxPos.y += 200.0f;
        
        float mapWidth = maxPos.x - minPos.x;
        float mapHeight = maxPos.y - minPos.y;
        float mapScale = std::min(minimapSize / mapWidth, minimapSize / mapHeight);
        
        // Draw nodes on minimap
        for (const auto& node : m_nodes)
        {
            ImVec2 mappedPos = ImVec2(minimapPos.x + (node.pos.x - minPos.x) * mapScale, minimapPos.y + (node.pos.y - minPos.y) * mapScale);
            ImVec2 mappedSize = ImVec2(node.size.x * mapScale, node.size.y * mapScale);
            drawList->AddRectFilled(mappedPos, ImVec2(mappedPos.x + mappedSize.x, mappedPos.y + mappedSize.y), IM_COL32(150, 150, 150, 200), 1.0f);
        }

        // Draw viewport rect on minimap
        ImVec2 viewStart = ImVec2(minimapPos.x + (-m_canvasScroll.x / m_canvasZoom - minPos.x) * mapScale, minimapPos.y + (-m_canvasScroll.y / m_canvasZoom - minPos.y) * mapScale);
        ImVec2 viewSize = ImVec2((winSize.x / m_canvasZoom) * mapScale, (winSize.y / m_canvasZoom) * mapScale);
        drawList->AddRect(viewStart, ImVec2(viewStart.x + viewSize.x, viewStart.y + viewSize.y), IM_COL32(255, 255, 255, 180), 0, 0, 1.0f);

        ImGui::End();
    }

    // ── 6. Bottom-Right Panel: Resizable Timeline Panel (Opaque) ──
    if (m_showDiagnostics)
    {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - rightWidth - timelineWidth, displaySize.y - bottomHeight));
        ImGui::SetNextWindowSize(ImVec2(timelineWidth, bottomHeight));
        
        ImGui::Begin("Playback Timeline", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);

        ImGui::TextDisabled("PLAYBACK TIMELINE CONTROLS");
        ImGui::Separator();
        ImGui::Spacing();

        // Control buttons
        if (ImGui::Button("[ |< ] Reset"))
        {
            particleSystem.SetSimulationTime(0.0f);
            result.resetRequested = true;
        }
        ImGui::SameLine();
        if (settings.paused)
        {
            if (ImGui::Button("[  >  ] Play")) settings.paused = false;
        }
        else
        {
            if (ImGui::Button("[  || ] Pause")) settings.paused = true;
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!settings.paused);
        if (ImGui::Button("[  >| ] Step"))
        {
            particleSystem.Update(0.016f * settings.timeScale, settings, const_cast<SimulationStats&>(simStats));
        }
        ImGui::EndDisabled();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Duration & Loop settings
        ImGui::Checkbox("Loop Playback", &m_loopTimeline);
        if (m_loopTimeline)
        {
            ImGui::PushItemWidth(140.0f);
            ImGui::SliderFloat("Duration", &m_timelineDuration, 1.0f, 30.0f, "%.1fs");
            ImGui::PopItemWidth();
        }

        ImGui::Spacing();

        // Playback speed scale
        ImGui::PushItemWidth(140.0f);
        ImGui::SliderFloat("Play Speed", &settings.timeScale, 0.1f, 3.0f, "%.2fx");
        ImGui::PopItemWidth();

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Time Progress bar
        float simTime = particleSystem.GetSimulationTime();
        ImGui::Text("Current Time: %5.2fs", simTime);
        float progress = m_loopTimeline ? (simTime / m_timelineDuration) : 0.0f;
        if (progress > 1.0f) progress = 1.0f;
        
        char progressLabel[32];
        snprintf(progressLabel, sizeof(progressLabel), "%5.2fs / %5.2fs", simTime, m_loopTimeline ? m_timelineDuration : 999.0f);
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), progressLabel);

        // Check looping condition
        if (m_loopTimeline && simTime > m_timelineDuration)
        {
            particleSystem.SetSimulationTime(0.0f);
            result.resetRequested = true;
        }

        ImGui::End();
    }

    ImGui::PopStyleVar(4); // pop styling

    // ── Render Custom Splitters for Resizing Panels ──
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));

    // Left Splitter (Vertical)
    if (m_showOutliner)
    {
        ImGui::SetNextWindowPos(ImVec2(m_leftPanelWidth - 3.0f, menuBarHeight));
        ImGui::SetNextWindowSize(ImVec2(6.0f, displaySize.y - menuBarHeight));
        ImGui::Begin("##SplitterLeft", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
        ImGui::InvisibleButton("##split_l", ImVec2(6.0f, displaySize.y - menuBarHeight));
        bool hovered = ImGui::IsItemHovered();
        bool active = ImGui::IsItemActive();
        if (hovered || active)
        {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(ImGui::GetWindowPos().x + 2.0f, menuBarHeight),
                ImVec2(ImGui::GetWindowPos().x + 4.0f, displaySize.y),
                IM_COL32(0, 210, 255, 200)
            );
        }
        if (active)
        {
            m_leftPanelWidth += ImGui::GetIO().MouseDelta.x;
            m_leftPanelWidth = std::max(150.0f, std::min(m_leftPanelWidth, 500.0f));
        }
        ImGui::End();
    }

    // Right Splitter (Vertical)
    if (m_showDetails)
    {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - m_rightPanelWidth - 3.0f, menuBarHeight));
        ImGui::SetNextWindowSize(ImVec2(6.0f, displaySize.y - menuBarHeight));
        ImGui::Begin("##SplitterRight", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
        ImGui::InvisibleButton("##split_r", ImVec2(6.0f, displaySize.y - menuBarHeight));
        bool hovered = ImGui::IsItemHovered();
        bool active = ImGui::IsItemActive();
        if (hovered || active)
        {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(ImGui::GetWindowPos().x + 2.0f, menuBarHeight),
                ImVec2(ImGui::GetWindowPos().x + 4.0f, displaySize.y),
                IM_COL32(0, 210, 255, 200)
            );
        }
        if (active)
        {
            m_rightPanelWidth -= ImGui::GetIO().MouseDelta.x;
            m_rightPanelWidth = std::max(200.0f, std::min(m_rightPanelWidth, 600.0f));
        }
        ImGui::End();
    }

    // Bottom Splitter (Horizontal)
    if (m_showNodeGraph || m_showDiagnostics)
    {
        ImGui::SetNextWindowPos(ImVec2(leftWidth, displaySize.y - m_bottomPanelHeight - 3.0f));
        ImGui::SetNextWindowSize(ImVec2(displaySize.x - leftWidth - rightWidth, 6.0f));
        ImGui::Begin("##SplitterBottom", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
        ImGui::InvisibleButton("##split_b", ImVec2(displaySize.x - leftWidth - rightWidth, 6.0f));
        bool hovered = ImGui::IsItemHovered();
        bool active = ImGui::IsItemActive();
        if (hovered || active)
        {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(leftWidth, ImGui::GetWindowPos().y + 2.0f),
                ImVec2(displaySize.x - rightWidth, ImGui::GetWindowPos().y + 4.0f),
                IM_COL32(0, 210, 255, 200)
            );
        }
        if (active)
        {
            m_bottomPanelHeight -= ImGui::GetIO().MouseDelta.y;
            m_bottomPanelHeight = std::max(150.0f, std::min(m_bottomPanelHeight, 600.0f));
        }
        ImGui::End();
    }

    // Timeline Splitter (Vertical, between Node Graph and Playback Timeline)
    if (m_showNodeGraph && m_showDiagnostics)
    {
        ImGui::SetNextWindowPos(ImVec2(displaySize.x - rightWidth - timelineWidth - 3.0f, displaySize.y - bottomHeight));
        ImGui::SetNextWindowSize(ImVec2(6.0f, bottomHeight));
        ImGui::Begin("##SplitterTimeline", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);
        ImGui::InvisibleButton("##split_t", ImVec2(6.0f, bottomHeight));
        bool hovered = ImGui::IsItemHovered();
        bool active = ImGui::IsItemActive();
        if (hovered || active)
        {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(ImGui::GetWindowPos().x + 2.0f, displaySize.y - bottomHeight),
                ImVec2(ImGui::GetWindowPos().x + 4.0f, displaySize.y),
                IM_COL32(0, 210, 255, 200)
            );
        }
        if (active)
        {
            m_timelineWidth -= ImGui::GetIO().MouseDelta.x;
            m_timelineWidth = std::max(200.0f, std::min(m_timelineWidth, 600.0f));
        }
        ImGui::End();
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    // ── 7. Undo/Redo State Tracker ──
    bool isAnyItemActive = ImGui::IsAnyItemActive();
    if (isAnyItemActive && !m_wasAnyItemActive)
    {
        m_settingsBeforeEdit = startOfFrameSettings;
    }
    else if (!isAnyItemActive && m_wasAnyItemActive)
    {
        if (settings != m_settingsBeforeEdit)
        {
            undoSystem.PushUndo(m_settingsBeforeEdit);
        }
    }
    m_wasAnyItemActive = isAnyItemActive;

    m_prevConnEmitter = m_connEmitter;
    m_prevConnForces = m_connForces;
    m_prevConnColor = m_connColor;
    m_prevConnVelocity = m_connVelocity;
    m_prevConnSize = m_connSize;

    return result;
}
