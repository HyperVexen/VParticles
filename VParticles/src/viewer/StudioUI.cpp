#include "StudioUI.h"
#include "StudioTheme.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <iomanip>
#include <sstream>

namespace vparticles::viewer {

bool StudioUI::init(GLFWwindow* window) {
    if (initialized_) return true;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Apply custom JangaFX-inspired charcoal/slate styling
    applyStudioTheme();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        return false;
    }
    if (!ImGui_ImplOpenGL3_Init("#version 450")) {
        return false;
    }

    initialized_ = true;
    return true;
}

void StudioUI::newFrame() {
    if (!initialized_) return;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();
}

void StudioUI::draw() {
    if (!initialized_) return;

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void StudioUI::shutdown() {
    if (!initialized_) return;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    initialized_ = false;
}

void StudioUI::render(StudioState& state, StudioCamera& camera, GLuint viewportTexture, int& outVpWidth, int& outVpHeight) {
    if (!initialized_) return;

    renderDockspace(state);
    renderViewportPanel(state, camera, viewportTexture, outVpWidth, outVpHeight);
    renderOutlinerPanel(state);
    renderInspectorPanel(state);
    renderTimelinePanel(state);
    renderDiagnosticsPanel(state);
}

void StudioUI::renderDockspace(StudioState& state) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                   ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_MenuBar;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("StudioDockSpaceWindow", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    // --- Main Menu Bar ---
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Reset Layout")) {
                defaultLayoutApplied_ = false;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Esc")) {
                glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Infinite Grid", nullptr, &state.showInfiniteGrid);
            ImGui::MenuItem("Stats HUD", nullptr, &state.showStatsOverlay);
            ImGui::MenuItem("Transform Gizmo", nullptr, &state.showGizmo);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Shading")) {
            if (ImGui::MenuItem("Material Tint", "1", state.sandColorMode == 0)) state.sandColorMode = 0;
            if (ImGui::MenuItem("Velocity Heatmap", "2", state.sandColorMode == 1)) state.sandColorMode = 1;
            if (ImGui::MenuItem("Sleeping State (Blue/Amber)", "3", state.sandColorMode == 2)) state.sandColorMode = 2;
            ImGui::EndMenu();
        }

        // Right-aligned engine status branding
        std::string brand = "VParticles Studio | Granular DEM Core";
        float brandWidth = ImGui::CalcTextSize(brand.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - brandWidth - 20.0f);
        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "%s", brand.c_str());

        ImGui::EndMenuBar();
    }

    // Central DockSpace Node
    ImGuiID dockspaceId = ImGui::GetID("VParticlesStudioDockSpace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

    // Initial Docking Layout Builder (First Run)
    if (!defaultLayoutApplied_) {
        defaultLayoutApplied_ = true;
        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->Size);

        ImGuiID dockMain = dockspaceId;
        ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down, 0.22f, nullptr, &dockMain);
        ImGuiID dockLeft   = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left, 0.20f, nullptr, &dockMain);
        ImGuiID dockRight  = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right, 0.28f, nullptr, &dockMain);

        ImGui::DockBuilderDockWindow("Viewport", dockMain);
        ImGui::DockBuilderDockWindow("Scene Outliner", dockLeft);
        ImGui::DockBuilderDockWindow("Inspector", dockRight);
        ImGui::DockBuilderDockWindow("Timeline & Transport", dockBottom);
        ImGui::DockBuilderDockWindow("Diagnostics & Telemetry", dockBottom);

        ImGui::DockBuilderFinish(dockspaceId);
    }

    ImGui::End();
}

void StudioUI::renderViewportPanel(StudioState& state, StudioCamera& camera, GLuint viewportTexture, int& outVpWidth, int& outVpHeight) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();

    viewportHovered_ = ImGui::IsWindowHovered();
    viewportFocused_ = ImGui::IsWindowFocused();

    ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
    outVpWidth  = std::max(1, static_cast<int>(viewportPanelSize.x));
    outVpHeight = std::max(1, static_cast<int>(viewportPanelSize.y));

    ImVec2 windowPos = ImGui::GetWindowPos();
    ImVec2 contentMin = ImGui::GetWindowContentRegionMin();
    ImVec2 vpMin = ImVec2(windowPos.x + contentMin.x, windowPos.y + contentMin.y);

    // 1. Render 3D OpenGL Viewport Texture
    if (viewportTexture != 0) {
        ImGui::Image(static_cast<ImTextureID>(viewportTexture),
                     viewportPanelSize, ImVec2(0, 1), ImVec2(1, 0));
    }

    // 2. ImGuizmo 3D Transform Manipulation
    isUsingGizmo_ = false;
    if (state.showGizmo && state.currentGizmoOperation != static_cast<ImGuizmo::OPERATION>(0) && !state.isFlyModeActive) {
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(vpMin.x, vpMin.y, viewportPanelSize.x, viewportPanelSize.y);

        Mat4 view = camera.viewMatrix();
        Mat4 proj = camera.projectionMatrix(static_cast<float>(outVpWidth) / static_cast<float>(outVpHeight));

        float matrix[16] = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        bool hasTransformTarget = false;
        Float3* targetPos = nullptr;

        if (state.selectedType == SelectedEntityType::ColliderMovingPlow) {
            targetPos = &state.granularSettings.kinematicOrigin;
            hasTransformTarget = true;
        } else if (state.selectedType == SelectedEntityType::Emitter) {
            targetPos = &state.emitterPosition;
            hasTransformTarget = true;
        } else if (state.selectedType == SelectedEntityType::ColliderObstacleSphere) {
            targetPos = &state.granularSettings.obstacleSphere.center;
            hasTransformTarget = true;
        }

        if (hasTransformTarget && targetPos) {
            matrix[12] = targetPos->x;
            matrix[13] = targetPos->y;
            matrix[14] = targetPos->z;

            float snapValues[3] = {state.snapTranslation, state.snapTranslation, state.snapTranslation};
            if (state.currentGizmoOperation == ImGuizmo::ROTATE) {
                snapValues[0] = snapValues[1] = snapValues[2] = state.snapRotation;
            } else if (state.currentGizmoOperation == ImGuizmo::SCALE) {
                snapValues[0] = snapValues[1] = snapValues[2] = state.snapScale;
            }

            ImGuizmo::Manipulate(view.data(), proj.data(),
                                 state.currentGizmoOperation, state.currentGizmoMode,
                                 matrix, nullptr, state.useSnap ? snapValues : nullptr);

            if (ImGuizmo::IsUsing()) {
                isUsingGizmo_ = true;
                targetPos->x = matrix[12];
                targetPos->y = matrix[13];
                targetPos->z = matrix[14];
            }
        }
    }

    // 3. JangaFX-Inspired Floating Viewport Toolbars (Overlaid on top of 3D canvas)
    // --- Top-Left Toolbar: Gizmo Mode & Snapping Pill ---
    {
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + 12.0f, vpMin.y + 12.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.09f, 0.11f, 0.85f));
        ImGui::BeginChild("VpToolbarLeft", ImVec2(340.0f, 36.0f), true, ImGuiWindowFlags_NoScrollbar);

        // Gizmo Operation Selectors
        bool isTrans = (state.currentGizmoOperation == ImGuizmo::TRANSLATE);
        bool isRot   = (state.currentGizmoOperation == ImGuizmo::ROTATE);
        bool isScale = (state.currentGizmoOperation == ImGuizmo::SCALE);
        bool isNone  = (state.currentGizmoOperation == static_cast<ImGuizmo::OPERATION>(0));

        if (isTrans) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" W | Move ")) state.currentGizmoOperation = ImGuizmo::TRANSLATE;
        if (isTrans) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isRot) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" E | Rotate ")) state.currentGizmoOperation = ImGuizmo::ROTATE;
        if (isRot) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isScale) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" R | Scale ")) state.currentGizmoOperation = ImGuizmo::SCALE;
        if (isScale) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isNone) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" Q | View ")) state.currentGizmoOperation = static_cast<ImGuizmo::OPERATION>(0);
        if (isNone) ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        // World / Local Space Toggle
        bool isWorld = (state.currentGizmoMode == ImGuizmo::WORLD);
        if (ImGui::Button(isWorld ? "World" : "Local")) {
            state.currentGizmoMode = isWorld ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        }

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    // --- Top-Center Toolbar: Shading Mode Pill ---
    {
        float centerWidth = 260.0f;
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + (viewportPanelSize.x - centerWidth) * 0.5f, vpMin.y + 12.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.09f, 0.11f, 0.85f));
        ImGui::BeginChild("VpToolbarCenter", ImVec2(centerWidth, 36.0f), true, ImGuiWindowFlags_NoScrollbar);

        bool mode0 = (state.sandColorMode == 0);
        bool mode1 = (state.sandColorMode == 1);
        bool mode2 = (state.sandColorMode == 2);

        if (mode0) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.961f, 0.620f, 0.043f, 0.75f));
        if (ImGui::Button(" Tint ")) state.sandColorMode = 0;
        if (mode0) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (mode1) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.90f, 0.25f, 0.25f, 0.75f));
        if (ImGui::Button(" Velocity ")) state.sandColorMode = 1;
        if (mode1) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (mode2) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.75f));
        if (ImGui::Button(" Sleeping ")) state.sandColorMode = 2;
        if (mode2) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (ImGui::Button(" Grid ")) state.showInfiniteGrid = !state.showInfiniteGrid;

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    // --- Top-Right Toolbar: Viewport Orientation Widget ---
    if (!state.isFlyModeActive) {
        ImVec2 viewManipPos = ImVec2(vpMin.x + viewportPanelSize.x - 120.0f, vpMin.y + 12.0f);
        Mat4 view = camera.viewMatrix();
        float camDist = camera.distance();
        ImGuizmo::ViewManipulate(view.data(), camDist, viewManipPos, ImVec2(108, 108), 0x14141680);
    }

    // --- Bottom-Left HUD: Live Simulation Stats Overlay ---
    if (state.showStatsOverlay) {
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + 12.0f, vpMin.y + viewportPanelSize.y - 105.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.06f, 0.07f, 0.08f, 0.82f));
        ImGui::BeginChild("VpHudOverlay", ImVec2(240.0f, 92.0f), true, ImGuiWindowFlags_NoScrollbar);

        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "FPS: %.1f (%.2f ms)", state.currentFps, state.lastFrameTimeMs);
        ImGui::Text("Sim: %.3f ms | Render: %.2f ms", state.lastSimTimeMs, state.lastRenderTimeMs);
        ImGui::Separator();
        float asleepPct = state.liveAliveGrains > 0 ? (100.0f * state.liveAsleepGrains / state.liveAliveGrains) : 0.0f;
        ImGui::Text("Grains: %u | Asleep: %.1f%%", state.liveAliveGrains, asleepPct);
        ImGui::Text("Active Awake: %u grains", state.liveAliveGrains - state.liveAsleepGrains);

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::End();
}

void StudioUI::renderOutlinerPanel(StudioState& state) {
    ImGui::Begin("Scene Outliner");

    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.14f, 0.15f, 0.18f, 1.0f));

    // 1. Simulation Engine Node
    bool engSelected = (state.selectedType == SelectedEntityType::SimulationEngine);
    if (ImGui::Selectable("  Granular Simulation Engine", engSelected)) {
        state.selectedType = SelectedEntityType::SimulationEngine;
    }

    // 2. Emitters Category
    if (ImGui::TreeNodeEx("  Emitters", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool emitSelected = (state.selectedType == SelectedEntityType::Emitter);
        if (ImGui::Selectable("   Sand Influx Source", emitSelected)) {
            state.selectedType = SelectedEntityType::Emitter;
        }
        ImGui::TreePop();
    }

    // 3. Colliders Category
    if (ImGui::TreeNodeEx("  Colliders", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool plowSelected = (state.selectedType == SelectedEntityType::ColliderMovingPlow);
        if (ImGui::Selectable("   Kinematic Bulldozer Plow", plowSelected)) {
            state.selectedType = SelectedEntityType::ColliderMovingPlow;
        }

        bool planeSelected = (state.selectedType == SelectedEntityType::ColliderGroundPlane);
        if (ImGui::Selectable("   Ground Plane (Y=0)", planeSelected)) {
            state.selectedType = SelectedEntityType::ColliderGroundPlane;
        }

        bool sphereSelected = (state.selectedType == SelectedEntityType::ColliderObstacleSphere);
        if (ImGui::Selectable("   Deflector Sphere", sphereSelected)) {
            state.selectedType = SelectedEntityType::ColliderObstacleSphere;
        }

        bool boxSelected = (state.selectedType == SelectedEntityType::ColliderContainerBox);
        if (ImGui::Selectable("   Boundary Container Box", boxSelected)) {
            state.selectedType = SelectedEntityType::ColliderContainerBox;
        }
        ImGui::TreePop();
    }

    // 4. Material Physics
    if (ImGui::TreeNodeEx("  Materials & Constitutive", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool matSelected = (state.selectedType == SelectedEntityType::MaterialProperties);
        if (ImGui::Selectable("   Quartz Sand (Dry/Damp/Clumps)", matSelected)) {
            state.selectedType = SelectedEntityType::MaterialProperties;
        }
        ImGui::TreePop();
    }

    // 5. Environment & Gravity
    bool envSelected = (state.selectedType == SelectedEntityType::EnvironmentForces);
    if (ImGui::Selectable("  Gravity & Environment", envSelected)) {
        state.selectedType = SelectedEntityType::EnvironmentForces;
    }

    ImGui::PopStyleColor();
    ImGui::End();
}

void StudioUI::renderInspectorPanel(StudioState& state) {
    ImGui::Begin("Inspector");

    // Header displaying current selection
    ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "PROPERTIES");
    ImGui::Separator();

    if (state.selectedType == SelectedEntityType::ColliderMovingPlow) {
        ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "Kinematic Bulldozer Plow");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Origin", &state.granularSettings.kinematicOrigin.x, 0.02f);
            ImGui::DragFloat("Radius", &state.granularSettings.kinematicRadius, 0.01f, 0.02f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Motion Dynamics", ImGuiTreeNodeFlags_DefaultOpen)) {
            bool enabled = (state.granularSettings.kinematicColliderEnabled != 0);
            if (ImGui::Checkbox("Enable Kinematic Motion", &enabled)) {
                state.granularSettings.kinematicColliderEnabled = enabled ? 1 : 0;
            }

            const char* types[] = {"Linear Oscillating Bulldozer", "Circular Rotating Churner"};
            ImGui::Combo("Trajectory", &state.granularSettings.kinematicColliderType, types, 2);

            ImGui::SliderFloat("Speed / Frequency", &state.granularSettings.kinematicSpeed, 0.1f, 10.0f, "%.2f rad/s");
            ImGui::SliderFloat("Sweep Amplitude", &state.granularSettings.kinematicAmplitude, 0.05f, 1.5f, "%.2f m");
        }
    } else if (state.selectedType == SelectedEntityType::Emitter) {
        ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "Sand Influx Source");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Position", &state.emitterPosition.x, 0.02f);
            ImGui::DragFloat3("Rotation", &state.emitterRotation.x, 1.0f);
            ImGui::DragFloat3("Scale", &state.emitterScale.x, 0.05f);
        }

        if (ImGui::CollapsingHeader("Influx Control", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Active Emitter", &state.emitterActive);
            const char* shapes[] = {"Point Source", "Sphere Volume", "Box Extents", "Cylinder Column"};
            ImGui::Combo("Shape", &state.emitterShapeType, shapes, 4);
            ImGui::DragFloat("Spawn Rate", &state.emitterSpawnRate, 1000.0f, 0.0f, 500000.0f, "%.0f grains/s");
        }
    } else if (state.selectedType == SelectedEntityType::MaterialProperties) {
        ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "Granular Material Physics");
        ImGui::Spacing();

        GrainMaterial& mat = state.granularSettings.materials[0];

        if (ImGui::CollapsingHeader("Coulomb Friction & Contact", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Grain Radius", &state.granularSettings.grainRadius, 0.002f, 0.05f, "%.4f m");
            ImGui::SliderFloat("Radius Jitter", &state.granularSettings.radiusJitter, 0.0f, 0.35f, "%.2f");
            ImGui::SliderFloat("Density", &mat.density, 800.0f, 3500.0f, "%.0f kg/m3");
            ImGui::SliderFloat("Static Friction", &mat.staticFriction, 0.0f, 1.2f, "%.2f");
            ImGui::SliderFloat("Dynamic Friction", &mat.dynamicFriction, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Restitution", &mat.restitution, 0.0f, 0.8f, "%.2f");
            ImGui::SliderFloat("Damping", &mat.damping, 0.0f, 10.0f, "%.2f");
        }

        if (ImGui::CollapsingHeader("Capillary Cohesion & XPBD Bonds", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Capillary Meniscus", &state.granularSettings.defaultCohesion, 0.0f, 0.25f, "%.3f N");
            ImGui::SliderFloat("Bond Tensile Break Force", &state.granularSettings.bondBreakForce, 0.0f, 100.0f, "%.1f N");
            ImGui::SliderFloat("Bond Compliance", &state.granularSettings.bondCompliance, 0.0f, 1e-3f, "%.2e m/N");
            ImGui::SliderFloat("Clump Cluster Radius", &state.granularSettings.clumpRadius, 0.01f, 0.15f, "%.3f m");
        }
    } else if (state.selectedType == SelectedEntityType::SimulationEngine) {
        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "Simulation Solver Configuration");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Substepping & Solver Passes", ImGuiTreeNodeFlags_DefaultOpen)) {
            int ss = static_cast<int>(state.granularSettings.substeps);
            if (ImGui::SliderInt("Substeps", &ss, 1, 16)) {
                state.granularSettings.substeps = static_cast<uint32_t>(ss);
            }

            int iters = static_cast<int>(state.granularSettings.iterations);
            if (ImGui::SliderInt("GS Iterations", &iters, 1, 12)) {
                state.granularSettings.iterations = static_cast<uint32_t>(iters);
            }

            ImGui::SliderFloat("Fixed dt", &state.granularSettings.fixedDt, 0.004f, 0.0333f, "%.4f s");
            ImGui::SliderFloat("Verlet Skin Ratio", &state.granularSettings.skinRatio, 0.05f, 0.50f, "%.2f");
        }

        if (ImGui::CollapsingHeader("Sleeping Optimization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Grain Sleeping", &state.granularSettings.enableSleeping);
            ImGui::SliderFloat("Sleep Speed Threshold", &state.granularSettings.sleepSpeedThreshold, 0.005f, 0.10f, "%.3f m/s");
            int sf = static_cast<int>(state.granularSettings.sleepFrames);
            if (ImGui::SliderInt("Sleep Frames", &sf, 2, 30)) {
                state.granularSettings.sleepFrames = static_cast<uint32_t>(sf);
            }
        }
    } else if (state.selectedType == SelectedEntityType::EnvironmentForces) {
        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "Gravity & Boundary Planes");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Gravity", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Gravity Vector", &state.granularSettings.gravity.x, 0.2f);
        }

        if (ImGui::CollapsingHeader("Ground Plane", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Normal", &state.granularSettings.groundPlane.normal.x, 0.05f);
            ImGui::SliderFloat("Ground Friction", &state.granularSettings.groundPlane.friction, 0.0f, 1.0f, "%.2f");
            ImGui::SliderFloat("Ground Restitution", &state.granularSettings.groundPlane.bounce, 0.0f, 0.9f, "%.2f");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "HOUDINI SAND SCENARIOS");
    ImGui::Spacing();

    const char* scenarios[] = {
        "1. Pouring Sand Pile (Angle of Repose)",
        "2. Column Collapse (Dam-break Avalanche)",
        "3. Conical Funnel (Hourglass Flow)",
        "4. Sand Dune (Slipface Avalanche)",
        "5. Damp Sand Cliff (Capillary Cohesion)",
        "6. Bonded Block (XPBD Cantilever Beam)",
        "7. Destructible Clumps (Impact Crumbling)"
    };

    if (ImGui::Combo("Preset", &state.currentScenarioIndex, scenarios, 7)) {
        state.scenarioSwitchRequested = true;
    }
    if (ImGui::Button("Load Scenario Preset", ImVec2(-1, 28))) {
        state.scenarioSwitchRequested = true;
    }

    ImGui::End();
}

void StudioUI::renderTimelinePanel(StudioState& state) {
    ImGui::Begin("Timeline & Transport");

    // Transport Buttons (Center Row)
    float btnWidth = 32.0f;
    float rowWidth = btnWidth * 5 + 40.0f + 160.0f;
    ImGui::SetCursorPosX((ImGui::GetWindowWidth() - rowWidth) * 0.5f);

    if (ImGui::Button(" |< ", ImVec2(btnWidth, 24))) {
        state.currentFrame = 0;
        state.resetRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(" < ", ImVec2(btnWidth, 24))) {
        state.stepRequest = -1;
    }
    ImGui::SameLine();

    bool isPlay = state.isPlaying;
    if (isPlay) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.8f));
    if (ImGui::Button(isPlay ? " || " : " > ", ImVec2(btnWidth + 6.0f, 24))) {
        state.isPlaying = !state.isPlaying;
    }
    if (isPlay) ImGui::PopStyleColor();

    ImGui::SameLine();
    if (ImGui::Button(" >| ", ImVec2(btnWidth, 24))) {
        state.stepRequest = 1;
    }
    ImGui::SameLine();

    if (state.isLooping) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.961f, 0.620f, 0.043f, 0.7f));
    if (ImGui::Button(" [L] ", ImVec2(btnWidth + 6.0f, 24))) {
        state.isLooping = !state.isLooping;
    }
    if (state.isLooping) ImGui::PopStyleColor();

    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    const char* fpsModes[] = {"24 FPS", "30 FPS", "60 FPS", "120 FPS", "Free / Max"};
    ImGui::Combo("##FpsLock", &state.targetFpsMode, fpsModes, 5);

    ImGui::SameLine();
    ImGui::Text("  Frame: %u / %u  (%.2f s)", state.currentFrame, state.maxFrames,
                static_cast<float>(state.currentFrame) * state.granularSettings.fixedDt);

    // Frame Scrubber Slider
    ImGui::Spacing();
    ImGui::PushItemWidth(-1);
    int frameVal = static_cast<int>(state.currentFrame);
    if (ImGui::SliderInt("##TimelineSlider", &frameVal, 0, static_cast<int>(state.maxFrames), "Frame %d")) {
        state.currentFrame = static_cast<uint32_t>(frameVal);
    }
    ImGui::PopItemWidth();

    ImGui::End();
}

void StudioUI::renderDiagnosticsPanel(StudioState& state) {
    ImGui::Begin("Diagnostics & Telemetry");

    // Metrics Overview Cards
    ImGui::Columns(4, "MetricsCols", false);

    ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "FPS & FRAME TIME");
    ImGui::Text("%.1f FPS", state.currentFps);
    ImGui::Text("%.2f ms/frame", state.lastFrameTimeMs);
    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "SIMULATION TIME");
    ImGui::Text("%.3f ms solve", state.lastSimTimeMs);
    ImGui::Text("%.3f ms gather", state.lastGatherTimeMs);
    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.2f, 0.85f, 0.4f, 1.0f), "PARTICLES BREAKDOWN");
    ImGui::Text("%u alive", state.liveAliveGrains);
    float sleepPct = state.liveAliveGrains > 0 ? (100.0f * state.liveAsleepGrains / state.liveAliveGrains) : 0.0f;
    ImGui::Text("%u asleep (%.1f%%)", state.liveAsleepGrains, sleepPct);
    ImGui::NextColumn();

    ImGui::TextColored(ImVec4(0.7f, 0.5f, 0.9f, 1.0f), "VRAM & BUFFERS");
    ImGui::Text("%.1f MB VBO", static_cast<float>(state.liveVboMemoryBytes) / (1024.0f * 1024.0f));
    ImGui::Text("%.1f MB Pool", static_cast<float>(state.capacity * 60) / (1024.0f * 1024.0f));
    ImGui::Columns(1);

    ImGui::Separator();
    ImGui::Spacing();

    // Rolling latency graph
    ImGui::Text("Rolling Latency (ms):");
    ImGui::PlotLines("##SimHistory", state.metrics.simMs, RollingMetrics::kHistorySize,
                     state.metrics.cursor, "Sim Solve Latency (ms)", 0.0f, 25.0f, ImVec2(-1, 60.0f));

    ImGui::End();
}

} // namespace vparticles::viewer
