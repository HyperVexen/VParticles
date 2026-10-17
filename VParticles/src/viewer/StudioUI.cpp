#include "StudioUI.h"
#include "StudioTheme.h"
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <fstream>
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

    // Load clean desktop fonts and apply high-end studio theme
    setupStudioFonts(io);
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

    anyToolbarHovered_ = false;

    renderDockspace(state, camera);
    renderViewportPanel(state, camera, viewportTexture, outVpWidth, outVpHeight);

    // If viewport is not maximized, render the dockable tool panels
    if (!state.isMaximizedViewport) {
        renderOutlinerPanel(state);
        renderInspectorPanel(state);
        renderTimelinePanel(state);
        renderDiagnosticsPanel(state);
    }
}

void StudioUI::renderDockspace(StudioState& state, StudioCamera& camera) {
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
            if (ImGui::MenuItem("Reset Layout", nullptr)) {
                defaultLayoutApplied_ = false;
            }
            if (ImGui::MenuItem("Export Benchmark CSV...", "Ctrl+E")) {
                state.exportBenchmarkRequested = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Esc")) {
                glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Camera")) {
            if (ImGui::MenuItem("Front View", "Numpad 1")) camera.setSnapView(0.0f, 0.0f);
            if (ImGui::MenuItem("Back View", "Ctrl+Numpad 1")) camera.setSnapView(180.0f, 0.0f);
            if (ImGui::MenuItem("Right View", "Numpad 3")) camera.setSnapView(90.0f, 0.0f);
            if (ImGui::MenuItem("Left View", "Ctrl+Numpad 3")) camera.setSnapView(-90.0f, 0.0f);
            if (ImGui::MenuItem("Top View", "Numpad 7")) camera.setSnapView(0.0f, 89.9f);
            if (ImGui::MenuItem("Bottom View", "Ctrl+Numpad 7")) camera.setSnapView(0.0f, -89.9f);
            ImGui::Separator();
            bool ortho = camera.isOrthographic();
            if (ImGui::MenuItem("Orthographic Projection", "Numpad 5", &ortho)) {
                camera.setOrthographic(ortho);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Frame All", "Home")) {
                camera.reset({0.0f, 0.35f, 0.0f}, 3.0f, 25.0f, 22.0f);
            }
            if (ImGui::MenuItem("Reset Default View", "C")) {
                camera.reset({0.0f, 0.35f, 0.0f}, 2.5f, 25.0f, 22.0f);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Infinite Ground Grid", "G", &state.showInfiniteGrid);
            ImGui::MenuItem("Telemetry HUD", "H", &state.showStatsOverlay);
            ImGui::MenuItem("Colormap Legend", nullptr, &state.showColormapLegend);
            ImGui::MenuItem("Transform Gizmo", nullptr, &state.showGizmo);
            ImGui::Separator();
            ImGui::MenuItem("Maximize Viewport Mode", "Ctrl+Space", &state.isMaximizedViewport);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Shading")) {
            if (ImGui::MenuItem("Material Tint", "1", state.sandColorMode == 0)) state.sandColorMode = 0;
            if (ImGui::MenuItem("Velocity Heatmap", "2", state.sandColorMode == 1)) state.sandColorMode = 1;
            if (ImGui::MenuItem("Sleeping State (Blue/Gold)", "3", state.sandColorMode == 2)) state.sandColorMode = 2;
            if (ImGui::MenuItem("Height / Elevation", "4", state.sandColorMode == 3)) state.sandColorMode = 3;
            ImGui::Separator();
            if (ImGui::MenuItem("Shaded 3D Spheres", nullptr, state.renderStyle == 0)) state.renderStyle = 0;
            if (ImGui::MenuItem("Flat Circular Discs", nullptr, state.renderStyle == 1)) state.renderStyle = 1;
            if (ImGui::MenuItem("Diagnostic Wire Rings", nullptr, state.renderStyle == 2)) state.renderStyle = 2;
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

void StudioUI::renderColormapLegend(const StudioState& state, ImVec2 pos, ImVec2 size) {
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Background pill
    ImVec2 bgMin = pos;
    ImVec2 bgMax = ImVec2(pos.x + size.x, pos.y + size.y);
    drawList->AddRectFilled(bgMin, bgMax, IM_COL32(18, 20, 24, 220), 8.0f);
    drawList->AddRect(bgMin, bgMax, IM_COL32(45, 48, 55, 255), 8.0f);

    float barX = pos.x + 10.0f;
    float barY = pos.y + 24.0f;
    float barW = size.x - 20.0f;
    float barH = 12.0f;

    // Title label
    if (state.sandColorMode == 1) {
        drawList->AddText(ImVec2(barX, pos.y + 6.0f), IM_COL32(230, 230, 235, 255), "Velocity Magnitude (m/s)");
    } else {
        drawList->AddText(ImVec2(barX, pos.y + 6.0f), IM_COL32(230, 230, 235, 255), "Height Elevation (m)");
    }

    // Gradient bar (Turbo / Viridis approximation)
    const int numSegments = 32;
    float segW = barW / static_cast<float>(numSegments);
    for (int i = 0; i < numSegments; ++i) {
        float t0 = static_cast<float>(i) / numSegments;
        float t1 = static_cast<float>(i + 1) / numSegments;

        // Color ramp: Blue (0.0) -> Cyan (0.25) -> Green (0.5) -> Amber (0.75) -> Red (1.0)
        auto getRampCol = [](float t) -> ImU32 {
            float r = std::clamp(1.5f * t - 0.2f, 0.0f, 1.0f);
            float g = std::clamp(1.8f * (1.0f - std::abs(t - 0.5f) * 2.0f), 0.0f, 1.0f);
            float b = std::clamp(1.5f * (1.0f - t) - 0.1f, 0.0f, 1.0f);
            return IM_COL32(static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255), 255);
        };

        ImU32 c0 = getRampCol(t0);
        ImU32 c1 = getRampCol(t1);
        drawList->AddRectFilledMultiColor(
            ImVec2(barX + i * segW, barY),
            ImVec2(barX + (i + 1) * segW, barY + barH),
            c0, c1, c1, c0
        );
    }
    drawList->AddRect(ImVec2(barX, barY), ImVec2(barX + barW, barY + barH), IM_COL32(60, 65, 75, 255));

    // Numerical range readouts
    char minBuf[32];
    char maxBuf[32];
    if (state.sandColorMode == 1) {
        snprintf(minBuf, sizeof(minBuf), "0.0 m/s");
        snprintf(maxBuf, sizeof(maxBuf), "%.1f m/s", state.velocityRangeMax);
    } else {
        snprintf(minBuf, sizeof(minBuf), "0.0 m");
        snprintf(maxBuf, sizeof(maxBuf), "1.0 m");
    }

    drawList->AddText(ImVec2(barX, barY + barH + 4.0f), IM_COL32(160, 165, 175, 255), minBuf);
    float maxTextW = ImGui::CalcTextSize(maxBuf).x;
    drawList->AddText(ImVec2(barX + barW - maxTextW, barY + barH + 4.0f), IM_COL32(160, 165, 175, 255), maxBuf);
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

    // 2. Active Object Visual Indicator (3D Crosshairs / Wireframe Box at Entity Origin)
    Float3 targetOrigin = {0.0f, 0.0f, 0.0f};
    bool hasTargetOrigin = false;
    if (state.selectedType == SelectedEntityType::ColliderMovingPlow) {
        targetOrigin = state.granularSettings.kinematicOrigin;
        hasTargetOrigin = true;
    } else if (state.selectedType == SelectedEntityType::Emitter) {
        targetOrigin = state.emitterPosition;
        hasTargetOrigin = true;
    } else if (state.selectedType == SelectedEntityType::ColliderObstacleSphere) {
        targetOrigin = state.granularSettings.obstacleSphere.center;
        hasTargetOrigin = true;
    }

    if (hasTargetOrigin && !state.isFlyModeActive) {
        Mat4 vp = camera.viewProjectionMatrix(static_cast<float>(outVpWidth) / static_cast<float>(outVpHeight));
        Vec3 p3d = {targetOrigin.x, targetOrigin.y, targetOrigin.z};
        // Project world to screen
        Vec4 clip = vp * Vec4(p3d, 1.0f);
        if (clip.w > 0.01f) {
            Vec3 ndc = {clip.x / clip.w, clip.y / clip.w, clip.z / clip.w};
            if (ndc.z >= -1.0f && ndc.z <= 1.0f) {
                float sx = vpMin.x + (ndc.x * 0.5f + 0.5f) * viewportPanelSize.x;
                float sy = vpMin.y + (-ndc.y * 0.5f + 0.5f) * viewportPanelSize.y;
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                drawList->AddCircle(ImVec2(sx, sy), 8.0f, IM_COL32(245, 158, 11, 200), 16, 1.5f);
                drawList->AddLine(ImVec2(sx - 12.0f, sy), ImVec2(sx + 12.0f, sy), IM_COL32(245, 158, 11, 220), 1.5f);
                drawList->AddLine(ImVec2(sx, sy - 12.0f), ImVec2(sx, sy + 12.0f), IM_COL32(245, 158, 11, 220), 1.5f);
            }
        }
    }

    // 3. ImGuizmo 3D Transform Manipulation
    isUsingGizmo_ = false;
    if (state.showGizmo && state.currentGizmoOperation != static_cast<ImGuizmo::OPERATION>(0) && !state.isFlyModeActive) {
        ImGuizmo::SetOrthographic(camera.isOrthographic());
        ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
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

    // 4. Overlaid Viewport Toolbars
    // --- Top-Left Toolbar: Gizmo Mode & Snapping Pill ---
    {
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + 12.0f, vpMin.y + 12.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.09f, 0.11f, 0.88f));
        ImGui::BeginChild("VpToolbarLeft", ImVec2(380.0f, 36.0f), true, ImGuiWindowFlags_NoScrollbar);
        if (ImGui::IsWindowHovered()) anyToolbarHovered_ = true;

        bool isTrans = (state.currentGizmoOperation == ImGuizmo::TRANSLATE);
        bool isRot   = (state.currentGizmoOperation == ImGuizmo::ROTATE);
        bool isScale = (state.currentGizmoOperation == ImGuizmo::SCALE);
        bool isNone  = (state.currentGizmoOperation == static_cast<ImGuizmo::OPERATION>(0));

        if (isTrans) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" W Move ")) state.currentGizmoOperation = ImGuizmo::TRANSLATE;
        if (isTrans) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isRot) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" E Rotate ")) state.currentGizmoOperation = ImGuizmo::ROTATE;
        if (isRot) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isScale) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" R Scale ")) state.currentGizmoOperation = ImGuizmo::SCALE;
        if (isScale) ImGui::PopStyleColor();

        ImGui::SameLine();
        if (isNone) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(" Q View ")) state.currentGizmoOperation = static_cast<ImGuizmo::OPERATION>(0);
        if (isNone) ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        bool isWorld = (state.currentGizmoMode == ImGuizmo::WORLD);
        if (ImGui::Button(isWorld ? "World" : "Local")) {
            state.currentGizmoMode = isWorld ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        }

        ImGui::SameLine();
        if (state.useSnap) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.961f, 0.620f, 0.043f, 0.7f));
        if (ImGui::Button("Snap")) state.useSnap = !state.useSnap;
        if (state.useSnap) ImGui::PopStyleColor();

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    // --- Top-Center Toolbar: Shading, Render Style & Particle Display Scale ---
    {
        float centerWidth = 530.0f;
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + (viewportPanelSize.x - centerWidth) * 0.5f, vpMin.y + 12.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.09f, 0.11f, 0.88f));
        ImGui::BeginChild("VpToolbarCenter", ImVec2(centerWidth, 36.0f), true, ImGuiWindowFlags_NoScrollbar);
        if (ImGui::IsWindowHovered()) anyToolbarHovered_ = true;

        bool mode0 = (state.sandColorMode == 0);
        bool mode1 = (state.sandColorMode == 1);
        bool mode2 = (state.sandColorMode == 2);
        bool mode3 = (state.sandColorMode == 3);

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
        if (mode3) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.60f, 0.25f, 0.85f, 0.75f));
        if (ImGui::Button(" Height ")) state.sandColorMode = 3;
        if (mode3) ImGui::PopStyleColor();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        // Render Style dropdown
        const char* styles[] = {"Sphere", "Disc", "Wire"};
        ImGui::SetNextItemWidth(75.0f);
        ImGui::Combo("##Style", &state.renderStyle, styles, 3);

        ImGui::SameLine();
        ImGui::Text("Scale:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(70.0f);
        ImGui::SliderFloat("##DisplayScale", &state.displayScale, 0.1f, 3.0f, "%.2fx");

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    // --- Top-Right Toolbar: Viewport Orientation Widget & Projection Mode ---
    if (!state.isFlyModeActive) {
        ImVec2 viewManipPos = ImVec2(vpMin.x + viewportPanelSize.x - 120.0f, vpMin.y + 12.0f);
        Mat4 view = camera.viewMatrix();
        float camDist = camera.distance();
        ImGuizmo::ViewManipulate(view.data(), camDist, viewManipPos, ImVec2(108, 108), 0x14141680);

        // Camera mode pills below view manipulator
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + viewportPanelSize.x - 125.0f, vpMin.y + 126.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.09f, 0.11f, 0.85f));
        ImGui::BeginChild("VpCameraPills", ImVec2(118.0f, 32.0f), true, ImGuiWindowFlags_NoScrollbar);
        if (ImGui::IsWindowHovered()) anyToolbarHovered_ = true;

        bool isOrtho = camera.isOrthographic();
        if (ImGui::Button(isOrtho ? " Ortho " : " Persp ")) {
            camera.toggleOrthographic();
        }
        ImGui::SameLine();
        if (state.isMaximizedViewport) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.055f, 0.647f, 0.812f, 0.7f));
        if (ImGui::Button(state.isMaximizedViewport ? " Rest " : " Max ")) {
            state.isMaximizedViewport = !state.isMaximizedViewport;
        }
        if (state.isMaximizedViewport) ImGui::PopStyleColor();

        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    // --- Colormap Legend Overlay (Velocity or Height mode) ---
    if (state.showColormapLegend && (state.sandColorMode == 1 || state.sandColorMode == 3)) {
        renderColormapLegend(state, ImVec2(vpMin.x + viewportPanelSize.x - 220.0f, vpMin.y + viewportPanelSize.y - 75.0f), ImVec2(208.0f, 62.0f));
    }

    // --- Bottom-Left HUD: Live Simulation Stats Overlay ---
    if (state.showStatsOverlay) {
        ImGui::SetCursorScreenPos(ImVec2(vpMin.x + 12.0f, vpMin.y + viewportPanelSize.y - 120.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.06f, 0.07f, 0.08f, 0.85f));
        ImGui::BeginChild("VpHudOverlay", ImVec2(270.0f, 108.0f), true, ImGuiWindowFlags_NoScrollbar);
        if (ImGui::IsWindowHovered()) anyToolbarHovered_ = true;

        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "FPS: %.1f (%.2f ms)", state.currentFps, state.lastFrameTimeMs);
        ImGui::Text("Sim: %.3f ms | Gather: %.2f ms | GL: %.2f ms", state.lastSimTimeMs, state.lastGatherTimeMs, state.lastRenderTimeMs);
        ImGui::Separator();
        float asleepPct = state.liveAliveGrains > 0 ? (100.0f * state.liveAsleepGrains / state.liveAliveGrains) : 0.0f;
        ImGui::Text("Grains: %u | Asleep: %.1f%%", state.liveAliveGrains, asleepPct);
        ImGui::Text("Awake: %u | Target: (%.1f, %.1f, %.1f)", state.liveAliveGrains - state.liveAsleepGrains,
                    camera.target().x, camera.target().y, camera.target().z);

        if (state.isFlyModeActive) {
            ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "FREE-FLY (Hold RMB: Look | WASD: Move)");
        }

        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::End();
}

void StudioUI::renderOutlinerPanel(StudioState& state) {
    ImGui::Begin("Scene Outliner");

    // Search filter input
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##OutlinerFilter", "Search scene hierarchy...", state.outlinerFilter, sizeof(state.outlinerFilter));
    ImGui::Separator();
    ImGui::Spacing();

    std::string filter = state.outlinerFilter;
    std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);

    auto matchesFilter = [&](const char* name) -> bool {
        if (filter.empty()) return true;
        std::string n = name;
        std::transform(n.begin(), n.end(), n.begin(), ::tolower);
        return n.find(filter) != std::string::npos;
    };

    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.14f, 0.15f, 0.18f, 1.0f));

    // 1. Simulation Engine Node
    if (matchesFilter("Granular Simulation Engine")) {
        bool engSelected = (state.selectedType == SelectedEntityType::SimulationEngine);
        if (ImGui::Selectable(" [SIM] Granular Solver Engine", engSelected)) {
            state.selectedType = SelectedEntityType::SimulationEngine;
        }
    }

    // 2. Emitters Category
    if (ImGui::TreeNodeEx("  Emitters", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (matchesFilter("Sand Influx Source")) {
            bool emitSelected = (state.selectedType == SelectedEntityType::Emitter);
            if (ImGui::Selectable(state.showEmitters ? "  [o] Sand Influx Source" : "  [-] Sand Influx Source", emitSelected)) {
                state.selectedType = SelectedEntityType::Emitter;
            }
        }
        ImGui::TreePop();
    }

    // 3. Colliders Category
    if (ImGui::TreeNodeEx("  Colliders", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (matchesFilter("Kinematic Bulldozer Plow")) {
            bool plowSelected = (state.selectedType == SelectedEntityType::ColliderMovingPlow);
            if (ImGui::Selectable("  [>] Kinematic Bulldozer Plow", plowSelected)) {
                state.selectedType = SelectedEntityType::ColliderMovingPlow;
            }
        }

        if (matchesFilter("Ground Plane")) {
            bool planeSelected = (state.selectedType == SelectedEntityType::ColliderGroundPlane);
            if (ImGui::Selectable("  [_] Ground Plane (Y=0)", planeSelected)) {
                state.selectedType = SelectedEntityType::ColliderGroundPlane;
            }
        }

        if (matchesFilter("Deflector Sphere")) {
            bool sphereSelected = (state.selectedType == SelectedEntityType::ColliderObstacleSphere);
            if (ImGui::Selectable("  (O) Deflector Sphere", sphereSelected)) {
                state.selectedType = SelectedEntityType::ColliderObstacleSphere;
            }
        }

        if (matchesFilter("Boundary Container Box")) {
            bool boxSelected = (state.selectedType == SelectedEntityType::ColliderContainerBox);
            if (ImGui::Selectable("  [#] Boundary Container Box", boxSelected)) {
                state.selectedType = SelectedEntityType::ColliderContainerBox;
            }
        }
        ImGui::TreePop();
    }

    // 4. Material Physics
    if (ImGui::TreeNodeEx("  Materials & Constitutive", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (matchesFilter("Quartz Sand")) {
            bool matSelected = (state.selectedType == SelectedEntityType::MaterialProperties);
            if (ImGui::Selectable("  [*] Quartz Sand (DEM Grains)", matSelected)) {
                state.selectedType = SelectedEntityType::MaterialProperties;
            }
        }
        ImGui::TreePop();
    }

    // 5. Environment & Gravity
    if (matchesFilter("Gravity & Environment")) {
        bool envSelected = (state.selectedType == SelectedEntityType::EnvironmentForces);
        if (ImGui::Selectable(" [ENV] Gravity & Boundary Conditions", envSelected)) {
            state.selectedType = SelectedEntityType::EnvironmentForces;
        }
    }

    ImGui::PopStyleColor();
    ImGui::End();
}

void StudioUI::renderInspectorPanel(StudioState& state) {
    ImGui::Begin("Inspector");

    ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "PROPERTIES");
    ImGui::Separator();

    if (state.selectedType == SelectedEntityType::ColliderMovingPlow) {
        ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "Kinematic Bulldozer Plow");
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Origin", &state.granularSettings.kinematicOrigin.x, 0.02f);
            ImGui::DragFloat("Radius", &state.granularSettings.kinematicRadius, 0.01f, 0.02f, 1.0f, "%.3f m");
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

        // 1. Particle Display Size (Completely decoupled from physical radius)
        if (ImGui::CollapsingHeader("Particle Display & Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "Visual Size (Does NOT alter simulation physics):");
            ImGui::SliderFloat("Display Scale", &state.displayScale, 0.1f, 4.0f, "%.2fx");
            ImGui::DragFloat("Min Screen Size", &state.minPointSize, 0.5f, 1.0f, 16.0f, "%.1f px");
            ImGui::DragFloat("Max Screen Size", &state.maxPointSize, 1.0f, 16.0f, 256.0f, "%.1f px");

            const char* styles[] = {"Shaded 3D Sphere", "Flat Anti-aliased Disc", "Diagnostic Wire Ring"};
            ImGui::Combo("Render Style", &state.renderStyle, styles, 3);

            const char* colors[] = {"Material Tint", "Velocity Heatmap", "Sleeping State", "Height / Elevation"};
            ImGui::Combo("Colormap", &state.sandColorMode, colors, 4);

            if (state.sandColorMode == 1) {
                ImGui::SliderFloat("Max Velocity Range", &state.velocityRangeMax, 0.5f, 10.0f, "%.1f m/s");
            }
        }

        // 2. Physical Simulation Radius & Contact Dynamics
        GrainMaterial& mat = state.granularSettings.materials[0];

        if (ImGui::CollapsingHeader("Coulomb Friction & Contact Law", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(ImVec4(0.961f, 0.620f, 0.043f, 1.0f), "Physical Collision Parameters:");
            ImGui::SliderFloat("Physical Grain Radius", &state.granularSettings.grainRadius, 0.002f, 0.05f, "%.4f m");
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
        ImGui::TextColored(ImVec4(0.055f, 0.647f, 0.812f, 1.0f), "Gravity & Boundary Conditions");
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

    // Rolling latency graph with min/avg/max readouts
    float avgSim = 0.0f;
    float maxSim = 0.0f;
    if (state.metrics.count > 0) {
        float sum = 0.0f;
        for (int i = 0; i < state.metrics.count; ++i) {
            float v = state.metrics.simMs[i];
            sum += v;
            if (v > maxSim) maxSim = v;
        }
        avgSim = sum / state.metrics.count;
    }

    ImGui::Text("Rolling Latency (ms) [Avg: %.2f ms | Max: %.2f ms]:", avgSim, maxSim);
    ImGui::PlotLines("##SimHistory", state.metrics.simMs, RollingMetrics::kHistorySize,
                     state.metrics.cursor, "Sim Solve Latency (ms)", 0.0f, 25.0f, ImVec2(-1, 60.0f));

    ImGui::Spacing();
    if (ImGui::Button("Export Benchmark CSV", ImVec2(160, 24))) {
        state.exportBenchmarkRequested = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset History", ImVec2(110, 24))) {
        state.metrics.reset();
    }

    ImGui::End();
}

} // namespace vparticles::viewer
