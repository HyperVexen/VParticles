#pragma once

#include "StudioCamera.h"
#include "StudioState.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace vparticles::viewer {

class StudioUI {
public:
    StudioUI() = default;
    ~StudioUI() { shutdown(); }

    bool init(GLFWwindow* window);
    void newFrame();
    void render(StudioState& state, StudioCamera& camera, GLuint viewportTexture, int& outVpWidth, int& outVpHeight);
    void draw();
    void shutdown();

    bool isViewportHovered() const { return viewportHovered_; }
    bool isViewportFocused() const { return viewportFocused_; }
    bool isUsingGizmo() const { return isUsingGizmo_; }
    bool isAnyToolbarHovered() const { return anyToolbarHovered_; }

private:
    void renderDockspace(StudioState& state, StudioCamera& camera);
    void renderViewportPanel(StudioState& state, StudioCamera& camera, GLuint viewportTexture, int& outVpWidth, int& outVpHeight);
    void renderOutlinerPanel(StudioState& state);
    void renderInspectorPanel(StudioState& state);
    void renderTimelinePanel(StudioState& state);
    void renderDiagnosticsPanel(StudioState& state);
    void renderColormapLegend(const StudioState& state, ImVec2 pos, ImVec2 size);

    bool initialized_ = false;
    bool viewportHovered_ = false;
    bool viewportFocused_ = false;
    bool isUsingGizmo_ = false;
    bool anyToolbarHovered_ = false;
    bool defaultLayoutApplied_ = false;
};

} // namespace vparticles::viewer
