#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "VParticles/ParticleSystem.h"
#include "VParticles/SimulationTypes.h"
#include "CudaGLBridge.h"
#include "GridShader.h"
#include "Math3D.h"
#include "Shaders.h"
#include "StudioCamera.h"
#include "StudioState.h"
#include "StudioTheme.h"
#include "StudioUI.h"
#include "ViewportFramebuffer.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace vparticles;
using namespace vparticles::viewer;

namespace {

struct ViewerConfig {
    uint32_t capacity = 500000;
    int windowWidth = 1600;
    int windowHeight = 900;
    bool vsync = true;
    float pointScale = 1.0f;
    uint32_t maxFrames = 0;
    int samples = 0;

    int sandScenario = 1; // 0: pile, 1: column_collapse, 2: funnel, 3: dune, 4: cliff, 5: bonded_block, 6: clumps
    float cohesion = 0.0f;
    float bondBreakForce = 0.0f;
    float clumpRadius = 0.04f;
    float grainRadius = 0.01f;
    bool kinematicCollider = true;
};

struct AppState {
    StudioCamera camera;
    StudioState state;
    StudioUI ui;
    InfiniteGridRenderer grid;
    ViewportFramebuffer fbo;
    ViewerConfig config;

    GLFWwindow* window = nullptr;
    bool leftMouseDown = false;
    bool rightMouseDown = false;
    bool middleMouseDown = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    int vpWidth = 1280;
    int vpHeight = 720;
};

AppState g_app;

void frameSelectedObject(StudioState& state, StudioCamera& camera) {
    if (state.selectedType == SelectedEntityType::ColliderMovingPlow) {
        camera.frameTarget({state.granularSettings.kinematicOrigin.x,
                            state.granularSettings.kinematicOrigin.y,
                            state.granularSettings.kinematicOrigin.z},
                           std::max(0.8f, state.granularSettings.kinematicRadius * 3.5f));
    } else if (state.selectedType == SelectedEntityType::Emitter) {
        camera.frameTarget({state.emitterPosition.x, state.emitterPosition.y, state.emitterPosition.z}, 1.5f);
    } else if (state.selectedType == SelectedEntityType::ColliderObstacleSphere) {
        camera.frameTarget({state.granularSettings.obstacleSphere.center.x,
                            state.granularSettings.obstacleSphere.center.y,
                            state.granularSettings.obstacleSphere.center.z},
                           std::max(0.8f, state.granularSettings.obstacleSphere.radius * 3.5f));
    } else {
        camera.frameTarget({0.0f, 0.35f, 0.0f}, 2.5f);
    }
}

void framebufferSizeCallback(GLFWwindow* /*window*/, int width, int height) {
    if (width > 0 && height > 0) {
        g_app.config.windowWidth = width;
        g_app.config.windowHeight = height;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    bool canNavigate = g_app.ui.isViewportHovered() && !g_app.ui.isAnyToolbarHovered() && !g_app.ui.isUsingGizmo() && !ImGuizmo::IsOver();

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            if (canNavigate) {
                g_app.leftMouseDown = true;
            }
        } else {
            g_app.leftMouseDown = false;
        }
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            if (g_app.ui.isViewportHovered() && !g_app.ui.isAnyToolbarHovered() && !g_app.ui.isUsingGizmo()) {
                g_app.rightMouseDown = true;
                g_app.state.isFlyModeActive = true;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwGetCursorPos(window, &g_app.lastMouseX, &g_app.lastMouseY);
            }
        } else {
            if (g_app.rightMouseDown) {
                g_app.rightMouseDown = false;
                g_app.state.isFlyModeActive = false;
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
        }
    } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            if (canNavigate) {
                g_app.middleMouseDown = true;
            }
        } else {
            g_app.middleMouseDown = false;
        }
    }

    glfwGetCursorPos(window, &g_app.lastMouseX, &g_app.lastMouseY);
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    float dx = static_cast<float>(xpos - g_app.lastMouseX);
    float dy = static_cast<float>(ypos - g_app.lastMouseY);
    g_app.lastMouseX = xpos;
    g_app.lastMouseY = ypos;

    // 1. JangaFX Free-Fly Mode: look around
    if (g_app.rightMouseDown && g_app.state.isFlyModeActive) {
        g_app.camera.processOrbit(-dx, -dy, 0.16f);
        return;
    }

    // 2. Middle Mouse Button (Blender-standard navigation: MMB orbit, Shift+MMB pan, Ctrl+MMB zoom)
    if (g_app.middleMouseDown) {
        bool shiftDown = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
        bool ctrlDown  = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);

        if (shiftDown) {
            g_app.camera.processPan(dx, dy, 0.003f);
        } else if (ctrlDown) {
            g_app.camera.processZoom(-dy * 0.02f, 0.15f);
        } else {
            g_app.camera.processOrbit(dx, dy, 0.28f);
        }
        return;
    }

    // 3. Left Mouse Button fallback (Alt + LMB / Trackpad navigation)
    if (g_app.leftMouseDown && !g_app.ui.isUsingGizmo() && !ImGuizmo::IsOver()) {
        bool altDown = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS);
        if (altDown) {
            bool shiftDown = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
            if (shiftDown) {
                g_app.camera.processPan(dx, dy, 0.003f);
            } else {
                g_app.camera.processOrbit(dx, dy, 0.28f);
            }
        }
    }
}

void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    if (g_app.rightMouseDown && g_app.state.isFlyModeActive) {
        // Adjust fly speed when scrolling during RMB hold
        g_app.camera.adjustFlySpeed(static_cast<float>(yoffset));
    } else if (g_app.ui.isViewportHovered() && !g_app.ui.isAnyToolbarHovered()) {
        g_app.camera.processZoom(static_cast<float>(yoffset), 0.15f);
    }
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) return;
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    bool ctrlDown = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_SPACE:
        if (ctrlDown) {
            g_app.state.isMaximizedViewport = !g_app.state.isMaximizedViewport;
        } else {
            g_app.state.isPlaying = !g_app.state.isPlaying;
        }
        break;
    case GLFW_KEY_F11:
        g_app.state.isMaximizedViewport = !g_app.state.isMaximizedViewport;
        break;
    case GLFW_KEY_R:
        if (ctrlDown || !g_app.ui.isViewportFocused()) {
            g_app.state.resetRequested = true;
        } else {
            g_app.state.currentGizmoOperation = ImGuizmo::SCALE;
        }
        break;
    case GLFW_KEY_W:
        if (g_app.ui.isViewportFocused() && !g_app.state.isFlyModeActive) {
            g_app.state.currentGizmoOperation = ImGuizmo::TRANSLATE;
        }
        break;
    case GLFW_KEY_E:
        if (g_app.ui.isViewportFocused() && !g_app.state.isFlyModeActive) {
            g_app.state.currentGizmoOperation = ImGuizmo::ROTATE;
        }
        break;
    case GLFW_KEY_Q:
        if (g_app.ui.isViewportFocused() && !g_app.state.isFlyModeActive) {
            g_app.state.currentGizmoOperation = static_cast<ImGuizmo::OPERATION>(0);
        }
        break;
    case GLFW_KEY_C:
        g_app.camera.reset({0.0f, 0.35f, 0.0f}, 2.5f, 25.0f, 22.0f);
        break;
    case GLFW_KEY_G:
        g_app.state.showInfiniteGrid = !g_app.state.showInfiniteGrid;
        break;
    case GLFW_KEY_H:
        g_app.state.showStatsOverlay = !g_app.state.showStatsOverlay;
        break;
    case GLFW_KEY_HOME:
        g_app.camera.reset({0.0f, 0.35f, 0.0f}, 3.0f, 25.0f, 22.0f);
        break;
    case GLFW_KEY_KP_DECIMAL:
        frameSelectedObject(g_app.state, g_app.camera);
        break;
    case GLFW_KEY_KP_1:
        if (ctrlDown) {
            g_app.camera.setSnapView(180.0f, 0.0f); // Back
        } else {
            g_app.camera.setSnapView(0.0f, 0.0f);   // Front
        }
        break;
    case GLFW_KEY_KP_3:
        if (ctrlDown) {
            g_app.camera.setSnapView(-90.0f, 0.0f); // Left
        } else {
            g_app.camera.setSnapView(90.0f, 0.0f);  // Right
        }
        break;
    case GLFW_KEY_KP_7:
        if (ctrlDown) {
            g_app.camera.setSnapView(0.0f, -89.9f); // Bottom
        } else {
            g_app.camera.setSnapView(0.0f, 89.9f);  // Top
        }
        break;
    case GLFW_KEY_KP_5:
        g_app.camera.toggleOrthographic();
        break;
    case GLFW_KEY_1:
        g_app.state.currentScenarioIndex = 0;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_2:
        g_app.state.currentScenarioIndex = 1;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_3:
        g_app.state.currentScenarioIndex = 2;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_4:
        g_app.state.currentScenarioIndex = 3;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_5:
        g_app.state.currentScenarioIndex = 4;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_6:
        g_app.state.currentScenarioIndex = 5;
        g_app.state.scenarioSwitchRequested = true;
        break;
    case GLFW_KEY_7:
        g_app.state.currentScenarioIndex = 6;
        g_app.state.scenarioSwitchRequested = true;
        break;
    default:
        break;
    }
}

const char* getSandScenarioName(int scenario) {
    switch (scenario) {
    case 0: return "Pouring Sand Pile (Angle of Repose)";
    case 1: return "Column Collapse (Dam Break Avalanche)";
    case 2: return "Conical Funnel (Hourglass Flow)";
    case 3: return "Sand Dune (Slipface Avalanche)";
    case 4: return "Damp Sand Cliff (Capillary Cohesion)";
    case 5: return "Bonded Block (XPBD Cantilever Beam)";
    case 6: return "Destructible Clumps (Impact Crumbling)";
    default: return "Custom Sand";
    }
}

void applySandScenario(ParticleSystem& system, int scenario, const ViewerConfig& config, bool enableKinematic) {
    system.reset();
    system.clearEmitters();
    system.setSimulationMode(SimulationMode::Granular);

    GranularSettings settings;
    settings.grainRadius = config.grainRadius > 0.0f ? config.grainRadius : 0.01f;
    settings.enableSleeping = true;
    settings.sleepSpeedThreshold = 0.02f;
    settings.substeps = 4;
    settings.iterations = 4;
    settings.groundPlane.enabled = 1;
    settings.groundPlane.point = {0.0f, 0.0f, 0.0f};
    settings.groundPlane.normal = {0.0f, 1.0f, 0.0f};
    settings.groundPlane.friction = 0.58f;

    if (enableKinematic) {
        settings.kinematicColliderEnabled = 1;
        settings.kinematicColliderType = (scenario % 2); // alternate linear plow / rotating churner
        settings.kinematicOrigin = {0.0f, 0.15f, 0.0f};
        settings.kinematicRadius = 0.10f;
        settings.kinematicSpeed = 2.0f;
        settings.kinematicAmplitude = 0.35f;
    }

    const uint32_t count = std::min(config.capacity, 500000u);

    switch (scenario) {
    case 0: { // Pouring Sand Pile
        system.setGranularSettings(settings);
        system.spawnGranularPouringPile({0.0f, 0.6f, 0.0f}, count, 0.08f, 0.6f, 0.15f);
        break;
    }
    case 1: { // Column Collapse
        system.setGranularSettings(settings);
        system.spawnGranularColumnCollapse({0.0f, 0.0f, 0.0f}, count, 0.02f, 0.12f);
        break;
    }
    case 2: { // Conical Funnel
        settings.containerBox.enabled = 1;
        settings.containerBox.minBounds = {-0.6f, 0.0f, -0.6f};
        settings.containerBox.maxBounds = {0.6f, 1.2f, 0.6f};
        system.setGranularSettings(settings);
        system.spawnGranularFunnel({0.0f, 0.4f, 0.0f}, count, 0.25f, 0.04f, 0.5f, 0.10f);
        break;
    }
    case 3: { // Sand Dune
        system.setGranularSettings(settings);
        system.spawnGranularDune({0.0f, 0.0f, 0.0f}, count, 1.0f, 0.8f, 25.0f, 0.10f);
        break;
    }
    case 4: { // Damp Sand Cliff
        settings.materials[0].cohesion = config.cohesion > 0.0f ? config.cohesion : 60.0f;
        settings.materials[0].cohesionRuptureRatio = 0.15f;
        system.setGranularSettings(settings);
        system.spawnGranularCliff({0.0f, 0.0f, 0.0f}, count, 0.02f, 0.12f);
        break;
    }
    case 5: { // Bonded Cantilever Block
        const float compliance = 1e-5f;
        const float breakForce = config.bondBreakForce > 0.0f ? config.bondBreakForce : 40.0f;
        system.setGranularSettings(settings);
        system.spawnGranularBondedBlock({-0.3f, 0.25f, 0.0f}, std::min(count, 4096u), 0.02f, 0.15f, compliance, breakForce);
        break;
    }
    case 6: { // Destructible Clumps
        const float clumpR = config.clumpRadius > 0.0f ? config.clumpRadius : 0.04f;
        const float breakForce = config.bondBreakForce > 0.0f ? config.bondBreakForce : 25.0f;
        system.setGranularSettings(settings);
        system.spawnGranularClumps({0.0f, 0.6f, 0.0f}, std::min(count, 50000u), 0.02f, clumpR, 1e-5f, breakForce, 0.10f);
        break;
    }
    default:
        break;
    }
}

void exportTelemetryToCsv(const StudioState& state) {
    std::ofstream out("benchmark_telemetry.csv");
    if (!out.is_open()) return;

    out << "SampleIndex,SimSolveMs,GatherMs,RenderMs,TotalMs,FPS\n";
    for (int i = 0; i < state.metrics.count; ++i) {
        int idx = (state.metrics.cursor - state.metrics.count + i + RollingMetrics::kHistorySize) % RollingMetrics::kHistorySize;
        out << i << ","
            << std::fixed << std::setprecision(3) << state.metrics.simMs[idx] << ","
            << state.metrics.gatherMs[idx] << ","
            << state.metrics.renderMs[idx] << ","
            << state.metrics.totalMs[idx] << ","
            << std::setprecision(1) << state.metrics.fps[idx] << "\n";
    }
    std::cout << "[Studio] Exported " << state.metrics.count << " samples to benchmark_telemetry.csv\n";
}

void printHelp() {
    std::cout << "VParticles Simulation Studio - Industrial Granular DEM Engine\n"
              << "Usage: VParticlesViewer [options]\n\n"
              << "General Options:\n"
              << "  --capacity <N>          Particle capacity (default: 500000)\n"
              << "  --width <W>             Window width (default: 1600)\n"
              << "  --height <H>            Window height (default: 900)\n"
              << "  --no-vsync              Disable vsync for uncapped framerate\n"
              << "  --scale <S>             Point size multiplier (default: 1.0)\n"
              << "  --scenario <S>          Sand scenario: pile, column, funnel, dune, cliff, block, clumps\n"
              << "  --cohesion <C>          Capillary cohesion strength (default: 60.0)\n"
              << "  --bond-break-force <F>  Glue bond tensile break force (N)\n"
              << "  --clump-radius <R>      Clump cluster radius (m)\n"
              << "  --grain-radius <R>      Base grain radius (m, default: 0.01)\n"
              << "  --frames <N>            Exit after N frames (benchmark)\n"
              << "  --help                  Display this help message\n\n"
              << "Studio Navigation Controls (Blender Convention):\n"
              << "  MMB Drag                Turntable Orbit camera around target\n"
              << "  Shift + MMB Drag        Pan camera across view plane\n"
              << "  Ctrl + MMB Drag / Wheel Zoom camera distance\n"
              << "  Hold RMB + WASD/QE      JangaFX Free-Fly Camera (Shift for turbo, Scroll for speed)\n"
              << "  Numpad 1 / 3 / 7        Front / Right / Top orthographic snap views (Ctrl for opposite)\n"
              << "  Numpad 5                Toggle Perspective / Orthographic projection\n"
              << "  Home                    Frame entire simulation scene bounds\n"
              << "  Numpad .                Frame selected object\n"
              << "  Space                   Play / Pause simulation\n"
              << "  Ctrl + Space / F11      Toggle Maximize Viewport mode\n"
              << "  Ctrl + R                Reset active simulation scenario\n"
              << "  W / E / R / Q           Gizmo Transform: Translate / Rotate / Scale / View\n"
              << "  G                       Toggle Infinite Ground Grid\n"
              << "  H                       Toggle Telemetry HUD overlay\n"
              << "  1 - 7                   Load Scenarios (Pile, Column, Funnel, Dune, Cliff, Block, Clumps)\n"
              << "  Escape                  Quit Studio\n";
}

} // anonymous namespace

int main(int argc, char** argv) {
    ViewerConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        } else if (arg == "--capacity" && i + 1 < argc) {
            config.capacity = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            config.windowWidth = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.windowHeight = std::stoi(argv[++i]);
        } else if (arg == "--no-vsync") {
            config.vsync = false;
        } else if (arg == "--scale" && i + 1 < argc) {
            config.pointScale = std::stof(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            config.maxFrames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if ((arg == "--scenario" || arg == "--sand-scenario") && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "pile") config.sandScenario = 0;
            else if (s == "column" || s == "column_collapse") config.sandScenario = 1;
            else if (s == "funnel") config.sandScenario = 2;
            else if (s == "dune") config.sandScenario = 3;
            else if (s == "cliff") config.sandScenario = 4;
            else if (s == "block" || s == "bonded_block") config.sandScenario = 5;
            else if (s == "clumps" || s == "clump") config.sandScenario = 6;
        } else if (arg == "--cohesion" && i + 1 < argc) {
            config.cohesion = std::stof(argv[++i]);
        } else if (arg == "--bond-break-force" && i + 1 < argc) {
            config.bondBreakForce = std::stof(argv[++i]);
        } else if (arg == "--clump-radius" && i + 1 < argc) {
            config.clumpRadius = std::stof(argv[++i]);
        } else if (arg == "--grain-radius" && i + 1 < argc) {
            config.grainRadius = std::stof(argv[++i]);
        }
    }

    g_app.config = config;
    g_app.state.currentScenarioIndex = config.sandScenario;
    g_app.state.displayScale = config.pointScale;
    g_app.state.capacity = config.capacity;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        config.windowWidth,
        config.windowHeight,
        "VParticles Simulation Studio - Industrial Granular DEM Engine",
        nullptr,
        nullptr);

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    g_app.window = window;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(config.vsync ? 1 : 0);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << "\n"
              << "OpenGL Version:  " << glGetString(GL_VERSION) << "\n"
              << "GLSL Version:    " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    // Register user callbacks BEFORE ImGui installs its backend, so ImGui chains them
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    // Initialize Studio UI (ImGui Dockspace + Desktop Fonts + Studio Theme + ImGuizmo)
    if (!g_app.ui.init(window)) {
        std::cerr << "Failed to initialize Studio UI" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize Infinite Procedural Ground Grid Renderer
    if (!g_app.grid.init()) {
        std::cerr << "Failed to initialize Infinite Grid Renderer" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize Offscreen Viewport Framebuffer
    if (!g_app.fbo.init(g_app.vpWidth, g_app.vpHeight)) {
        std::cerr << "Failed to initialize Viewport Framebuffer" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);

    ShaderProgram shader;
    if (!shader.compile(kParticleVertexShaderSrc, kParticleFragmentShaderSrc)) {
        std::cerr << "Failed to compile particle shaders" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Allocate OpenGL buffers for particle rendering
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const size_t vboBytes = static_cast<size_t>(config.capacity) * sizeof(RenderVertex);
    glBufferData(GL_ARRAY_BUFFER, vboBytes, nullptr, GL_DYNAMIC_DRAW);

    // Location 0: in_Position (vec4: xyz = pos, w = physical radius)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(0));

    // Location 1: in_Color (vec4: rgba)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(RenderVertex), reinterpret_cast<void*>(sizeof(float) * 4));

    GLuint indirectBuffer = 0;
    glGenBuffers(1, &indirectBuffer);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawArraysIndirectCommand), nullptr, GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

    // Initialize CUDA-OpenGL Bridge
    CudaGLBridge bridge;
    if (!bridge.init(vbo, indirectBuffer, config.capacity)) {
        std::cerr << "Failed to initialize CUDA-OpenGL bridge" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize ParticleSystem in Granular DEM mode
    ParticleSystem system(config.capacity, StorageMode::FP32, true);
    applySandScenario(system, g_app.state.currentScenarioIndex, config, config.kinematicCollider);
    g_app.state.granularSettings = system.granularSettings();

    std::cout << "VParticles Studio Initialized (" << config.capacity
              << " grain capacity, " << (vboBytes / (1024 * 1024)) << " MB VBO)\n"
              << "Scenario: [" << getSandScenarioName(g_app.state.currentScenarioIndex) << "]\n"
              << "Ready for interactive simulation." << std::endl;

    auto lastTime = std::chrono::high_resolution_clock::now();
    auto lastTitleUpdate = lastTime;
    auto appStartTime = lastTime;
    uint32_t totalRenderedFrames = 0;

    while (!glfwWindowShouldClose(window)) {
        auto frameStartTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = frameStartTime - lastTime;
        lastTime = frameStartTime;
        float dt = elapsed.count();
        if (dt > 0.1f) dt = 0.1f;

        // 1. JangaFX Free-Fly Camera Movement (WASD / QE / Shift Boost)
        if (g_app.state.isFlyModeActive) {
            float fwd = 0.0f;
            float rgt = 0.0f;
            float up  = 0.0f;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) fwd += 1.0f;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) fwd -= 1.0f;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) rgt += 1.0f;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) rgt -= 1.0f;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) up  += 1.0f;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) up  -= 1.0f;
            bool shiftFast = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS);
            g_app.camera.processFlyMovement(fwd, rgt, up, dt, shiftFast);
        }

        // 2. Scenario Switch or Reset Requests
        if (g_app.state.scenarioSwitchRequested || g_app.state.resetRequested) {
            applySandScenario(system, g_app.state.currentScenarioIndex, g_app.config,
                              g_app.state.granularSettings.kinematicColliderEnabled != 0);
            g_app.state.granularSettings = system.granularSettings();
            g_app.state.scenarioSwitchRequested = false;
            g_app.state.resetRequested = false;
        }

        // 3. Export Telemetry Request
        if (g_app.state.exportBenchmarkRequested) {
            exportTelemetryToCsv(g_app.state);
            g_app.state.exportBenchmarkRequested = false;
        }

        // 4. Sync live parameters from Inspector to CUDA ParticleSystem
        system.setGranularSettings(g_app.state.granularSettings);

        // 5. Physical simulation step
        float simMs = 0.0f;
        if (g_app.state.isPlaying || g_app.state.stepRequest != 0) {
            auto simStart = std::chrono::high_resolution_clock::now();
            float simDt = g_app.state.granularSettings.fixedDt > 0.0f ? g_app.state.granularSettings.fixedDt : (1.0f / 60.0f);
            simDt *= g_app.state.timeScale;
            system.update(simDt);
            system.synchronize();
            auto simEnd = std::chrono::high_resolution_clock::now();
            simMs = std::chrono::duration<float, std::milli>(simEnd - simStart).count();

            g_app.state.currentFrame++;
            if (g_app.state.currentFrame >= g_app.state.maxFrames) {
                if (g_app.state.isLooping) {
                    g_app.state.currentFrame = 0;
                    g_app.state.resetRequested = true;
                } else {
                    g_app.state.isPlaying = false;
                }
            }
            g_app.state.stepRequest = 0;
        }

        // 6. Query GPU Simulation Telemetry
        const auto& gStats = system.granularStats();
        g_app.state.liveAliveGrains = gStats.aliveCount;
        g_app.state.liveAsleepGrains = gStats.asleepCount;
        g_app.state.liveActiveBonds = gStats.activeBondCount;
        g_app.state.liveBrokenBonds = gStats.brokenBondCount;
        g_app.state.liveMeanHeight = gStats.meanHeight;
        g_app.state.liveMaxHeight = gStats.maxHeight;
        g_app.state.liveKineticEnergy = gStats.kineticEnergy;
        g_app.state.liveVboMemoryBytes = vboBytes;

        // 7. Gather into OpenGL VBO via CUDA-GL Bridge
        auto gatherStart = std::chrono::high_resolution_clock::now();
        bridge.gatherGranular(system.granularBuffers(), g_app.state.sandColorMode, g_app.state.velocityRangeMax, 1.0f);
        auto gatherEnd = std::chrono::high_resolution_clock::now();
        float gatherMs = std::chrono::duration<float, std::milli>(gatherEnd - gatherStart).count();

        // 8. Render 3D Scene into Viewport Offscreen FBO
        auto renderStart = std::chrono::high_resolution_clock::now();
        if (g_app.vpWidth > 0 && g_app.vpHeight > 0) {
            g_app.fbo.resize(g_app.vpWidth, g_app.vpHeight);
        }

        g_app.fbo.bind();
        glViewport(0, 0, g_app.fbo.width(), g_app.fbo.height());
        glClearColor(0.055f, 0.059f, 0.071f, 1.0f); // Dark Charcoal Slate
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        float aspect = g_app.fbo.aspect();
        Mat4 view = g_app.camera.viewMatrix();
        Mat4 proj = g_app.camera.projectionMatrix(aspect);
        Vec3 eye  = g_app.camera.eyePosition();

        // 8a. Infinite Ground Grid
        if (g_app.state.showInfiniteGrid) {
            g_app.grid.draw(view, proj, eye);
        }

        // 8b. Granular Particles with Physical Perspective & Decoupled Display Scale
        if (g_app.state.showParticles) {
            Mat4 vp = proj * view;
            shader.use();
            shader.setViewProj(vp.data());
            shader.setEyePos(eye.x, eye.y, eye.z);
            shader.setViewportHeight(static_cast<float>(g_app.fbo.height()));
            shader.setDisplayScale(g_app.state.displayScale);
            shader.setMinMaxPointSize(g_app.state.minPointSize, g_app.state.maxPointSize);
            float orthoHeight = g_app.camera.distance() * std::tan(g_app.camera.fovRadians() * 0.5f) * 2.0f;
            shader.setCameraProjection(g_app.camera.fovRadians(), g_app.camera.isOrthographic(), orthoHeight);
            shader.setRenderStyle(g_app.state.renderStyle);

            glBindVertexArray(vao);
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
            glDrawArraysIndirect(GL_POINTS, reinterpret_cast<const void*>(0));
            glBindVertexArray(0);
        }

        g_app.fbo.unbind();
        auto renderEnd = std::chrono::high_resolution_clock::now();
        float renderMs = std::chrono::duration<float, std::milli>(renderEnd - renderStart).count();

        // 9. Render Studio UI (Dockspace, Panels, Overlays) to Main Window
        int dispW = 0, dispH = 0;
        glfwGetFramebufferSize(window, &dispW, &dispH);
        glViewport(0, 0, dispW, dispH);
        glClearColor(0.04f, 0.04f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        g_app.ui.newFrame();
        g_app.ui.render(g_app.state, g_app.camera, g_app.fbo.texture(), g_app.vpWidth, g_app.vpHeight);
        g_app.ui.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();

        // 10. Frame Metrics & Diagnostics
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        float totalFrameMs = std::chrono::duration<float, std::milli>(frameEndTime - frameStartTime).count();
        float currentFps = (totalFrameMs > 0.0f) ? (1000.0f / totalFrameMs) : 0.0f;

        g_app.state.lastSimTimeMs = simMs;
        g_app.state.lastGatherTimeMs = gatherMs;
        g_app.state.lastRenderTimeMs = renderMs;
        g_app.state.lastFrameTimeMs = totalFrameMs;
        g_app.state.currentFps = currentFps;
        g_app.state.metrics.record(simMs, gatherMs, renderMs, totalFrameMs, currentFps);

        totalRenderedFrames++;
        if (config.maxFrames > 0 && totalRenderedFrames >= config.maxFrames) {
            auto totalElapsed = std::chrono::duration<double>(frameEndTime - appStartTime).count();
            std::cout << "\n--- Studio Benchmark Complete (" << totalRenderedFrames << " frames) ---\n"
                      << "Average FPS:        " << std::fixed << std::setprecision(1)
                      << (totalRenderedFrames / (totalElapsed > 0.0 ? totalElapsed : 1.0)) << "\n"
                      << "Live Grains:        " << g_app.state.liveAliveGrains << "\n"
                      << "Last Sim Time:      " << std::setprecision(3) << simMs << " ms\n"
                      << "Capacity:           " << config.capacity << "\n";
            break;
        }

        // 11. Window Title Telemetry (~4 Hz)
        std::chrono::duration<double> titleElapsed = frameEndTime - lastTitleUpdate;
        if (titleElapsed.count() >= 0.25) {
            lastTitleUpdate = frameEndTime;
            std::ostringstream ss;
            ss << "VParticles Studio | Granular DEM Core | "
               << g_app.state.liveAliveGrains << " Grains | "
               << std::fixed << std::setprecision(1) << currentFps << " FPS | "
               << "Sim: " << std::setprecision(2) << simMs << " ms | "
               << getSandScenarioName(g_app.state.currentScenarioIndex);
            glfwSetWindowTitle(window, ss.str().c_str());
        }
    }

    bridge.shutdown();
    g_app.ui.shutdown();
    g_app.grid.cleanup();
    g_app.fbo.destroy();

    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &indirectBuffer);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
