#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "VParticles/ParticleSystem.h"
#include "VParticles/SimulationTypes.h"
#include "Camera.h"
#include "CudaGLBridge.h"
#include "Math3D.h"
#include "Shaders.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace vparticles;
using namespace vparticles::viewer;

namespace {

struct ViewerConfig {
    uint32_t capacity = 1000000;
    uint32_t emitterCount = 8;
    uint32_t recipeCount = 8;
    float spawnRate = 250000.0f;
    int windowWidth = 1600;
    int windowHeight = 900;
    bool vsync = true;
    bool additiveBlend = true;
    float pointScale = 2.5f;
    uint32_t maxFrames = 0;
    int samples = 0;
};

struct AppState {
    Camera camera;
    ViewerConfig config;
    bool isPaused = false;
    bool leftMouseDown = false;
    bool rightMouseDown = false;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
    int currentPreset = 0;
    float pointScale = 2.5f;
    bool additiveBlend = true;
    bool resetRequested = false;
};

AppState g_app;

void framebufferSizeCallback(GLFWwindow* /*window*/, int width, int height) {
    if (width > 0 && height > 0) {
        g_app.config.windowWidth = width;
        g_app.config.windowHeight = height;
        glViewport(0, 0, width, height);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_app.leftMouseDown = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_app.rightMouseDown = (action == GLFW_PRESS);
    }
    glfwGetCursorPos(window, &g_app.lastMouseX, &g_app.lastMouseY);
}

void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    float dx = static_cast<float>(xpos - g_app.lastMouseX);
    float dy = static_cast<float>(ypos - g_app.lastMouseY);
    g_app.lastMouseX = xpos;
    g_app.lastMouseY = ypos;

    if (g_app.leftMouseDown) {
        g_app.camera.processOrbit(dx, dy);
    } else if (g_app.rightMouseDown) {
        g_app.camera.processPan(dx, dy);
    }
}

void scrollCallback(GLFWwindow* /*window*/, double /*xoffset*/, double yoffset) {
    g_app.camera.processZoom(static_cast<float>(yoffset));
}

void keyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_SPACE:
        g_app.isPaused = !g_app.isPaused;
        break;
    case GLFW_KEY_R:
        g_app.resetRequested = true;
        break;
    case GLFW_KEY_C:
        g_app.camera.reset({0.0f, 5.0f, 0.0f}, 40.0f, 45.0f, 25.0f);
        break;
    case GLFW_KEY_B:
        g_app.additiveBlend = !g_app.additiveBlend;
        break;
    case GLFW_KEY_EQUAL: // '+' key
    case GLFW_KEY_KP_ADD:
        g_app.pointScale = std::min(g_app.pointScale * 1.25f, 20.0f);
        break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
        g_app.pointScale = std::max(g_app.pointScale * 0.8f, 0.2f);
        break;
    case GLFW_KEY_1: g_app.currentPreset = 0; g_app.resetRequested = true; break;
    case GLFW_KEY_2: g_app.currentPreset = 1; g_app.resetRequested = true; break;
    case GLFW_KEY_3: g_app.currentPreset = 2; g_app.resetRequested = true; break;
    case GLFW_KEY_4: g_app.currentPreset = 3; g_app.resetRequested = true; break;
    case GLFW_KEY_5: g_app.currentPreset = 4; g_app.resetRequested = true; break;
    case GLFW_KEY_6: g_app.currentPreset = 5; g_app.resetRequested = true; break;
    case GLFW_KEY_7: g_app.currentPreset = 6; g_app.resetRequested = true; break;
    case GLFW_KEY_8: g_app.currentPreset = 7; g_app.resetRequested = true; break;
    case GLFW_KEY_TAB:
        g_app.currentPreset = (g_app.currentPreset + 1) % 8;
        g_app.resetRequested = true;
        break;
    default:
        break;
    }
}

void applyPreset(ParticleSystem& system, int preset, const ViewerConfig& config) {
    system.reset();
    system.clearEmitters();

    SimulationSettings settings = system.settings();
    settings.recipeCount = 8;

    // --- Base 8 Showcase Recipes (used in Showroom) ---
    // Recipe 0: Smoke Plume (buoyancy, curl turbulence, expanding ash)
    EffectRecipe r0 = {};
    r0.gravity = {0.0f, 1.8f, 0.0f};
    r0.drag = 0.22f;
    r0.turbulence = {3.5f, 0.2f, 0.5f};
    r0.curves = {1.0f, 4.5f, 7.0f, {0.9f, 0.9f, 0.95f, 0.85f}, {0.2f, 0.2f, 0.2f, 0.0f}, 1};
    r0.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.1f, 0.5f, 1};

    // Recipe 1: Golden Sparks (high gravity, bouncy floor)
    EffectRecipe r1 = {};
    r1.gravity = {0.0f, -22.0f, 0.0f};
    r1.drag = 0.015f;
    r1.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.85f, 0.08f, 1};
    r1.curves = {0.7f, 0.5f, 0.2f, {1.0f, 0.95f, 0.8f, 1.0f}, {0.6f, 0.1f, 0.01f, 0.0f}, 1};

    // Recipe 2: Bonfire Flames (rising heat, obstacle sphere)
    EffectRecipe r2 = {};
    r2.gravity = {0.0f, 7.0f, 0.0f};
    r2.drag = 0.08f;
    r2.turbulence = {3.0f, 0.35f, 1.2f};
    r2.curves = {1.0f, 3.2f, 0.5f, {1.0f, 0.95f, 0.35f, 1.0f}, {0.25f, 0.05f, 0.02f, 0.0f}, 1};

    // Recipe 3: Water Fountain (arching trajectory, wind, basin)
    EffectRecipe r3 = {};
    r3.gravity = {0.0f, -9.81f, 0.0f};
    r3.wind = {1.5f, 0.0f, 0.8f};
    r3.drag = 0.04f;
    r3.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.3f, 0.3f, 1};
    r3.curves = {0.8f, 1.4f, 0.5f, {0.2f, 0.7f, 1.0f, 0.9f}, {0.9f, 0.98f, 1.0f, 0.1f}, 1};

    // Recipe 4: Plasma Vortex (zero-G, containment sphere)
    EffectRecipe r4 = {};
    r4.gravity = {0.0f, 0.0f, 0.0f};
    r4.drag = 0.015f;
    r4.turbulence = {7.0f, 0.45f, 1.5f};
    r4.curves = {1.2f, 2.4f, 0.6f, {0.9f, 0.1f, 1.0f, 1.0f}, {0.05f, 0.85f, 1.0f, 0.0f}, 1};

    // Recipe 5: Shrapnel Blast (extreme gravity, floor ricochet)
    EffectRecipe r5 = {};
    r5.gravity = {0.0f, -32.0f, 0.0f};
    r5.drag = 0.02f;
    r5.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.72f, 0.22f, 1};
    r5.curves = {1.4f, 1.0f, 0.3f, {1.0f, 0.7f, 0.3f, 1.0f}, {0.3f, 0.1f, 0.05f, 0.0f}, 1};

    // Recipe 6: Magic Shimmer Galaxy (gentle float, high frequency swirl)
    EffectRecipe r6 = {};
    r6.gravity = {0.0f, -0.6f, 0.0f};
    r6.drag = 0.12f;
    r6.turbulence = {3.5f, 0.65f, 0.8f};
    r6.curves = {0.5f, 1.8f, 0.2f, {0.2f, 1.0f, 0.6f, 1.0f}, {0.8f, 0.3f, 1.0f, 0.0f}, 1};

    // Recipe 7: Firework Burst (rocket fountain)
    EffectRecipe r7 = {};
    r7.gravity = {0.0f, -8.0f, 0.0f};
    r7.wind = {-2.0f, 0.0f, 2.0f};
    r7.drag = 0.05f;
    r7.curves = {1.0f, 2.2f, 0.4f, {1.0f, 0.2f, 0.5f, 1.0f}, {1.0f, 0.9f, 0.2f, 0.0f}, 1};

    if (preset == 0) {
        // --- PRESET 0: SHOWROOM (Museum of 8 Effects) ---
        const float radius = 14.0f;
        // Position obstacle sphere specifically above pedestal 2
        float angle2 = (2.0f / 8.0f) * 2.0f * kPi;
        r2.sphere = {{std::cos(angle2) * radius, 6.0f, std::sin(angle2) * radius}, 2.5f, 0.4f, 0.2f, 1, 0};

        // Position containment sphere specifically around pedestal 4
        float angle4 = (4.0f / 8.0f) * 2.0f * kPi;
        r4.sphere = {{std::cos(angle4) * radius, 5.0f, std::sin(angle4) * radius}, 4.2f, 0.92f, 0.02f, 1, 1};

        settings.recipes[0] = r0;
        settings.recipes[1] = r1;
        settings.recipes[2] = r2;
        settings.recipes[3] = r3;
        settings.recipes[4] = r4;
        settings.recipes[5] = r5;
        settings.recipes[6] = r6;
        settings.recipes[7] = r7;
        system.setSettings(settings);

        const float perEmitterRate = config.spawnRate / 8.0f;
        for (uint32_t i = 0; i < 8; ++i) {
            EmitterDesc emitter = {};
            float angle = (static_cast<float>(i) / 8.0f) * 2.0f * kPi;
            float px = std::cos(angle) * radius;
            float pz = std::sin(angle) * radius;
            emitter.recipeId = static_cast<uint16_t>(i);
            emitter.spawnRate = perEmitterRate;

            switch (i) {
            case 0: // Smoke Plume
                emitter.position = {px, 0.5f, pz};
                emitter.velocity = {0.0f, 3.5f, 0.0f};
                emitter.velocityVariance = {1.2f, 1.0f, 1.2f};
                emitter.lifetime = 4.5f;
                emitter.lifetimeVariance = 0.2f;
                emitter.size = 1.2f;
                emitter.sizeVariance = 0.3f;
                emitter.color = {0.9f, 0.9f, 0.95f, 0.85f};
                break;
            case 1: // Golden Sparks (shower downward onto floor)
                emitter.position = {px, 6.0f, pz};
                emitter.velocity = {0.0f, -8.0f, 0.0f};
                emitter.velocityVariance = {6.0f, 3.0f, 6.0f};
                emitter.lifetime = 3.0f;
                emitter.lifetimeVariance = 0.25f;
                emitter.size = 0.7f;
                emitter.sizeVariance = 0.2f;
                emitter.color = {1.0f, 0.95f, 0.8f, 1.0f};
                break;
            case 2: // Bonfire & Obstacle Sphere
                emitter.position = {px, 0.5f, pz};
                emitter.velocity = {0.0f, 7.5f, 0.0f};
                emitter.velocityVariance = {1.5f, 2.0f, 1.5f};
                emitter.lifetime = 2.2f;
                emitter.lifetimeVariance = 0.2f;
                emitter.size = 1.0f;
                emitter.sizeVariance = 0.3f;
                emitter.color = {1.0f, 0.9f, 0.35f, 1.0f};
                break;
            case 3: // Water Fountain
                emitter.position = {px, 0.5f, pz};
                emitter.velocity = {0.0f, 16.0f, 0.0f};
                emitter.velocityVariance = {2.5f, 2.0f, 2.5f};
                emitter.lifetime = 3.5f;
                emitter.lifetimeVariance = 0.2f;
                emitter.size = 0.8f;
                emitter.sizeVariance = 0.2f;
                emitter.color = {0.2f, 0.7f, 1.0f, 0.9f};
                break;
            case 4: // Plasma Vortex (Containment)
                emitter.position = {px, 5.0f, pz};
                emitter.velocity = {2.0f, 2.0f, 2.0f};
                emitter.velocityVariance = {4.0f, 4.0f, 4.0f};
                emitter.lifetime = 4.0f;
                emitter.lifetimeVariance = 0.25f;
                emitter.size = 1.2f;
                emitter.sizeVariance = 0.3f;
                emitter.color = {0.9f, 0.1f, 1.0f, 1.0f};
                break;
            case 5: // Shrapnel Blast
                emitter.position = {px, 1.5f, pz};
                emitter.velocity = {0.0f, 10.0f, 0.0f};
                emitter.velocityVariance = {7.0f, 5.0f, 7.0f};
                emitter.lifetime = 2.0f;
                emitter.lifetimeVariance = 0.2f;
                emitter.size = 1.4f;
                emitter.sizeVariance = 0.3f;
                emitter.color = {1.0f, 0.6f, 0.2f, 1.0f};
                break;
            case 6: // Magic Shimmer
                emitter.position = {px, 1.0f, pz};
                emitter.velocity = {0.0f, 2.0f, 0.0f};
                emitter.velocityVariance = {2.0f, 1.5f, 2.0f};
                emitter.lifetime = 5.0f;
                emitter.lifetimeVariance = 0.25f;
                emitter.size = 0.6f;
                emitter.sizeVariance = 0.2f;
                emitter.color = {0.2f, 1.0f, 0.6f, 1.0f};
                break;
            case 7: // Firework Burst
                emitter.position = {px, 0.5f, pz};
                emitter.velocity = {0.0f, 18.0f, 0.0f};
                emitter.velocityVariance = {4.0f, 4.0f, 4.0f};
                emitter.lifetime = 3.2f;
                emitter.lifetimeVariance = 0.2f;
                emitter.size = 1.0f;
                emitter.sizeVariance = 0.3f;
                emitter.color = {1.0f, 0.2f, 0.5f, 1.0f};
                break;
            }
            system.addEmitter(emitter);
        }
    } else {
        // --- DEDICATED PRESETS (Full Scene Focus) ---
        EffectRecipe dedicatedRecipe = {};

        switch (preset) {
        case 1: { // SMOKE PLUME
            dedicatedRecipe.gravity = {0.0f, 1.8f, 0.0f};
            dedicatedRecipe.drag = 0.22f;
            dedicatedRecipe.turbulence = {4.5f, 0.16f, 0.45f};
            dedicatedRecipe.curves = {1.2f, 5.5f, 8.5f, {0.9f, 0.9f, 0.92f, 0.85f}, {0.18f, 0.18f, 0.18f, 0.0f}, 1};
            dedicatedRecipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.1f, 0.5f, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            const uint32_t numEmitters = 4;
            const float perRate = config.spawnRate / numEmitters;
            for (uint32_t i = 0; i < numEmitters; ++i) {
                float angle = (static_cast<float>(i) / numEmitters) * 2.0f * kPi;
                EmitterDesc e = {};
                e.position = {std::cos(angle) * 1.2f, 0.2f, std::sin(angle) * 1.2f};
                e.velocity = {0.0f, 4.0f, 0.0f};
                e.velocityVariance = {1.5f, 1.0f, 1.5f};
                e.spawnRate = perRate;
                e.lifetime = 4.5f;
                e.lifetimeVariance = 0.25f;
                e.size = 1.2f;
                e.sizeVariance = 0.3f;
                e.recipeId = static_cast<uint16_t>(i);
                e.color = {0.9f, 0.9f, 0.92f, 0.85f};
                system.addEmitter(e);
            }
            break;
        }
        case 2: { // SPARKS & WELDING RICOCHET
            dedicatedRecipe.gravity = {0.0f, -22.0f, 0.0f};
            dedicatedRecipe.drag = 0.015f;
            dedicatedRecipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.84f, 0.07f, 1}; // High bounce!
            dedicatedRecipe.curves = {0.8f, 0.6f, 0.2f, {1.0f, 1.0f, 0.9f, 1.0f}, {0.7f, 0.12f, 0.01f, 0.0f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            // Elevated nozzle blasting sparks downward
            EmitterDesc mainNozzle = {};
            mainNozzle.position = {0.0f, 16.0f, 0.0f};
            mainNozzle.velocity = {0.0f, -14.0f, 0.0f};
            mainNozzle.velocityVariance = {12.0f, 4.0f, 12.0f};
            mainNozzle.spawnRate = config.spawnRate * 0.7f;
            mainNozzle.lifetime = 3.0f;
            mainNozzle.lifetimeVariance = 0.3f;
            mainNozzle.size = 0.7f;
            mainNozzle.sizeVariance = 0.2f;
            mainNozzle.recipeId = 0;
            mainNozzle.color = {1.0f, 1.0f, 0.9f, 1.0f};
            system.addEmitter(mainNozzle);

            // 2 angled ricochet nozzles
            for (int k = -1; k <= 1; k += 2) {
                EmitterDesc side = {};
                side.position = {k * 2.0f, 14.0f, 0.0f};
                side.velocity = {k * 6.0f, -10.0f, 0.0f};
                side.velocityVariance = {8.0f, 3.0f, 8.0f};
                side.spawnRate = config.spawnRate * 0.15f;
                side.lifetime = 2.8f;
                side.lifetimeVariance = 0.25f;
                side.size = 0.6f;
                side.sizeVariance = 0.2f;
                side.recipeId = 1;
                side.color = {1.0f, 0.85f, 0.3f, 1.0f};
                system.addEmitter(side);
            }
            break;
        }
        case 3: { // BONFIRE & OBSTACLE DEFLECTOR SPHERE
            dedicatedRecipe.gravity = {0.0f, 8.0f, 0.0f};
            dedicatedRecipe.drag = 0.08f;
            dedicatedRecipe.turbulence = {3.8f, 0.32f, 1.4f};
            // Obstacle sphere suspended directly in the fire path!
            dedicatedRecipe.sphere = {{0.0f, 7.0f, 0.0f}, 3.5f, 0.4f, 0.2f, 1, 0};
            dedicatedRecipe.curves = {1.2f, 3.6f, 0.5f, {1.0f, 0.95f, 0.4f, 1.0f}, {0.25f, 0.04f, 0.01f, 0.0f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            const uint32_t numFlames = 6;
            const float perRate = config.spawnRate / numFlames;
            for (uint32_t i = 0; i < numFlames; ++i) {
                float angle = (static_cast<float>(i) / numFlames) * 2.0f * kPi;
                EmitterDesc flame = {};
                flame.position = {std::cos(angle) * 2.0f, 0.2f, std::sin(angle) * 2.0f};
                flame.velocity = {0.0f, 8.5f, 0.0f};
                flame.velocityVariance = {1.5f, 2.0f, 1.5f};
                flame.spawnRate = perRate;
                flame.lifetime = 2.2f;
                flame.lifetimeVariance = 0.2f;
                flame.size = 1.1f;
                flame.sizeVariance = 0.3f;
                flame.recipeId = static_cast<uint16_t>(i);
                flame.color = {1.0f, 0.9f, 0.35f, 1.0f};
                system.addEmitter(flame);
            }
            break;
        }
        case 4: { // GRAND WATER FOUNTAIN
            dedicatedRecipe.gravity = {0.0f, -9.81f, 0.0f};
            dedicatedRecipe.wind = {2.5f, 0.0f, 1.2f};
            dedicatedRecipe.drag = 0.04f;
            dedicatedRecipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.28f, 0.25f, 1};
            dedicatedRecipe.box = {{-20.0f, 0.0f, -20.0f}, {20.0f, 32.0f, 20.0f}, 0.4f, 0.2f, 1};
            dedicatedRecipe.curves = {0.8f, 1.5f, 0.4f, {0.2f, 0.75f, 1.0f, 0.9f}, {0.9f, 0.98f, 1.0f, 0.15f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            // Central geyser
            EmitterDesc center = {};
            center.position = {0.0f, 0.5f, 0.0f};
            center.velocity = {0.0f, 24.0f, 0.0f};
            center.velocityVariance = {1.5f, 2.0f, 1.5f};
            center.spawnRate = config.spawnRate * 0.45f;
            center.lifetime = 3.8f;
            center.lifetimeVariance = 0.15f;
            center.size = 0.9f;
            center.sizeVariance = 0.2f;
            center.recipeId = 0;
            center.color = {0.25f, 0.8f, 1.0f, 0.95f};
            system.addEmitter(center);

            // 6 outward arching jets
            const uint32_t ringJets = 6;
            const float ringRate = (config.spawnRate * 0.55f) / ringJets;
            for (uint32_t i = 0; i < ringJets; ++i) {
                float angle = (static_cast<float>(i) / ringJets) * 2.0f * kPi;
                EmitterDesc jet = {};
                jet.position = {std::cos(angle) * 4.0f, 0.5f, std::sin(angle) * 4.0f};
                jet.velocity = {std::cos(angle) * 6.0f, 17.0f, std::sin(angle) * 6.0f};
                jet.velocityVariance = {2.0f, 1.5f, 2.0f};
                jet.spawnRate = ringRate;
                jet.lifetime = 3.5f;
                jet.lifetimeVariance = 0.2f;
                jet.size = 0.8f;
                jet.sizeVariance = 0.2f;
                jet.recipeId = static_cast<uint16_t>(i + 1);
                jet.color = {0.2f, 0.7f, 1.0f, 0.9f};
                system.addEmitter(jet);
            }
            break;
        }
        case 5: { // PLASMA CONTAINMENT VORTEX
            dedicatedRecipe.gravity = {0.0f, 0.0f, 0.0f}; // Zero-G
            dedicatedRecipe.drag = 0.012f;
            dedicatedRecipe.turbulence = {8.5f, 0.45f, 1.8f};
            // Inverted sphere: keeps particles trapped inside the 9m sphere!
            dedicatedRecipe.sphere = {{0.0f, 6.0f, 0.0f}, 9.0f, 0.94f, 0.015f, 1, 1};
            dedicatedRecipe.curves = {1.3f, 2.8f, 0.7f, {0.92f, 0.08f, 1.0f, 1.0f}, {0.05f, 0.88f, 1.0f, 0.0f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            const uint32_t numJets = 6;
            const float perRate = config.spawnRate / numJets;
            const Float3 directions[6] = {
                {5.0f, 0.0f, 0.0f}, {-5.0f, 0.0f, 0.0f},
                {0.0f, 5.0f, 0.0f}, {0.0f, -5.0f, 0.0f},
                {0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, -5.0f}
            };
            for (uint32_t i = 0; i < numJets; ++i) {
                EmitterDesc e = {};
                e.position = {0.0f, 6.0f, 0.0f};
                e.velocity = directions[i];
                e.velocityVariance = {4.0f, 4.0f, 4.0f};
                e.spawnRate = perRate;
                e.lifetime = 4.5f;
                e.lifetimeVariance = 0.25f;
                e.size = 1.2f;
                e.sizeVariance = 0.3f;
                e.recipeId = static_cast<uint16_t>(i);
                e.color = {0.9f, 0.1f, 1.0f, 1.0f};
                system.addEmitter(e);
            }
            break;
        }
        case 6: { // SHRAPNEL & SHOCKWAVE
            dedicatedRecipe.gravity = {0.0f, -35.0f, 0.0f}; // Crushing gravity!
            dedicatedRecipe.drag = 0.02f;
            dedicatedRecipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.74f, 0.22f, 1};
            dedicatedRecipe.curves = {1.6f, 1.2f, 0.3f, {1.0f, 0.85f, 0.4f, 1.0f}, {0.25f, 0.1f, 0.05f, 0.0f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            const uint32_t numBursts = 8;
            const float perRate = config.spawnRate / numBursts;
            for (uint32_t i = 0; i < numBursts; ++i) {
                float angle = (static_cast<float>(i) / numBursts) * 2.0f * kPi;
                EmitterDesc e = {};
                e.position = {0.0f, 0.8f, 0.0f};
                e.velocity = {std::cos(angle) * 24.0f, 11.0f, std::sin(angle) * 24.0f};
                e.velocityVariance = {8.0f, 6.0f, 8.0f};
                e.spawnRate = perRate;
                e.lifetime = 2.0f;
                e.lifetimeVariance = 0.25f;
                e.size = 1.5f;
                e.sizeVariance = 0.4f;
                e.recipeId = static_cast<uint16_t>(i);
                e.color = {1.0f, 0.7f, 0.3f, 1.0f};
                system.addEmitter(e);
            }
            break;
        }
        case 7: { // MAGIC SHIMMER GALAXY
            dedicatedRecipe.gravity = {0.0f, -0.5f, 0.0f};
            dedicatedRecipe.drag = 0.10f;
            dedicatedRecipe.turbulence = {4.0f, 0.75f, 0.9f};
            dedicatedRecipe.curves = {0.5f, 1.8f, 0.2f, {0.15f, 1.0f, 0.65f, 1.0f}, {0.85f, 0.35f, 1.0f, 0.0f}, 1};

            for (uint32_t r = 0; r < 8; ++r) settings.recipes[r] = dedicatedRecipe;
            system.setSettings(settings);

            const uint32_t numSpirals = 4;
            const float perRate = config.spawnRate / numSpirals;
            for (uint32_t i = 0; i < numSpirals; ++i) {
                float angle = (static_cast<float>(i) / numSpirals) * 2.0f * kPi;
                EmitterDesc e = {};
                e.position = {std::cos(angle) * 3.0f, 1.5f + i * 0.8f, std::sin(angle) * 3.0f};
                e.velocity = {-std::sin(angle) * 4.0f, 1.5f, std::cos(angle) * 4.0f};
                e.velocityVariance = {2.0f, 1.5f, 2.0f};
                e.spawnRate = perRate;
                e.lifetime = 5.5f;
                e.lifetimeVariance = 0.3f;
                e.size = 0.6f;
                e.sizeVariance = 0.2f;
                e.recipeId = static_cast<uint16_t>(i);
                e.color = {0.2f, 1.0f, 0.65f, 1.0f};
                system.addEmitter(e);
            }
            break;
        }
        default:
            break;
        }
    }
}

const char* getPresetName(int preset) {
    switch (preset) {
    case 0: return "Showroom (8 Effects)";
    case 1: return "Smoke Plume";
    case 2: return "Sparks & Ricochet";
    case 3: return "Bonfire & Deflector Sphere";
    case 4: return "Grand Water Fountain";
    case 5: return "Plasma Vortex (Containment)";
    case 6: return "Shrapnel & Shockwave";
    case 7: return "Magic Shimmer Galaxy";
    default: return "Custom";
    }
}

void printHelp() {
    std::cout << "VParticles Interactive OpenGL Viewer\n"
              << "Usage: VParticlesViewer [options]\n\n"
              << "Options:\n"
              << "  --capacity <N>      Pool particle capacity (default: 1000000)\n"
              << "  --rate <N>          Total spawn rate / sec (default: 250000)\n"
              << "  --width <W>         Window width (default: 1600)\n"
              << "  --height <H>        Window height (default: 900)\n"
              << "  --no-vsync          Disable vsync for uncapped framerate\n"
              << "  --blend <add|alpha> Initial blend mode (default: add)\n"
              << "  --scale <S>         Point size multiplier (default: 2.5)\n"
              << "  --samples <N>       MSAA samples (default: 0, procedural AA in shader)\n"
              << "  --frames <N>        Exit automatically after N frames and report stats\n"
              << "  --help              Display this help message\n\n"
              << "Controls:\n"
              << "  Left Drag           Orbit camera\n"
              << "  Right Drag          Pan camera\n"
              << "  Scroll Wheel        Zoom\n"
              << "  C                   Reset camera view\n"
              << "  Space               Pause / resume simulation\n"
              << "  R                   Reset simulation\n"
              << "  B                   Toggle blend mode (Additive / Alpha)\n"
              << "  + / -               Increase / decrease particle point size\n"
              << "  1                   Preset: Showroom (all 8 effects around museum ring)\n"
              << "  2                   Preset: Smoke Plume (buoyancy, curl turbulence, expanding ash)\n"
              << "  3                   Preset: Sparks & Ricochet (high-speed downward shower, floor bounce)\n"
              << "  4                   Preset: Bonfire & Deflector Sphere (obstacle collision, flame wrap)\n"
              << "  5                   Preset: Grand Water Fountain (geyser, arching jets, basin)\n"
              << "  6                   Preset: Plasma Vortex (zero-G magnetic containment sphere)\n"
              << "  7                   Preset: Shrapnel & Shockwave (crushing gravity, floor ricochet)\n"
              << "  8                   Preset: Magic Shimmer Galaxy (stardust spiral, ethereal float)\n"
              << "  Tab                 Cycle through presets\n"
              << "  Escape              Quit\n";
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
        } else if (arg == "--emitters" && i + 1 < argc) {
            config.emitterCount = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--recipes" && i + 1 < argc) {
            config.recipeCount = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--rate" && i + 1 < argc) {
            config.spawnRate = std::stof(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            config.windowWidth = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.windowHeight = std::stoi(argv[++i]);
        } else if (arg == "--no-vsync") {
            config.vsync = false;
        } else if (arg == "--blend" && i + 1 < argc) {
            std::string b = argv[++i];
            config.additiveBlend = (b != "alpha");
        } else if (arg == "--scale" && i + 1 < argc) {
            config.pointScale = std::stof(argv[++i]);
        } else if (arg == "--samples" && i + 1 < argc) {
            config.samples = std::stoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            config.maxFrames = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }

    g_app.config = config;
    g_app.pointScale = config.pointScale;
    g_app.additiveBlend = config.additiveBlend;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (config.samples > 0) {
        glfwWindowHint(GLFW_SAMPLES, config.samples);
    }

    GLFWwindow* window = glfwCreateWindow(
        config.windowWidth,
        config.windowHeight,
        "VParticles Viewer",
        nullptr,
        nullptr);

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

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

    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);

    ShaderProgram shader;
    if (!shader.compile(kParticleVertexShaderSrc, kParticleFragmentShaderSrc)) {
        std::cerr << "Failed to compile particle shaders" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Allocate OpenGL buffers
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    const size_t vboBytes = static_cast<size_t>(config.capacity) * sizeof(RenderVertex);
    glBufferData(GL_ARRAY_BUFFER, vboBytes, nullptr, GL_DYNAMIC_DRAW);

    // Location 0: in_Position (vec4: xyz = pos, w = size)
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

    // Initialize ParticleSystem (FP32 mode, CUDA Graphs enabled)
    ParticleSystem system(config.capacity, StorageMode::FP32, true);
    applyPreset(system, g_app.currentPreset, config);

    std::cout << "Particle system initialized with capacity " << config.capacity
              << " (" << (vboBytes / (1024 * 1024)) << " MB VBO)" << std::endl;

    auto lastTime = std::chrono::high_resolution_clock::now();
    auto lastTitleUpdate = lastTime;
    auto appStartTime = lastTime;
    uint32_t frameCount = 0;
    uint32_t totalRenderedFrames = 0;
    double accumulatedFps = 0.0;
    double lastSimTimeMs = 0.0;

    while (!glfwWindowShouldClose(window)) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = currentTime - lastTime;
        lastTime = currentTime;
        float dt = elapsed.count();
        if (dt > 0.05f) dt = 0.05f; // Clamp delta time to avoid large jumps

        if (g_app.resetRequested) {
            applyPreset(system, g_app.currentPreset, g_app.config);
            g_app.resetRequested = false;
        }

        // Simulation update step
        if (!g_app.isPaused) {
            auto simStart = std::chrono::high_resolution_clock::now();
            system.update(dt);
            system.synchronize(); // Ensure simulation completes before gather
            auto simEnd = std::chrono::high_resolution_clock::now();
            lastSimTimeMs = std::chrono::duration<double, std::milli>(simEnd - simStart).count();
        }

        // Gather live particles from CUDA pool into registered OpenGL VBO
        bridge.gather(system.gpuBuffers());

        // Render pass
        int displayW = 0, displayH = 0;
        glfwGetFramebufferSize(window, &displayW, &displayH);
        glViewport(0, 0, displayW, displayH);

        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_BLEND);
        if (g_app.additiveBlend) {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        } else {
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

        float aspect = (displayH > 0) ? (static_cast<float>(displayW) / static_cast<float>(displayH)) : 1.0f;
        Mat4 vp = g_app.camera.viewProjectionMatrix(aspect);
        Vec3 eye = g_app.camera.eyePosition();

        shader.use();
        shader.setViewProj(vp.data());
        shader.setEyePos(eye.x, eye.y, eye.z);
        shader.setViewportHeight(static_cast<float>(displayH));
        shader.setPointScale(g_app.pointScale);

        glBindVertexArray(vao);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
        glDrawArraysIndirect(GL_POINTS, reinterpret_cast<const void*>(0));
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();

        totalRenderedFrames++;
        if (config.maxFrames > 0 && totalRenderedFrames >= config.maxFrames) {
            auto totalElapsed = std::chrono::duration<double>(currentTime - appStartTime).count();
            std::cout << "\n--- Viewer Benchmark Complete (" << totalRenderedFrames << " frames) ---\n"
                      << "Average FPS:        " << std::fixed << std::setprecision(1)
                      << (totalRenderedFrames / (totalElapsed > 0.0 ? totalElapsed : 1.0)) << "\n"
                      << "Live Particles:     " << system.stats().aliveCount << "\n"
                      << "Last Sim Time:      " << std::setprecision(3) << lastSimTimeMs << " ms\n"
                      << "Capacity:           " << config.capacity << "\n";
            break;
        }

        // Update window title telemetry at ~10 Hz
        frameCount++;
        accumulatedFps += (dt > 0.0f) ? (1.0 / dt) : 0.0;

        std::chrono::duration<double> titleElapsed = currentTime - lastTitleUpdate;
        if (titleElapsed.count() >= 0.1) {
            double avgFps = accumulatedFps / frameCount;
            frameCount = 0;
            accumulatedFps = 0.0;
            lastTitleUpdate = currentTime;

            uint32_t aliveCount = system.stats().aliveCount;

            std::ostringstream ss;
            ss << "VParticles Viewer | "
               << aliveCount << " Particles | "
               << std::fixed << std::setprecision(1) << avgFps << " FPS | "
               << "Sim: " << std::setprecision(2) << lastSimTimeMs << " ms | "
               << getPresetName(g_app.currentPreset)
               << (g_app.isPaused ? " [PAUSED]" : "")
               << (g_app.additiveBlend ? " [Add]" : " [Alpha]")
               << " | Scale: " << std::setprecision(1) << g_app.pointScale << "x";

            glfwSetWindowTitle(window, ss.str().c_str());
        }
    }

    bridge.shutdown();
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &indirectBuffer);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
