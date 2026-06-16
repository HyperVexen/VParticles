#pragma once

#include "CudaTypes.h"
#include <array>
#include <cstdint>

enum class EmitterShape
{
    Point,
    Circle,
    Box,
    Sphere,
    Cube,
    Grid
};

enum class EmissionMode
{
    Volume, // Emit uniformly inside boundary
    Surface  // Emit uniformly on boundary/surface
};


enum class ArtistPreset
{
    Custom,
    Rain,
    Snow,
    Smoke,
    Dust,
    Sparks
};

enum class ComputeMode
{
    GPU     // GPU Native
};

struct SimulationSettings
{
    // Timeline & Determinism
    float startTime = 0.0f;
    float endTime = 100000.0f; // Basically infinite continuous by default
    uint32_t randomSeed = 12345;

    // Emission & Transform
    float emitterX = 640.0f;
    float emitterY = 360.0f;
    float emitterZ = 0.0f;
    EmitterShape shape = EmitterShape::Point;
    EmissionMode emissionMode = EmissionMode::Volume;
    float emitRadius = 50.0f;
    float emitWidth = 100.0f;
    float emitHeight = 100.0f;
    float emitDepth = 100.0f;
    float emitCubeSize = 100.0f;
    
    int gridColumns = 10;
    int gridRows = 10;
    int gridSlices = 1;
    float gridSpacingX = 20.0f;
    float gridSpacingY = 20.0f;
    float gridSpacingZ = 20.0f;

    bool showVisualGrid = true;

    float spawnRate = 100.0f; // particles/sec
    int burstCount = 0;
    int particleCount = 1000000; // max active cap
    bool paused = false;
    float timeScale = 1.0f;

    // Physics
    float gravity = 0.0f;
    float windX = 0.0f;
    float windY = 0.0f;
    float windZ = 0.0f;
    float drag = 0.0f;
    float baseMass = 1.0f;
    float massRandomness = 0.0f;

    // Velocity
    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float velocityZ = 0.0f;
    float velocityVarianceX = 100.0f;
    float velocityVarianceY = 100.0f;
    float velocityVarianceZ = 0.0f;

    // Lifetime
    float particleLifetime = 2.0f;
    float lifetimeRandomness = 0.0f; // e.g. 0.5 for 50% variance

    // Size
    float baseSize = 4.0f;
    float sizeRandomness = 0.0f;
    float sizeStart = 1.0f;
    float sizeEnd = 1.0f;

    // Rotation
    float initialRotationMin = 0.0f;
    float initialRotationMax = 0.0f;
    float angularVelocityMin = 0.0f;
    float angularVelocityMax = 0.0f;

    // Color
    float colorRandomness = 0.0f;
    CudaColor colorGradient[4] = {
        CudaColor(255, 255, 255, 255),
        CudaColor(255, 255, 255, 255),
        CudaColor(255, 255, 255, 255),
        CudaColor(255, 255, 255, 255)
    };

    ArtistPreset activePreset = ArtistPreset::Custom;
    ComputeMode  computeMode  = ComputeMode::GPU;
};

inline bool operator==(const SimulationSettings& a, const SimulationSettings& b)
{
    if (a.startTime != b.startTime) return false;
    if (a.endTime != b.endTime) return false;
    if (a.randomSeed != b.randomSeed) return false;
    if (a.emitterX != b.emitterX) return false;
    if (a.emitterY != b.emitterY) return false;
    if (a.emitterZ != b.emitterZ) return false;
    if (a.shape != b.shape) return false;
    if (a.emissionMode != b.emissionMode) return false;
    if (a.emitRadius != b.emitRadius) return false;
    if (a.emitWidth != b.emitWidth) return false;
    if (a.emitHeight != b.emitHeight) return false;
    if (a.emitDepth != b.emitDepth) return false;
    if (a.emitCubeSize != b.emitCubeSize) return false;
    if (a.gridColumns != b.gridColumns) return false;
    if (a.gridRows != b.gridRows) return false;
    if (a.gridSlices != b.gridSlices) return false;
    if (a.gridSpacingX != b.gridSpacingX) return false;
    if (a.gridSpacingY != b.gridSpacingY) return false;
    if (a.gridSpacingZ != b.gridSpacingZ) return false;
    if (a.showVisualGrid != b.showVisualGrid) return false;
    if (a.spawnRate != b.spawnRate) return false;
    if (a.burstCount != b.burstCount) return false;
    if (a.particleCount != b.particleCount) return false;
    if (a.paused != b.paused) return false;
    if (a.timeScale != b.timeScale) return false;
    if (a.gravity != b.gravity) return false;
    if (a.windX != b.windX) return false;
    if (a.windY != b.windY) return false;
    if (a.windZ != b.windZ) return false;
    if (a.drag != b.drag) return false;
    if (a.baseMass != b.baseMass) return false;
    if (a.massRandomness != b.massRandomness) return false;
    if (a.velocityX != b.velocityX) return false;
    if (a.velocityY != b.velocityY) return false;
    if (a.velocityZ != b.velocityZ) return false;
    if (a.velocityVarianceX != b.velocityVarianceX) return false;
    if (a.velocityVarianceY != b.velocityVarianceY) return false;
    if (a.velocityVarianceZ != b.velocityVarianceZ) return false;
    if (a.particleLifetime != b.particleLifetime) return false;
    if (a.lifetimeRandomness != b.lifetimeRandomness) return false;
    if (a.baseSize != b.baseSize) return false;
    if (a.sizeRandomness != b.sizeRandomness) return false;
    if (a.sizeStart != b.sizeStart) return false;
    if (a.sizeEnd != b.sizeEnd) return false;
    if (a.initialRotationMin != b.initialRotationMin) return false;
    if (a.initialRotationMax != b.initialRotationMax) return false;
    if (a.angularVelocityMin != b.angularVelocityMin) return false;
    if (a.angularVelocityMax != b.angularVelocityMax) return false;
    if (a.colorRandomness != b.colorRandomness) return false;
    for (int i = 0; i < 4; ++i)
    {
        if (a.colorGradient[i].r != b.colorGradient[i].r) return false;
        if (a.colorGradient[i].g != b.colorGradient[i].g) return false;
        if (a.colorGradient[i].b != b.colorGradient[i].b) return false;
        if (a.colorGradient[i].a != b.colorGradient[i].a) return false;
    }
    if (a.activePreset != b.activePreset) return false;
    if (a.computeMode != b.computeMode) return false;
    return true;
}

inline bool operator!=(const SimulationSettings& a, const SimulationSettings& b)
{
    return !(a == b);
}
