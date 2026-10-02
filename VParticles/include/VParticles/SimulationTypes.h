#pragma once

#include <cstdint>

namespace vparticles {

struct Float3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Color4 {
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    float a = 1.0f;
};

enum class StorageMode : uint8_t {
    FP32   = 0,   // float4 pos/vel/color + float age/lifetime + uint32 systemId
    Packed = 1,   // 26 bytes/particle SoA: int16 pos + fp16 vel + uint16 age + fp16 lifetime + uint32 colorPacked + fp16 size + uint16 systemId
};

struct EmitterDesc {
    Float3 position = {};
    Float3 velocity = {0.0f, 4.0f, 0.0f};
    Float3 velocityVariance = {1.0f, 1.0f, 1.0f};
    Color4 color = {1.0f, 0.55f, 0.18f, 1.0f};
    float spawnRate = 100000.0f;
    float lifetime = 4.0f;
    float lifetimeVariance = 0.25f;
    float size = 1.0f;
    float sizeVariance = 0.2f;
    uint16_t tileId = 0;
    uint16_t recipeId = 0;
};

struct CollisionPlane {
    Float3 point = {0.0f, 0.0f, 0.0f};
    Float3 normal = {0.0f, 1.0f, 0.0f};
    float bounce = 0.5f;     // Coefficient of restitution [0, 1]
    float friction = 0.1f;   // Tangential friction damping [0, 1]
    uint32_t enabled = 0;
};

struct CollisionSphere {
    Float3 center = {0.0f, 0.0f, 0.0f};
    float radius = 5.0f;
    float bounce = 0.5f;
    float friction = 0.1f;
    uint32_t enabled = 0;
    uint32_t invert = 0;     // 1 = contain particles inside, 0 = solid obstacle
};

struct CollisionBox {
    Float3 minBounds = {-10.0f, 0.0f, -10.0f};
    Float3 maxBounds = {10.0f, 20.0f, 10.0f};
    float bounce = 0.5f;
    float friction = 0.1f;
    uint32_t enabled = 0;
};

struct TurbulenceSettings {
    float strength = 0.0f;   // 0.0 = disabled
    float frequency = 0.2f;
    float speed = 0.5f;
};

struct CurveSettings {
    float startSize = 1.0f;
    float peakSize = 1.5f;
    float endSize = 0.0f;
    Color4 startColor = {1.0f, 0.6f, 0.2f, 1.0f};
    Color4 endColor = {0.2f, 0.05f, 0.02f, 0.0f};
    uint32_t enabled = 0;
};

constexpr uint32_t kMaxRecipes = 64;

struct EffectRecipe {
    Float3 gravity = {0.0f, -9.81f, 0.0f};
    Float3 wind = {0.0f, 0.0f, 0.0f};
    float drag = 0.05f;
    TurbulenceSettings turbulence = {};
    CollisionPlane plane = {};
    CollisionSphere sphere = {};
    CollisionBox box = {};
    CurveSettings curves = {};
    float lifetimeScale = 1.0f;
    float reserved = 0.0f;
};

constexpr uint32_t kMaxTiles = 256;

struct TileDesc {
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;
    float reserved = 0.0f;
};

struct TileStats {
    uint32_t tileId = 0;
    uint32_t aliveCount = 0;
    Float3 origin = {};
};

struct SimulationSettings {
    Float3 gravity = {0.0f, -9.81f, 0.0f};
    Float3 wind = {};
    float drag = 0.05f;
    uint32_t seed = 0xA341316Cu;
    TurbulenceSettings turbulence = {};
    CollisionPlane plane = {};
    CollisionSphere sphere = {};
    CollisionBox box = {};
    CurveSettings curves = {};
    Float3 tileOrigin = {};
    float posQuantizationScale = 100.0f;  // int16 units per world unit (default: 1cm)
    TileDesc tiles[kMaxTiles] = {};
    uint32_t tileCount = 1;
    EffectRecipe recipes[kMaxRecipes] = {};
    uint32_t recipeCount = 1;
};

struct SimulationStats {
    // Telemetry is delivered asynchronously. These identify the GPU frame that
    // produced this sample and how many later submissions were already queued
    // when the host observed it.
    uint64_t frameIndex = 0;
    uint32_t telemetryLatencyFrames = 0;
    bool valid = false;
    // True when this sample was submitted through a CUDA Graph. Graph frames
    // expose aggregate pipeline timing; per-stage fields are intentionally 0.
    bool graphActive = false;
    uint32_t graphRebuildCount = 0;
    uint32_t capacity = 0;
    uint32_t aliveCount = 0;
    uint32_t requestedSpawn = 0;
    uint32_t spawned = 0;
    uint32_t dropped = 0;
    uint32_t deadCount = 0;
    float spawnMs = 0.0f;
    float simulateMs = 0.0f;
    float compactMs = 0.0f;
    float totalMs = 0.0f;
};

} // namespace vparticles
