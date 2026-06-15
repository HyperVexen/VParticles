#pragma once

#include <SFML/Graphics/Color.hpp>
#include <array>
#include <cstdint>

enum class EmitterShape
{
    Point,
    Circle,
    Box
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

struct SimulationSettings
{
    // Timeline & Determinism
    float startTime = 0.0f;
    float endTime = 100000.0f; // Basically infinite continuous by default
    uint32_t randomSeed = 12345;

    // Emission & Transform
    float emitterX = 640.0f;
    float emitterY = 360.0f;
    EmitterShape shape = EmitterShape::Point;
    float emitRadius = 50.0f;
    float emitWidth = 100.0f;
    float emitHeight = 100.0f;

    float spawnRate = 100.0f; // particles/sec
    int burstCount = 0;
    int particleCount = 1000000; // max active cap
    bool paused = false;

    // Physics
    float gravity = 0.0f;
    float windX = 0.0f;
    float windY = 0.0f;
    float drag = 0.0f;
    float baseMass = 1.0f;
    float massRandomness = 0.0f;

    // Velocity
    float velocityX = 0.0f;
    float velocityY = 0.0f;
    float velocityVarianceX = 100.0f;
    float velocityVarianceY = 100.0f;

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
    std::array<sf::Color, 4> colorGradient = {
        sf::Color::White,
        sf::Color::White,
        sf::Color::White,
        sf::Color::White
    };

    ArtistPreset activePreset = ArtistPreset::Custom;
};
