#pragma once

#include <cstdint>

struct Particle
{
    uint32_t id;

    float x;
    float y;

    float vx;
    float vy;

    float lifetime;
    float maxLifetime;

    float size;
    float baseSize;

    float rotation;
    float angularVelocity;

    float mass;

    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;

    std::uint8_t baseR;
    std::uint8_t baseG;
    std::uint8_t baseB;
    std::uint8_t baseA;
};