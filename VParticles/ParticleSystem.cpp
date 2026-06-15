#include "ParticleSystem.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to linearly interpolate between two values
static float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

// Helper to lerp colors
static sf::Color LerpColor(const sf::Color& a, const sf::Color& b, float t)
{
    return sf::Color(
        static_cast<std::uint8_t>(a.r + (b.r - a.r) * t),
        static_cast<std::uint8_t>(a.g + (b.g - a.g) * t),
        static_cast<std::uint8_t>(a.b + (b.b - a.b) * t),
        static_cast<std::uint8_t>(a.a + (b.a - a.a) * t)
    );
}

void ParticleSystem::InitializePool(std::size_t maxParticles)
{
    particles.resize(maxParticles);
    activeCount = 0;
}

void ParticleSystem::Reset(const SimulationSettings& settings)
{
    activeCount = 0;
    simulationTime = 0.0f;
    spawnAccumulator = 0.0f;
    burstFired = false;
    nextParticleId = 0;
    rng.seed(settings.randomSeed);
}

bool ParticleSystem::SpawnParticle(const SimulationSettings& settings)
{
    if (activeCount >= particles.size()) return false;
    if (activeCount >= static_cast<std::size_t>(settings.particleCount)) return false;

    Particle& p = particles[activeCount];
    p.id = nextParticleId++;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> distM1to1(-1.0f, 1.0f);

    // Emitter Shape
    p.x = settings.emitterX;
    p.y = settings.emitterY;

    if (settings.shape == EmitterShape::Circle)
    {
        float angle = dist01(rng) * 2.0f * static_cast<float>(M_PI);
        float radius = std::sqrt(dist01(rng)) * settings.emitRadius;
        p.x += std::cos(angle) * radius;
        p.y += std::sin(angle) * radius;
    }
    else if (settings.shape == EmitterShape::Box)
    {
        p.x += distM1to1(rng) * (settings.emitWidth * 0.5f);
        p.y += distM1to1(rng) * (settings.emitHeight * 0.5f);
    }

    // Velocity
    p.vx = settings.velocityX + distM1to1(rng) * settings.velocityVarianceX;
    p.vy = settings.velocityY + distM1to1(rng) * settings.velocityVarianceY;

    // Lifetime
    float lifeVar = distM1to1(rng) * settings.lifetimeRandomness * settings.particleLifetime;
    p.maxLifetime = settings.particleLifetime + lifeVar;
    if (p.maxLifetime < 0.05f) p.maxLifetime = 0.05f;
    p.lifetime = p.maxLifetime;

    // Mass
    float massVar = distM1to1(rng) * settings.massRandomness * settings.baseMass;
    p.mass = settings.baseMass + massVar;
    if (p.mass < 0.01f) p.mass = 0.01f;

    // Size
    float sizeVar = distM1to1(rng) * settings.sizeRandomness * settings.baseSize;
    p.baseSize = settings.baseSize + sizeVar;
    if (p.baseSize < 0.1f) p.baseSize = 0.1f;
    p.size = p.baseSize;

    // Rotation
    p.rotation = settings.initialRotationMin + dist01(rng) * (settings.initialRotationMax - settings.initialRotationMin);
    p.angularVelocity = settings.angularVelocityMin + dist01(rng) * (settings.angularVelocityMax - settings.angularVelocityMin);

    // Color Randomness (tint the base color slightly)
    sf::Color baseC = settings.colorGradient[0];
    float tintVarR = distM1to1(rng) * settings.colorRandomness * 255.0f;
    float tintVarG = distM1to1(rng) * settings.colorRandomness * 255.0f;
    float tintVarB = distM1to1(rng) * settings.colorRandomness * 255.0f;

    p.baseR = static_cast<std::uint8_t>(std::clamp(baseC.r + tintVarR, 0.0f, 255.0f));
    p.baseG = static_cast<std::uint8_t>(std::clamp(baseC.g + tintVarG, 0.0f, 255.0f));
    p.baseB = static_cast<std::uint8_t>(std::clamp(baseC.b + tintVarB, 0.0f, 255.0f));
    p.baseA = baseC.a;

    p.r = p.baseR;
    p.g = p.baseG;
    p.b = p.baseB;
    p.a = p.baseA;

    activeCount++;
    return true;
}

void ParticleSystem::Update(float dt, const SimulationSettings& settings, SimulationStats& stats)
{
    if (settings.paused) return;

    simulationTime += dt;
    stats.spawnedThisFrame = 0;
    stats.killedThisFrame = 0;

    // Spawning logic
    if (simulationTime >= settings.startTime && simulationTime <= settings.endTime)
    {
        if (!burstFired && settings.burstCount > 0)
        {
            for (int i = 0; i < settings.burstCount; ++i)
            {
                if (SpawnParticle(settings)) stats.spawnedThisFrame++;
            }
            burstFired = true;
        }

        if (settings.spawnRate > 0.0f)
        {
            spawnAccumulator += settings.spawnRate * dt;
            int spawnsThisFrame = static_cast<int>(spawnAccumulator);
            spawnAccumulator -= spawnsThisFrame;
            for (int i = 0; i < spawnsThisFrame; ++i)
            {
                if (SpawnParticle(settings)) stats.spawnedThisFrame++;
            }
        }
    }

    // Particle update logic
    for (std::size_t i = 0; i < activeCount; )
    {
        Particle& p = particles[i];

        p.lifetime -= dt;
        if (p.lifetime <= 0.0f)
        {
            particles[i] = particles[activeCount - 1];
            activeCount--;
            stats.killedThisFrame++;
            continue;
        }

        // Physics
        float forceX = settings.windX;
        float forceY = settings.windY + settings.gravity;
        
        float ax = forceX / p.mass;
        float ay = forceY / p.mass;

        p.vx += ax * dt;
        p.vy += ay * dt;

        p.vx -= p.vx * settings.drag * dt;
        p.vy -= p.vy * settings.drag * dt;

        p.x += p.vx * dt;
        p.y += p.vy * dt;

        p.rotation += p.angularVelocity * dt;

        // Visual Evolution
        float t = 1.0f - (p.lifetime / p.maxLifetime); // 0.0 to 1.0

        p.size = p.baseSize * Lerp(settings.sizeStart, settings.sizeEnd, t);

        // Gradient logic (4 stops)
        sf::Color gradColor;
        if (t < 0.333333f)
        {
            gradColor = LerpColor(settings.colorGradient[0], settings.colorGradient[1], t * 3.0f);
        }
        else if (t < 0.666666f)
        {
            gradColor = LerpColor(settings.colorGradient[1], settings.colorGradient[2], (t - 0.333333f) * 3.0f);
        }
        else
        {
            gradColor = LerpColor(settings.colorGradient[2], settings.colorGradient[3], (t - 0.666666f) * 3.0f);
        }

        // Apply gradient to base color (multiply)
        p.r = static_cast<std::uint8_t>((p.baseR * gradColor.r) / 255);
        p.g = static_cast<std::uint8_t>((p.baseG * gradColor.g) / 255);
        p.b = static_cast<std::uint8_t>((p.baseB * gradColor.b) / 255);
        p.a = static_cast<std::uint8_t>((p.baseA * gradColor.a) / 255);

        i++;
    }

    stats.activeParticles = static_cast<uint32_t>(activeCount);
}
