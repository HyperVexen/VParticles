#include "ParticleRenderer.h"

#include "SimulationSettings.h"

void ParticleRenderer::Draw(
    sf::RenderWindow& window,
    const ParticleSystem& particleSystem) const
{
    sf::CircleShape dot(SimulationDefaults::ParticleRadius);
    dot.setOrigin({
        SimulationDefaults::ParticleRadius,
        SimulationDefaults::ParticleRadius
    });

    for (const auto& p : particleSystem.particles)
    {
        dot.setPosition({ p.x, p.y });

        window.draw(dot);
    }
}
