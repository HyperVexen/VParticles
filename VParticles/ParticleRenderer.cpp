#include "ParticleRenderer.h"
#include "SimulationSettings.h"
#include <cmath>

ParticleRenderer::ParticleRenderer()
{
    m_vertices.setPrimitiveType(sf::PrimitiveType::Triangles);
}

void ParticleRenderer::Draw(
    sf::RenderWindow& window,
    const ParticleSystem& particleSystem)
{
    std::size_t vertexCount = particleSystem.activeCount * 6;
    
    if (m_vertices.getVertexCount() != vertexCount)
    {
        m_vertices.resize(vertexCount);
    }

    for (std::size_t i = 0; i < particleSystem.activeCount; ++i)
    {
        const Particle& p = particleSystem.particles[i];
        std::size_t vertexIndex = i * 6;
        
        float halfSize = p.size * 0.5f;
        
        sf::Color color(p.r, p.g, p.b, p.a);

        // Precompute rotation
        float c = std::cos(p.rotation);
        float s = std::sin(p.rotation);

        auto rotate = [c, s](float x, float y) -> sf::Vector2f {
            return sf::Vector2f(x * c - y * s, x * s + y * c);
        };

        // Top-Left
        sf::Vector2f tl = rotate(-halfSize, -halfSize);
        tl.x += p.x; tl.y += p.y;
        // Top-Right
        sf::Vector2f tr = rotate(halfSize, -halfSize);
        tr.x += p.x; tr.y += p.y;
        // Bottom-Right
        sf::Vector2f br = rotate(halfSize, halfSize);
        br.x += p.x; br.y += p.y;
        // Bottom-Left
        sf::Vector2f bl = rotate(-halfSize, halfSize);
        bl.x += p.x; bl.y += p.y;

        // Triangle 1: TL, TR, BR
        m_vertices[vertexIndex].position = tl;
        m_vertices[vertexIndex + 1].position = tr;
        m_vertices[vertexIndex + 2].position = br;

        // Triangle 2: TL, BR, BL
        m_vertices[vertexIndex + 3].position = tl;
        m_vertices[vertexIndex + 4].position = br;
        m_vertices[vertexIndex + 5].position = bl;

        for (int j = 0; j < 6; ++j)
        {
            m_vertices[vertexIndex + j].color = color;
        }
    }

    window.draw(m_vertices);
}
