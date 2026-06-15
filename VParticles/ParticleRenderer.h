#pragma once

#include <SFML/Graphics.hpp>

#include "ParticleSystem.h"

class ParticleRenderer
{
public:
    ParticleRenderer();

    void Draw(sf::RenderWindow& window, const ParticleSystem& particleSystem);

private:
    sf::VertexArray m_vertices;
};
