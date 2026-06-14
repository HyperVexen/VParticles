#pragma once

#include <SFML/Graphics.hpp>

#include "ParticleSystem.h"

class ParticleRenderer
{
public:
    void Draw(sf::RenderWindow& window, const ParticleSystem& particleSystem) const;
};
