#include <SFML/Graphics.hpp>
#include <optional>

#include "ParticleSystem.h"

int main()
{
    sf::RenderWindow window(
        sf::VideoMode({ 1280, 720 }),
        "VParticles"
    );

    ParticleSystem particleSystem;

    particleSystem.Create(1000);

    sf::Clock clock;

    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();

        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        auto size = window.getSize();

        particleSystem.Update(
            dt,
            static_cast<float>(size.x),
            static_cast<float>(size.y)
        );

        window.clear();

        for (const auto& p : particleSystem.particles)
        {
            sf::CircleShape dot(2.f);

            dot.setPosition({ p.x, p.y });

            window.draw(dot);
        }

        window.display();
    }

    return 0;
}