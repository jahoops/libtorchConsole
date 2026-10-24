#include <SFML/Graphics.hpp>
#include "Entity.h"

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Evolving Entities");

    Entity entity;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        entity.update();

        window.clear(sf::Color::Black);
        entity.draw(window);
        window.display();
    }

    return 0;
}
