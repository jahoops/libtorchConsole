#include "Entity.h"

Entity::Entity() {
    // Initialize shape
    shape.setRadius(50);
    shape.setFillColor(sf::Color::Green);
    shape.setPosition(375, 275);

    // Initialize DNA (example)
    dna = torch::rand({10}); // Random DNA for demonstration
}

void Entity::update() {
    // Update entity based on DNA and neural network
}

void Entity::draw(sf::RenderWindow& window) {
    window.draw(shape);
}
