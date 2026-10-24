#pragma once
#include <SFML/Graphics.hpp>

struct DNA {
    sf::Color color;
    float shapeSize;

    // Constructor using member initializer list
    DNA(sf::Color color, float shapeSize) noexcept : color(color), shapeSize(shapeSize) {}
};
