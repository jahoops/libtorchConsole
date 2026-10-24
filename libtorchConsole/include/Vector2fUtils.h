// Vector2fUtils.h
#pragma once
#include <SFML/System/Vector2.hpp>

inline sf::Vector2f operator*(const sf::Vector2f& vec, float scalar) {
    return sf::Vector2f(vec.x * scalar, vec.y * scalar);
}
