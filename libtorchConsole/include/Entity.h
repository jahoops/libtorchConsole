#include <torch/torch.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include "NeuralNetwork.h"

struct DNA {
    sf::Color color;
    float shapeSize;
    DNA(sf::Color color, float shapeSize) noexcept : color(color), shapeSize(shapeSize) {}
};

class Entity {
public:
    Entity(DNA dna, sf::Vector2f position) noexcept;
    void update(const std::vector<Entity>& entities);
    void draw(sf::RenderWindow& window) const;
    sf::Vector2f getPosition() const;
    sf::Color getColor() const;
    void move(const sf::Vector2f& direction);
    float calculateReward() const;
    void updatePositionWithNeuralNetwork();
    void checkBounds();
    NeuralNetwork& getNeuralNetwork(); // Getter for neuralNetwork

private:
    DNA dna;
    sf::CircleShape shape;
    NeuralNetwork neuralNetwork;
    sf::Vector2f position;
    float speed;
    float reward;
};
