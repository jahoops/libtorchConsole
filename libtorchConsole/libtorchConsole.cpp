#include "Entity.h"
#include <torch/torch.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include <deque>
#include <random>
#include <iostream> // For debugging

struct Experience {
    torch::Tensor state;
    torch::Tensor action;
    float reward;
    torch::Tensor nextState;
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity) : capacity(capacity) {}

    void add(const Experience& experience) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back(experience);
    }

    std::vector<Experience> sample(size_t batchSize) {
        std::vector<Experience> batch;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch), batchSize, std::mt19937{ std::random_device{}() });
        return batch;
    }

    size_t size() const {
        return buffer.size();
    }

private:
    size_t capacity;
    std::deque<Experience> buffer;
};

void train(Entity& entity, ReplayBuffer& replayBuffer, torch::optim::Optimizer& optimizer, size_t batchSize) {
    if (replayBuffer.size() < batchSize) return;

    auto batch = replayBuffer.sample(batchSize);
    std::vector<torch::Tensor> states, actions, rewards, nextStates;

    for (const auto& exp : batch) {
        states.push_back(exp.state);
        actions.push_back(exp.action.argmax(1).unsqueeze(1)); // Ensure action is a single value per sample
        rewards.push_back(torch::tensor(exp.reward));
        nextStates.push_back(exp.nextState);
    }

    auto stateBatch = torch::stack(states).squeeze(1); // Remove the extra dimension
    auto actionBatch = torch::stack(actions).squeeze(2).to(torch::kLong); // Remove the extra dimension and ensure type Long
    auto rewardBatch = torch::stack(rewards);
    auto nextStateBatch = torch::stack(nextStates).squeeze(1); // Remove the extra dimension

    // Debugging information
    /*std::cout << "stateBatch shape: " << stateBatch.sizes() << std::endl;
    std::cout << "actionBatch shape: " << actionBatch.sizes() << std::endl;
    std::cout << "rewardBatch shape: " << rewardBatch.sizes() << std::endl;
    std::cout << "nextStateBatch shape: " << nextStateBatch.sizes() << std::endl;
    */
    auto qValues = entity.getNeuralNetwork().forward(stateBatch).squeeze(1); // Reshape to 2D
    auto nextQValues = entity.getNeuralNetwork().forward(nextStateBatch).detach().squeeze(1); // Reshape to 2D

    // Ensure qValues is a 2D tensor
    if (qValues.dim() != 2) {
        std::cerr << "Error: qValues is not a 2D tensor. It has " << qValues.dim() << " dimensions with shape: " << qValues.sizes() << std::endl;
        return;
    }

    auto targetQValues = rewardBatch.unsqueeze(1) + 0.99 * std::get<0>(nextQValues.max(1)).unsqueeze(1);

    // Ensure actionBatch has the correct shape for gather
    if (actionBatch.dim() == 1) {
        actionBatch = actionBatch.unsqueeze(1);
    }

    auto loss = torch::mse_loss(qValues.gather(1, actionBatch), targetQValues);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}

int main() {
    // Create an SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Neural Network Training");

    DNA dna(sf::Color::Red, 10.0f);
    std::vector<Entity> entities;
    ReplayBuffer replayBuffer(10000);

    // Initialize multiple entities
    for (int i = 0; i < 10; ++i) {
        entities.emplace_back(dna, sf::Vector2f(400.0f, 300.0f));
    }

    // Ensure the entities vector is not empty before accessing its elements
    if (entities.empty()) {
        std::cerr << "Error: No entities initialized." << std::endl;
        return -1;
    }

    // Initialize the optimizer with the parameters of the first entity's neural network
    torch::optim::Adam optimizer(entities[0].getNeuralNetwork().parameters(), torch::optim::AdamOptions(0.001));

    for (int episode = 0; episode < 1000; ++episode) {
        // Draw all entities
        window.clear();
        for (auto& entity : entities) {
            entity.draw(window);
        }
        window.display();

        for (int t = 0; t < 200; ++t) {
            for (auto& entity : entities) {
                auto state = torch::tensor({ entity.getPosition().x, entity.getPosition().y }).unsqueeze(0);
                auto action = entity.getNeuralNetwork().forward(state).detach();

                // Determine direction based on action
                sf::Vector2f direction(0.0f, 0.0f);
                int actionIndex = action.argmax(1).item<int>();
                switch (actionIndex) {
                case 0: direction = sf::Vector2f(0.0f, -1.0f); break; // North
                case 1: direction = sf::Vector2f(0.0f, 1.0f); break;  // South
                case 2: direction = sf::Vector2f(1.0f, 0.0f); break;  // East
                case 3: direction = sf::Vector2f(-1.0f, 0.0f); break; // West
                case 4: direction = sf::Vector2f(1.0f, -1.0f); break; // Northeast
                case 5: direction = sf::Vector2f(1.0f, 1.0f); break;  // Southeast
                case 6: direction = sf::Vector2f(-1.0f, -1.0f); break;// Northwest
                case 7: direction = sf::Vector2f(-1.0f, 1.0f); break; // Southwest
                }

                entity.move(direction);
                float reward = entity.calculateReward();
                auto nextState = torch::tensor({ entity.getPosition().x, entity.getPosition().y }).unsqueeze(0);
                replayBuffer.add({ state, action, reward, nextState });
                train(entity, replayBuffer, optimizer, 64);
            }

            // Clear the window
            window.clear();

            // Draw all entities
            for (auto& entity : entities) {
                entity.draw(window);
            }

            // Display the contents of the window
            window.display();

            // Handle window events
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                    return 0;
                }
            }
        }
    }

    return 0;
}

