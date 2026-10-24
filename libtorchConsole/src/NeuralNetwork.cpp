#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
    fc1 = register_module("fc1", torch::nn::Linear(10, 50));
    fc2 = register_module("fc2", torch::nn::Linear(50, 1));
}

torch::Tensor NeuralNetwork::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::sigmoid(fc2->forward(x));
    return x;
}
