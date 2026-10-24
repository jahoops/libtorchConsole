#pragma once
#include <torch/torch.h> // One-stop header.

class NeuralNetwork : public torch::nn::Module {
public:
    NeuralNetwork();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

