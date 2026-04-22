#pragma once

#include <torch/torch.h>

// Simple MLP: input -> hidden (ReLU) -> output
// Matches OpenSpiel/Python: single hidden layer [128]
struct MLPImpl : torch::nn::Module {
    MLPImpl(int input_size, int hidden_size, int output_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, output_size))) {}

    torch::Tensor forward(torch::Tensor x) {
        return fc2(torch::relu(fc1(x)));
    }

    torch::nn::Linear fc1, fc2;
};
TORCH_MODULE(MLP);

// Deep MLP: input -> [hidden (ReLU)]×N -> output
// Supports variable number of hidden layers
struct DeepMLPImpl : torch::nn::Module {
    DeepMLPImpl(int input_size, int hidden_size, int output_size, int num_layers = 4) {
        int in_sz = input_size;
        for (int i = 0; i < num_layers; ++i) {
            layers.push_back(register_module("fc" + std::to_string(i),
                             torch::nn::Linear(in_sz, hidden_size)));
            in_sz = hidden_size;
        }
        output = register_module("output", torch::nn::Linear(in_sz, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        for (auto& layer : layers)
            x = torch::relu(layer->forward(x));
        return output(x);
    }

    std::vector<torch::nn::Linear> layers;
    torch::nn::Linear output{nullptr};
};
TORCH_MODULE(DeepMLP);

// Variable MLP: input -> [h1 (ReLU)] -> [h2 (ReLU)] -> ... -> output
// Supports different sizes per layer, e.g. [1024, 512, 1024, 512]
struct VarMLPImpl : torch::nn::Module {
    VarMLPImpl(int input_size, std::vector<int> hidden_sizes, int output_size) {
        int in_sz = input_size;
        for (size_t i = 0; i < hidden_sizes.size(); ++i) {
            layers.push_back(register_module("fc" + std::to_string(i),
                             torch::nn::Linear(in_sz, hidden_sizes[i])));
            in_sz = hidden_sizes[i];
        }
        output = register_module("output", torch::nn::Linear(in_sz, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        for (auto& layer : layers)
            x = torch::relu(layer->forward(x));
        return output(x);
    }

    std::vector<torch::nn::Linear> layers;
    torch::nn::Linear output{nullptr};
};
TORCH_MODULE(VarMLP);

// IQN Network: state + cosine tau embedding -> quantile values
struct IQNNetworkImpl : torch::nn::Module {
    IQNNetworkImpl(int input_size, int hidden_size, int output_size, int embedding_dim = 64)
        : embedding_dim_(embedding_dim),
          state_fc(register_module("state_fc", torch::nn::Linear(input_size, hidden_size))),
          cos_embedding(register_module("cos_embedding", torch::nn::Linear(embedding_dim, hidden_size))),
          output_fc(register_module("output_fc", torch::nn::Linear(hidden_size, output_size))) {}

    // state: (batch, input_size), taus: (batch, num_quantiles)
    // returns: (batch, num_quantiles, num_actions)
    torch::Tensor forward(torch::Tensor state, torch::Tensor taus) {
        int batch_size = state.size(0);
        int num_quantiles = taus.size(1);

        // State features: (batch, hidden)
        auto state_features = torch::relu(state_fc(state));

        // Cosine embedding: (batch, num_quantiles, embedding_dim)
        auto i_pi = torch::arange(0, embedding_dim_, torch::kFloat32).to(taus.device()) * M_PI;
        auto cos_input = taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0);
        auto tau_features = torch::relu(cos_embedding(torch::cos(cos_input)));

        // Element-wise product: (batch, num_quantiles, hidden)
        auto combined = state_features.unsqueeze(1) * tau_features;

        // Output: (batch, num_quantiles, num_actions)
        return output_fc(combined);
    }

    int embedding_dim_;
    torch::nn::Linear state_fc, cos_embedding, output_fc;
};
TORCH_MODULE(IQNNetwork);
