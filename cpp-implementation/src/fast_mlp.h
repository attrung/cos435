#pragma once

#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <torch/torch.h>

// Raw-array MLP inference for the hot path (action selection).
// Avoids all LibTorch tensor overhead. ~100x faster than torch::Tensor forward.
//
// Layout: fc1(input_size, hidden_size) + relu -> fc2(hidden_size, output_size)
// Weights stored as contiguous arrays, synced from LibTorch parameters.

template<int IN, int HID, int OUT>
class FastMLP {
public:
    // Weight arrays (row-major)
    float w1[HID][IN];    // fc1 weight
    float b1[HID];        // fc1 bias
    float w2[OUT][HID];   // fc2 weight
    float b2[OUT];        // fc2 bias

    // Hidden buffer for intermediate result
    float hidden[HID];

    // Sync weights from a LibTorch MLP module
    void sync_from(torch::nn::Module& module) {
        auto params = module.named_parameters();
        // fc1.weight: (HID, IN), fc1.bias: (HID)
        // fc2.weight: (OUT, HID), fc2.bias: (OUT)
        for (auto& p : params) {
            auto data = p.value().data_ptr<float>();
            if (p.key() == "fc1.weight") {
                std::memcpy(w1, data, HID * IN * sizeof(float));
            } else if (p.key() == "fc1.bias") {
                std::memcpy(b1, data, HID * sizeof(float));
            } else if (p.key() == "fc2.weight") {
                std::memcpy(w2, data, OUT * HID * sizeof(float));
            } else if (p.key() == "fc2.bias") {
                std::memcpy(b2, data, OUT * sizeof(float));
            }
        }
    }

    // Forward pass: input[IN] -> output[OUT]
    void forward(const float* input, float* output) {
        // fc1 + relu
        for (int h = 0; h < HID; ++h) {
            float sum = b1[h];
            for (int i = 0; i < IN; ++i) {
                sum += w1[h][i] * input[i];
            }
            hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
        }
        // fc2
        for (int o = 0; o < OUT; ++o) {
            float sum = b2[o];
            for (int h = 0; h < HID; ++h) {
                sum += w2[o][h] * hidden[h];
            }
            output[o] = sum;
        }
    }

    // Forward with illegal action masking + argmax (for DQN action selection)
    // Skips illegal actions entirely (robust to NaN/inf Q-values)
    int forward_argmax(const float* input, const float* legal_mask) {
        float logits[OUT];
        forward(input, logits);
        int best = -1;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            if (best == -1 || logits[o] > best_val) {
                best_val = logits[o];
                best = o;
            }
        }
        return best;
    }

    // Forward with masking + softmax + sample (for avg policy action selection)
    // Only considers legal actions (robust to NaN/inf logits)
    int forward_sample(const float* input, const float* legal_mask, float rand_val) {
        float logits[OUT];
        forward(input, logits);
        // Find max over legal actions only
        float max_val = -std::numeric_limits<float>::infinity();
        int fallback = -1;
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            if (fallback == -1) fallback = o;
            if (logits[o] > max_val) max_val = logits[o];
        }
        // Softmax over legal actions only
        float sum = 0.0f;
        float probs[OUT] = {};
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            probs[o] = std::exp(logits[o] - max_val);
            sum += probs[o];
        }
        if (sum <= 0.0f || std::isnan(sum)) return fallback;
        for (int o = 0; o < OUT; ++o) probs[o] /= sum;
        // Sample
        float cumsum = 0.0f;
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            cumsum += probs[o];
            if (rand_val < cumsum) return o;
        }
        return OUT - 1;
    }

    // Forward with masking + softmax -> output probs
    void forward_probs(const float* input, const float* legal_mask, float* probs_out) {
        float logits[OUT];
        forward(input, logits);
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) logits[o] = -1e38f;
        }
        float max_val = *std::max_element(logits, logits + OUT);
        float sum = 0.0f;
        for (int o = 0; o < OUT; ++o) {
            probs_out[o] = std::exp(logits[o] - max_val);
            sum += probs_out[o];
        }
        for (int o = 0; o < OUT; ++o) probs_out[o] /= sum;
    }
};

// Deep FastMLP: supports variable number of hidden layers via dynamic allocation.
// Slightly slower than templated FastMLP but supports DeepMLP networks.
template<int IN, int HID, int OUT, int MAX_LAYERS = 4>
class DeepFastMLP {
public:
    float w[MAX_LAYERS][HID][HID];  // layer weights (first layer is [HID][IN] virtually)
    float b[MAX_LAYERS][HID];
    float w_out[OUT][HID];
    float b_out[OUT];
    float w_first[HID][IN];  // separate first layer (different input size)
    float b_first[HID];
    float buf[2][HID];  // ping-pong buffers
    int num_layers = 0;

    void sync_from(torch::nn::Module& module) {
        auto params = module.named_parameters();
        num_layers = 0;
        for (auto& p : params) {
            if (p.key().find("output") != std::string::npos) {
                auto data = p.value().data_ptr<float>();
                if (p.key().find("weight") != std::string::npos)
                    std::memcpy(w_out, data, OUT * HID * sizeof(float));
                else
                    std::memcpy(b_out, data, OUT * sizeof(float));
            } else if (p.key().find("weight") != std::string::npos) {
                int layer = std::stoi(p.key().substr(2, 1));  // "fc0", "fc1", ...
                auto data = p.value().data_ptr<float>();
                if (layer == 0)
                    std::memcpy(w_first, data, HID * IN * sizeof(float));
                else
                    std::memcpy(w[layer], data, HID * HID * sizeof(float));
                num_layers = std::max(num_layers, layer + 1);
            } else if (p.key().find("bias") != std::string::npos) {
                int layer = std::stoi(p.key().substr(2, 1));
                auto data = p.value().data_ptr<float>();
                if (layer == 0)
                    std::memcpy(b_first, data, HID * sizeof(float));
                else
                    std::memcpy(b[layer], data, HID * sizeof(float));
            }
        }
    }

    void forward(const float* input, float* output) {
        // First layer: IN -> HID
        for (int h = 0; h < HID; ++h) {
            float s = b_first[h];
            for (int i = 0; i < IN; ++i) s += w_first[h][i] * input[i];
            buf[0][h] = s > 0 ? s : 0;
        }
        // Middle layers: HID -> HID
        for (int l = 1; l < num_layers; ++l) {
            int src = (l - 1) % 2, dst = l % 2;
            for (int h = 0; h < HID; ++h) {
                float s = b[l][h];
                for (int j = 0; j < HID; ++j) s += w[l][h][j] * buf[src][j];
                buf[dst][h] = s > 0 ? s : 0;
            }
        }
        // Output layer
        int src = (num_layers - 1) % 2;
        for (int o = 0; o < OUT; ++o) {
            float s = b_out[o];
            for (int h = 0; h < HID; ++h) s += w_out[o][h] * buf[src][h];
            output[o] = s;
        }
    }

    int forward_argmax(const float* input, const float* legal_mask) {
        float logits[OUT];
        forward(input, logits);
        int best = -1;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            if (best == -1 || logits[o] > best_val) { best_val = logits[o]; best = o; }
        }
        return best;
    }

    int forward_sample(const float* input, const float* mask, float rand_val) {
        float logits[OUT];
        forward(input, logits);
        float mx = -std::numeric_limits<float>::infinity();
        int fallback = -1;
        for (int o = 0; o < OUT; ++o) {
            if (mask[o] < 0.5f) continue;
            if (fallback == -1) fallback = o;
            if (logits[o] > mx) mx = logits[o];
        }
        float sum = 0, probs[OUT] = {};
        for (int o = 0; o < OUT; ++o) {
            if (mask[o] < 0.5f) continue;
            probs[o] = std::exp(logits[o] - mx); sum += probs[o];
        }
        if (sum <= 0.0f || std::isnan(sum)) return fallback;
        for (int o = 0; o < OUT; ++o) probs[o] /= sum;
        float c = 0;
        for (int o = 0; o < OUT; ++o) { if (mask[o] < 0.5f) continue; c += probs[o]; if (rand_val < c) return o; }
        return fallback;
    }

    void forward_probs(const float* input, const float* mask, float* probs_out) {
        float logits[OUT];
        forward(input, logits);
        for (int o = 0; o < OUT; ++o)
            if (mask[o] < 0.5f) logits[o] = -1e38f;
        float mx = *std::max_element(logits, logits + OUT);
        float sum = 0;
        for (int o = 0; o < OUT; ++o) { probs_out[o] = std::exp(logits[o] - mx); sum += probs_out[o]; }
        for (int o = 0; o < OUT; ++o) probs_out[o] /= sum;
    }
};

// VarFastMLP: fully dynamic layer sizes for networks like [1024, 512, 1024, 512].
// Heap-allocated weights. Slower than templated versions but supports any architecture.
template<int IN, int OUT>
class VarFastMLP {
public:
    struct Layer {
        std::vector<float> w;  // [out_size][in_size] row-major
        std::vector<float> b;  // [out_size]
        int in_size, out_size;
    };

    std::vector<Layer> hidden_layers;
    Layer output_layer;
    std::vector<float> buf[2];
    int max_hidden = 0;

    void init(const std::vector<int>& hidden_sizes) {
        hidden_layers.resize(hidden_sizes.size());
        int in_sz = IN;
        max_hidden = 0;
        for (size_t i = 0; i < hidden_sizes.size(); ++i) {
            int h = hidden_sizes[i];
            hidden_layers[i].w.resize(h * in_sz);
            hidden_layers[i].b.resize(h);
            hidden_layers[i].in_size = in_sz;
            hidden_layers[i].out_size = h;
            if (h > max_hidden) max_hidden = h;
            in_sz = h;
        }
        output_layer.w.resize(OUT * in_sz);
        output_layer.b.resize(OUT);
        output_layer.in_size = in_sz;
        output_layer.out_size = OUT;
        buf[0].resize(max_hidden);
        buf[1].resize(max_hidden);
    }

    void sync_from(torch::nn::Module& module) {
        auto params = module.named_parameters();
        for (auto& p : params) {
            auto data = p.value().data_ptr<float>();
            int numel = p.value().numel();
            if (p.key().find("output") != std::string::npos) {
                if (p.key().find("weight") != std::string::npos)
                    std::memcpy(output_layer.w.data(), data, numel * sizeof(float));
                else
                    std::memcpy(output_layer.b.data(), data, numel * sizeof(float));
            } else {
                // Parse layer index from "fc0", "fc1", etc.
                size_t idx_start = p.key().find("fc") + 2;
                size_t idx_end = p.key().find('.', idx_start);
                int layer = std::stoi(p.key().substr(idx_start, idx_end - idx_start));
                if (layer < (int)hidden_layers.size()) {
                    if (p.key().find("weight") != std::string::npos)
                        std::memcpy(hidden_layers[layer].w.data(), data, numel * sizeof(float));
                    else
                        std::memcpy(hidden_layers[layer].b.data(), data, numel * sizeof(float));
                }
            }
        }
    }

    void forward(const float* input, float* output) {
        // First hidden layer
        const float* src = input;
        int src_size = IN;
        for (size_t l = 0; l < hidden_layers.size(); ++l) {
            auto& layer = hidden_layers[l];
            float* dst = buf[l % 2].data();
            for (int h = 0; h < layer.out_size; ++h) {
                float s = layer.b[h];
                const float* row = &layer.w[h * src_size];
                for (int j = 0; j < src_size; ++j) s += row[j] * src[j];
                dst[h] = s > 0 ? s : 0;  // ReLU
            }
            src = dst;
            src_size = layer.out_size;
        }
        // Output layer (no activation)
        for (int o = 0; o < OUT; ++o) {
            float s = output_layer.b[o];
            const float* row = &output_layer.w[o * src_size];
            for (int j = 0; j < src_size; ++j) s += row[j] * src[j];
            output[o] = s;
        }
    }

    int forward_argmax(const float* input, const float* legal_mask) {
        float logits[OUT];
        forward(input, logits);
        int best = -1;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int o = 0; o < OUT; ++o) {
            if (legal_mask[o] < 0.5f) continue;
            if (best == -1 || logits[o] > best_val) { best_val = logits[o]; best = o; }
        }
        return best;
    }

    int forward_sample(const float* input, const float* mask, float rand_val) {
        float logits[OUT];
        forward(input, logits);
        float mx = -std::numeric_limits<float>::infinity();
        int fallback = -1;
        for (int o = 0; o < OUT; ++o) {
            if (mask[o] < 0.5f) continue;
            if (fallback == -1) fallback = o;
            if (logits[o] > mx) mx = logits[o];
        }
        float sum = 0, probs[OUT] = {};
        for (int o = 0; o < OUT; ++o) {
            if (mask[o] < 0.5f) continue;
            probs[o] = std::exp(logits[o] - mx); sum += probs[o];
        }
        if (sum <= 0.0f || std::isnan(sum)) return fallback;
        for (int o = 0; o < OUT; ++o) probs[o] /= sum;
        float c = 0;
        for (int o = 0; o < OUT; ++o) { if (mask[o] < 0.5f) continue; c += probs[o]; if (rand_val < c) return o; }
        return fallback;
    }

    void forward_probs(const float* input, const float* mask, float* probs_out) {
        float logits[OUT];
        forward(input, logits);
        for (int o = 0; o < OUT; ++o)
            if (mask[o] < 0.5f) logits[o] = -std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (int o = 0; o < OUT; ++o) if (logits[o] > mx) mx = logits[o];
        float sum = 0;
        for (int o = 0; o < OUT; ++o) { probs_out[o] = std::exp(logits[o] - mx); sum += probs_out[o]; }
        if (sum > 0) for (int o = 0; o < OUT; ++o) probs_out[o] /= sum;
    }
};
