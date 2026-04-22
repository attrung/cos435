#pragma once

#include <array>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>

// Transition for DQN replay buffer
template<int S, int A>
struct Transition {
    std::array<float, S> state;
    int action;
    float reward;
    std::array<float, S> next_state;
    float done;
    std::array<float, A> legal_mask;
};

// Circular replay buffer (ring buffer) for DQN
template<int S, int A>
class CircularReplayBuffer {
public:
    explicit CircularReplayBuffer(int capacity)
        : capacity_(capacity), size_(0), pos_(0), data_(capacity) {}

    void add(const std::array<float, S>& state, int action, float reward,
             const std::array<float, S>& next_state, float done,
             const std::array<float, A>& legal_mask) {
        auto& t = data_[pos_];
        t.state = state;
        t.action = action;
        t.reward = reward;
        t.next_state = next_state;
        t.done = done;
        t.legal_mask = legal_mask;
        pos_ = (pos_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }

    // Sample batch into pre-allocated output arrays
    void sample(int batch_size, std::mt19937& rng,
                float* states, int64_t* actions, float* rewards,
                float* next_states, float* dones, float* legal_masks) const {
        std::uniform_int_distribution<int> dist(0, size_ - 1);
        for (int b = 0; b < batch_size; ++b) {
            int idx = dist(rng);
            const auto& t = data_[idx];
            std::memcpy(states + b * S, t.state.data(), S * sizeof(float));
            actions[b] = t.action;
            rewards[b] = t.reward;
            std::memcpy(next_states + b * S, t.next_state.data(), S * sizeof(float));
            dones[b] = t.done;
            std::memcpy(legal_masks + b * A, t.legal_mask.data(), A * sizeof(float));
        }
    }

    int size() const { return size_; }
    int position() const { return pos_; }
    size_t entry_bytes() const { return sizeof(Transition<S, A>); }
    const void* raw_data() const { return data_.data(); }
    void load_raw(int sz, int pos, std::istream& f) {
        size_ = sz; pos_ = pos;
        f.read((char*)data_.data(), sz * sizeof(Transition<S, A>));
    }

private:
    int capacity_, size_, pos_;
    std::vector<Transition<S, A>> data_;
};

// Reservoir sampling entry for average policy
template<int S, int A>
struct ReservoirEntry {
    std::array<float, S> state;
    std::array<float, A> action_probs;
    std::array<float, A> legal_mask;
};

// Reservoir buffer (Algorithm R) for average policy
template<int S, int A>
class ReservoirBuffer {
public:
    explicit ReservoirBuffer(int capacity)
        : capacity_(capacity), size_(0), total_seen_(0), data_(capacity) {}

    void add(const std::array<float, S>& state,
             const std::array<float, A>& action_probs,
             const std::array<float, A>& legal_mask, std::mt19937& rng) {
        total_seen_++;
        if (size_ < capacity_) {
            auto& e = data_[size_];
            e.state = state;
            e.action_probs = action_probs;
            e.legal_mask = legal_mask;
            size_++;
        } else {
            std::uniform_int_distribution<int64_t> dist(0, total_seen_ - 1);
            int64_t idx = dist(rng);
            if (idx < capacity_) {
                auto& e = data_[idx];
                e.state = state;
                e.action_probs = action_probs;
                e.legal_mask = legal_mask;
            }
        }
    }

    void sample(int batch_size, std::mt19937& rng,
                float* states, float* action_probs, float* legal_masks) const {
        std::uniform_int_distribution<int> dist(0, size_ - 1);
        for (int b = 0; b < batch_size; ++b) {
            int idx = dist(rng);
            const auto& e = data_[idx];
            std::memcpy(states + b * S, e.state.data(), S * sizeof(float));
            std::memcpy(action_probs + b * A, e.action_probs.data(), A * sizeof(float));
            std::memcpy(legal_masks + b * A, e.legal_mask.data(), A * sizeof(float));
        }
    }

    int size() const { return size_; }
    int64_t total_seen() const { return total_seen_; }
    size_t entry_bytes() const { return sizeof(ReservoirEntry<S, A>); }
    const void* raw_data() const { return data_.data(); }
    void load_raw(int sz, int64_t seen, std::istream& f) {
        size_ = sz; total_seen_ = seen;
        f.read((char*)data_.data(), sz * sizeof(ReservoirEntry<S, A>));
    }

private:
    int capacity_, size_;
    int64_t total_seen_;
    std::vector<ReservoirEntry<S, A>> data_;
};
