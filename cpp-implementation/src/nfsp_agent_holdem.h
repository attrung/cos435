#pragma once

// NFSP Agent for Hold'em with variable-size MLP (e.g. [1024, 512, 1024, 512]).
// Uses VarMLP (LibTorch) for gradient updates, VarFastMLP for fast worker inference.

#include <random>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <torch/torch.h>
#include "networks.h"
#include "fast_mlp.h"
#include "replay_buffer.h"
#include "holdem.h"

class NFSPAgent {
public:
    NFSPAgent(int player_id, float eta, float dqn_lr, float avg_lr,
              std::vector<int> hidden_sizes, int batch_size, int dqn_buffer_size,
              int reservoir_buffer_size, int learn_every, int min_buffer_size,
              int target_update_freq, float tau, float epsilon_start,
              float epsilon_end, int epsilon_decay_duration, float gamma,
              int num_dqn_updates = 2)
        : player_id_(player_id), eta_(eta), gamma_(gamma),
          batch_size_(batch_size), learn_every_(learn_every),
          min_buffer_size_(min_buffer_size),
          target_update_freq_(target_update_freq), tau_(tau),
          epsilon_start_(epsilon_start), epsilon_end_(epsilon_end),
          epsilon_decay_duration_(epsilon_decay_duration),
          num_dqn_updates_(num_dqn_updates),
          hidden_sizes_(hidden_sizes),
          q_net_(INFO_STATE_SIZE, hidden_sizes, NUM_ACTIONS),
          target_net_(INFO_STATE_SIZE, hidden_sizes, NUM_ACTIONS),
          avg_net_(INFO_STATE_SIZE, hidden_sizes, NUM_ACTIONS),
          q_optimizer_(q_net_->parameters(), torch::optim::AdamOptions(dqn_lr)),
          avg_optimizer_(avg_net_->parameters(), torch::optim::AdamOptions(avg_lr)),
          dqn_buffer_(dqn_buffer_size),
          reservoir_buffer_(reservoir_buffer_size),
          game_steps_(0), br_steps_(0), dqn_steps_(0),
          mode_(AVG_POLICY),
          dqn_s_(torch::zeros({batch_size, INFO_STATE_SIZE})),
          dqn_a_(torch::zeros({batch_size}, torch::kInt64)),
          dqn_r_(torch::zeros({batch_size})),
          dqn_ns_(torch::zeros({batch_size, INFO_STATE_SIZE})),
          dqn_d_(torch::zeros({batch_size})),
          dqn_lm_(torch::zeros({batch_size, NUM_ACTIONS})),
          avg_s_(torch::zeros({batch_size, INFO_STATE_SIZE})),
          avg_ap_(torch::zeros({batch_size, NUM_ACTIONS})),
          avg_lm_(torch::zeros({batch_size, NUM_ACTIONS})) {
        fast_q_ = std::make_unique<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>>();
        fast_avg_ = std::make_unique<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>>();
        fast_q_->init(hidden_sizes);
        fast_avg_->init(hidden_sizes);

        torch::NoGradGuard no_grad;
        auto q_params = q_net_->parameters();
        auto t_params = target_net_->parameters();
        for (size_t i = 0; i < q_params.size(); ++i) {
            t_params[i].copy_(q_params[i]);
        }
        sync_fast_weights();
    }

    enum Mode { BEST_RESPONSE, AVG_POLICY };

    void sample_episode_mode(std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        mode_ = (dist(rng) < eta_) ? BEST_RESPONSE : AVG_POLICY;
    }

    Mode mode() const { return mode_; }

    std::pair<int, std::array<float, NUM_ACTIONS>>
    step(const std::array<float, INFO_STATE_SIZE>& state,
         const std::vector<int>& legal_actions,
         const std::array<float, NUM_ACTIONS>& legal_mask,
         bool is_eval, std::mt19937& rng) {
        if (is_eval || mode_ == AVG_POLICY) {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            int action = fast_avg_->forward_sample(state.data(), legal_mask.data(), dist(rng));
            return {action, {}};
        }
        return dqn_action_with_probs(state, legal_actions, legal_mask, rng);
    }

    void add_transition(const std::array<float, INFO_STATE_SIZE>& state, int action,
                        float reward, const std::array<float, INFO_STATE_SIZE>& next_state,
                        float done, const std::array<float, NUM_ACTIONS>& legal_mask) {
        std::lock_guard<std::mutex> lock(buffer_mtx_);
        dqn_buffer_.add(state, action, reward, next_state, done, legal_mask);
    }

    void add_reservoir(const std::array<float, INFO_STATE_SIZE>& state,
                       const std::array<float, NUM_ACTIONS>& probs,
                       const std::array<float, NUM_ACTIONS>& legal_mask,
                       std::mt19937& rng) {
        std::lock_guard<std::mutex> lock(buffer_mtx_);
        reservoir_buffer_.add(state, probs, legal_mask, rng);
    }

    // Thread-safe version accepting separate game and BR step deltas.
    // Called from the dedicated gradient thread.
    bool catch_up_steps(int new_game_steps, int new_br_steps, std::mt19937& rng) {
        int old_steps = game_steps_;
        game_steps_ += new_game_steps;
        int old_br = br_steps_;
        br_steps_ += new_br_steps;

        bool did_learn = false;

        // Game-step learning: both DQN and avg policy updates
        int g_crossings = (game_steps_ / learn_every_) - (old_steps / learn_every_);
        for (int i = 0; i < g_crossings; ++i) {
            for (int u = 0; u < num_dqn_updates_; ++u) update_dqn(rng);
            for (int u = 0; u < num_dqn_updates_; ++u) update_avg_policy(rng);
            did_learn = true;
        }

        // Extra DQN-only learning triggered by BR step counter
        int old_dqn = dqn_steps_;
        dqn_steps_ += new_br_steps;  // matches old "is_br_mode" behaviour
        int b_crossings = (dqn_steps_ / learn_every_) - (old_dqn / learn_every_);
        for (int i = 0; i < b_crossings; ++i) {
            for (int u = 0; u < num_dqn_updates_; ++u) update_dqn(rng);
            did_learn = true;
        }

        // Target network refit
        int t_crossings = (br_steps_ / target_update_freq_) - (old_br / target_update_freq_);
        if (t_crossings > 0) {
            soft_update_target();
            did_learn = true;
        }

        return did_learn;
    }

    // Deprecated old interface (still used by Leduc train.cpp); wraps catch_up_steps.
    bool increment_steps(int n, bool is_br_mode, std::mt19937& rng) {
        bool learned = catch_up_steps(n, is_br_mode ? n : 0, rng);
        if (learned) sync_fast_weights();
        return learned;
    }

    std::array<float, NUM_ACTIONS>
    get_avg_policy_probs(const std::array<float, INFO_STATE_SIZE>& state,
                         const std::vector<int>& legal_actions) {
        std::array<float, NUM_ACTIONS> mask{};
        for (int a : legal_actions) mask[a] = 1.0f;
        std::array<float, NUM_ACTIONS> probs{};
        fast_avg_->forward_probs(state.data(), mask.data(), probs.data());
        return probs;
    }

    void sync_fast_to(VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>& ext_q,
                      VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>& ext_avg) {
        ext_q.sync_from(*q_net_);
        ext_avg.sync_from(*avg_net_);
    }

    const std::vector<int>& hidden_sizes() const { return hidden_sizes_; }

    int br_steps_val() const { return br_steps_; }

    void save_weights(const std::string& path) {
        torch::save(avg_net_, path + "_avg.pt");
        torch::save(q_net_, path + "_q.pt");
    }

    // Load just network weights (no optimizer state, no buffers). Used when
    // freezing this agent as an opponent (no further training).
    void load_frozen_weights(const std::string& avg_path, const std::string& q_path) {
        torch::load(avg_net_, avg_path);
        torch::load(q_net_, q_path);
        torch::load(target_net_, q_path);  // target = q_net for frozen agent
        sync_fast_weights();
    }

    void save_checkpoint(const std::string& dir) {
        std::system(("mkdir -p " + dir).c_str());
        torch::save(q_net_, dir + "/q_net.pt");
        torch::save(target_net_, dir + "/target_net.pt");
        torch::save(avg_net_, dir + "/avg_net.pt");
        torch::save(q_optimizer_, dir + "/q_opt.pt");
        torch::save(avg_optimizer_, dir + "/avg_opt.pt");
        std::ofstream f(dir + "/state.bin", std::ios::binary);
        f.write((char*)&game_steps_, sizeof(int));
        f.write((char*)&br_steps_, sizeof(int));
        f.write((char*)&dqn_steps_, sizeof(int));
        int dqn_sz = dqn_buffer_.size(), dqn_pos = dqn_buffer_.position();
        f.write((char*)&dqn_sz, sizeof(int));
        f.write((char*)&dqn_pos, sizeof(int));
        f.write((char*)dqn_buffer_.raw_data(), dqn_sz * dqn_buffer_.entry_bytes());
        int res_sz = reservoir_buffer_.size();
        int64_t res_seen = reservoir_buffer_.total_seen();
        f.write((char*)&res_sz, sizeof(int));
        f.write((char*)&res_seen, sizeof(int64_t));
        f.write((char*)reservoir_buffer_.raw_data(), res_sz * reservoir_buffer_.entry_bytes());
    }

    bool load_checkpoint(const std::string& dir) {
        std::ifstream f(dir + "/state.bin", std::ios::binary);
        if (!f.is_open()) return false;
        torch::load(q_net_, dir + "/q_net.pt");
        torch::load(target_net_, dir + "/target_net.pt");
        torch::load(avg_net_, dir + "/avg_net.pt");
        torch::load(q_optimizer_, dir + "/q_opt.pt");
        torch::load(avg_optimizer_, dir + "/avg_opt.pt");
        f.read((char*)&game_steps_, sizeof(int));
        f.read((char*)&br_steps_, sizeof(int));
        f.read((char*)&dqn_steps_, sizeof(int));
        int dqn_sz, dqn_pos;
        f.read((char*)&dqn_sz, sizeof(int));
        f.read((char*)&dqn_pos, sizeof(int));
        dqn_buffer_.load_raw(dqn_sz, dqn_pos, f);
        int res_sz; int64_t res_seen;
        f.read((char*)&res_sz, sizeof(int));
        f.read((char*)&res_seen, sizeof(int64_t));
        reservoir_buffer_.load_raw(res_sz, res_seen, f);
        sync_fast_weights();
        return true;
    }

    float last_br_loss() const { return last_br_loss_.load(std::memory_order_relaxed); }
    float last_avg_loss() const { return last_avg_loss_.load(std::memory_order_relaxed); }
    int game_steps() const { return game_steps_; }

    std::vector<torch::Tensor> get_avg_params() { return avg_net_->parameters(); }

private:
    void sync_fast_weights() {
        fast_q_->sync_from(*q_net_);
        fast_avg_->sync_from(*avg_net_);
    }

    float get_epsilon() const {
        int t = std::min(br_steps_, epsilon_decay_duration_);
        return epsilon_end_ + (epsilon_start_ - epsilon_end_) *
               std::exp(-(float)t / epsilon_decay_duration_);
    }

    std::pair<int, std::array<float, NUM_ACTIONS>>
    dqn_action_with_probs(const std::array<float, INFO_STATE_SIZE>& state,
                          const std::vector<int>& legal_actions,
                          const std::array<float, NUM_ACTIONS>& legal_mask,
                          std::mt19937& rng) {
        std::array<float, NUM_ACTIONS> probs{};
        float epsilon = get_epsilon();
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        if (uniform(rng) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, (int)legal_actions.size() - 1);
            int action = legal_actions[action_dist(rng)];
            for (int a : legal_actions) probs[a] = 1.0f / legal_actions.size();
            return {action, probs};
        }

        int action = fast_q_->forward_argmax(state.data(), legal_mask.data());
        probs[action] = 1.0f;
        return {action, probs};
    }

    void update_dqn(std::mt19937& rng) {
        {
            std::lock_guard<std::mutex> lock(buffer_mtx_);
            if (dqn_buffer_.size() < std::max(batch_size_, min_buffer_size_)) return;
            dqn_buffer_.sample(batch_size_, rng,
                              dqn_s_.data_ptr<float>(), dqn_a_.data_ptr<int64_t>(),
                              dqn_r_.data_ptr<float>(), dqn_ns_.data_ptr<float>(),
                              dqn_d_.data_ptr<float>(), dqn_lm_.data_ptr<float>());
        }

        auto& s = dqn_s_; auto& a = dqn_a_; auto& r = dqn_r_;
        auto& ns = dqn_ns_; auto& d = dqn_d_; auto& lm = dqn_lm_;

        auto q_values = q_net_->forward(s).gather(1, a.unsqueeze(1)).squeeze(1);

        torch::Tensor targets;
        {
            torch::NoGradGuard no_grad;
            auto next_q = target_net_->forward(ns);
            next_q = next_q.masked_fill(lm < 0.5f, -std::numeric_limits<float>::infinity());
            auto next_q_max = std::get<0>(next_q.max(1));
            targets = r + gamma_ * (1.0f - d) * next_q_max;
        }

        auto loss = torch::mse_loss(q_values, targets);
        q_optimizer_.zero_grad();
        loss.backward();
        q_optimizer_.step();

        last_br_loss_.store(loss.item<float>(), std::memory_order_relaxed);
    }

    void update_avg_policy(std::mt19937& rng) {
        {
            std::lock_guard<std::mutex> lock(buffer_mtx_);
            if (reservoir_buffer_.size() < std::max(batch_size_, min_buffer_size_)) return;
            reservoir_buffer_.sample(batch_size_, rng,
                                    avg_s_.data_ptr<float>(), avg_ap_.data_ptr<float>(),
                                    avg_lm_.data_ptr<float>());
        }

        auto& s = avg_s_; auto& ap = avg_ap_; auto& lm = avg_lm_;

        auto logits = avg_net_->forward(s);
        logits = logits + (lm - 1.0f) * 1e38f;
        auto loss = torch::nn::functional::cross_entropy(logits, ap);

        avg_optimizer_.zero_grad();
        loss.backward();
        avg_optimizer_.step();

        last_avg_loss_.store(loss.item<float>(), std::memory_order_relaxed);
    }

    void soft_update_target() {
        torch::NoGradGuard no_grad;
        auto q_params = q_net_->parameters();
        auto t_params = target_net_->parameters();
        for (size_t i = 0; i < q_params.size(); ++i) {
            t_params[i].copy_(tau_ * q_params[i] + (1.0f - tau_) * t_params[i]);
        }
    }

    int player_id_;
    float eta_, gamma_;
    int batch_size_, learn_every_, min_buffer_size_;
    int target_update_freq_;
    float tau_, epsilon_start_, epsilon_end_;
    int epsilon_decay_duration_;
    int num_dqn_updates_;
    std::vector<int> hidden_sizes_;

    VarMLP q_net_, target_net_, avg_net_;
    torch::optim::Adam q_optimizer_;
    torch::optim::Adam avg_optimizer_;

    std::unique_ptr<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>> fast_q_;
    std::unique_ptr<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>> fast_avg_;

    CircularReplayBuffer<INFO_STATE_SIZE, NUM_ACTIONS> dqn_buffer_;
    ReservoirBuffer<INFO_STATE_SIZE, NUM_ACTIONS> reservoir_buffer_;

    int game_steps_, br_steps_, dqn_steps_;
    Mode mode_;
    std::atomic<float> last_br_loss_{0.0f};
    std::atomic<float> last_avg_loss_{0.0f};

    torch::Tensor dqn_s_, dqn_a_, dqn_r_, dqn_ns_, dqn_d_, dqn_lm_;
    torch::Tensor avg_s_, avg_ap_, avg_lm_;

    // Protects buffers (add/sample). Counters and tensors are only touched by
    // the dedicated gradient thread, so they don't need separate protection.
    mutable std::mutex buffer_mtx_;
};
