#pragma once

// NFSP IQN Agent for Hold'em with variable-size MLP avg policy.
// IQN network uses its own architecture (state_fc + cos_embedding + output_fc).
// Avg policy uses VarMLP matching the baseline's [1024, 512, 1024, 512].

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

// Risk distortion types (only define if not already defined by baseline agent)
#ifndef RISK_DISTORTION_DEFINED
#define RISK_DISTORTION_DEFINED
enum class RiskDistortion { NONE, CVAR, SEEKING };
#endif

class NFSPIQNAgent {
public:
    NFSPIQNAgent(int player_id, float eta, float dqn_lr, float avg_lr,
                 std::vector<int> hidden_sizes, int batch_size, int dqn_buffer_size,
                 int reservoir_buffer_size, int learn_every, int min_buffer_size,
                 int target_update_freq, float tau, float epsilon_start,
                 float epsilon_end, int epsilon_decay_duration, float gamma,
                 int iqn_num_quantiles, int iqn_num_quantiles_eval,
                 int iqn_embedding_dim, float iqn_kappa,
                 RiskDistortion distortion, float distortion_param1, float distortion_param2,
                 float variance_penalty, int num_dqn_updates = 2)
        : player_id_(player_id), eta_(eta), gamma_(gamma),
          batch_size_(batch_size), learn_every_(learn_every),
          min_buffer_size_(min_buffer_size),
          target_update_freq_(target_update_freq), tau_(tau),
          epsilon_start_(epsilon_start), epsilon_end_(epsilon_end),
          epsilon_decay_duration_(epsilon_decay_duration),
          num_dqn_updates_(num_dqn_updates),
          N_(iqn_num_quantiles), K_(iqn_num_quantiles_eval),
          kappa_(iqn_kappa),
          distortion_(distortion), dist_p1_(distortion_param1), dist_p2_(distortion_param2),
          variance_penalty_(variance_penalty),
          hidden_sizes_(hidden_sizes),
          // IQN uses its own network architecture (not VarMLP)
          // hidden_size for IQN = first hidden layer size
          iqn_net_(INFO_STATE_SIZE, hidden_sizes[0], NUM_ACTIONS, iqn_embedding_dim),
          iqn_target_(INFO_STATE_SIZE, hidden_sizes[0], NUM_ACTIONS, iqn_embedding_dim),
          // Avg policy uses the same VarMLP architecture as baseline
          avg_net_(INFO_STATE_SIZE, hidden_sizes, NUM_ACTIONS),
          iqn_optimizer_(iqn_net_->parameters(), torch::optim::AdamOptions(dqn_lr)),
          avg_optimizer_(avg_net_->parameters(), torch::optim::AdamOptions(avg_lr)),
          dqn_buffer_(dqn_buffer_size),
          reservoir_buffer_(reservoir_buffer_size),
          game_steps_(0), br_steps_(0), dqn_steps_(0),
          mode_(AVG_POLICY) {
        torch::NoGradGuard no_grad;
        auto p1 = iqn_net_->parameters();
        auto p2 = iqn_target_->parameters();
        for (size_t i = 0; i < p1.size(); ++i) p2[i].copy_(p1[i]);

        dqn_s_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        dqn_a_ = torch::zeros({batch_size}, torch::kInt64);
        dqn_r_ = torch::zeros({batch_size});
        dqn_ns_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        dqn_d_ = torch::zeros({batch_size});
        dqn_lm_ = torch::zeros({batch_size, NUM_ACTIONS});
        avg_s_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        avg_ap_ = torch::zeros({batch_size, NUM_ACTIONS});
        avg_lm_ = torch::zeros({batch_size, NUM_ACTIONS});

        fast_avg_ = std::make_unique<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>>();
        fast_avg_->init(hidden_sizes);
        // For IQN action selection in workers, we precompute mean Q-values into a VarFastMLP
        fast_q_ = std::make_unique<VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>>();
        // IQN Q approximation only uses 1 effective hidden layer (state_fc * mean_tau -> output)
        fast_q_->init({hidden_sizes[0]});
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
        // BR mode: use fast_q_ (precomputed mean Q-values)
        std::array<float, NUM_ACTIONS> probs{};
        float epsilon = get_epsilon();
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        if (uniform(rng) < epsilon) {
            std::uniform_int_distribution<int> adist(0, (int)legal_actions.size() - 1);
            int action = legal_actions[adist(rng)];
            for (int a : legal_actions) probs[a] = 1.0f / legal_actions.size();
            return {action, probs};
        }
        int action = fast_q_->forward_argmax(state.data(), legal_mask.data());
        probs[action] = 1.0f;
        return {action, probs};
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

    // Thread-safe version for async gradient thread.
    bool catch_up_steps(int new_game_steps, int new_br_steps, std::mt19937& rng) {
        int old_steps = game_steps_;
        game_steps_ += new_game_steps;
        int old_br = br_steps_;
        br_steps_ += new_br_steps;

        bool did_learn = false;

        int g_crossings = (game_steps_ / learn_every_) - (old_steps / learn_every_);
        for (int i = 0; i < g_crossings; ++i) {
            for (int u = 0; u < num_dqn_updates_; ++u) update_iqn(rng);
            for (int u = 0; u < num_dqn_updates_; ++u) update_avg_policy(rng);
            did_learn = true;
        }

        int old_dqn = dqn_steps_;
        dqn_steps_ += new_br_steps;
        int b_crossings = (dqn_steps_ / learn_every_) - (old_dqn / learn_every_);
        for (int i = 0; i < b_crossings; ++i) {
            for (int u = 0; u < num_dqn_updates_; ++u) update_iqn(rng);
            did_learn = true;
        }

        int t_crossings = (br_steps_ / target_update_freq_) - (old_br / target_update_freq_);
        if (t_crossings > 0) {
            soft_update_target();
            did_learn = true;
        }

        return did_learn;
    }

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
        compute_mean_q_snapshot(ext_q);
        ext_avg.sync_from(*avg_net_);
    }

    const std::vector<int>& hidden_sizes() const { return hidden_sizes_; }

    void save_weights(const std::string& path) {
        torch::save(avg_net_, path + "_avg.pt");
        torch::save(iqn_net_, path + "_iqn_net.pt");
    }

    // Load frozen target: avg + iqn_net (BR head). play_br=true forces the agent
    // to always play its IQN BR head (eta=1). play_br=false forces AVG mode.
    void load_frozen_weights(const std::string& avg_path, const std::string& iqn_path, bool play_br) {
        torch::load(avg_net_, avg_path);
        torch::load(iqn_net_, iqn_path);
        torch::load(iqn_target_, iqn_path);
        eta_ = play_br ? 1.0f : 0.0f;
        sync_fast_weights();
    }

    // Load frozen target: avg only. Forces AVG mode (can't play BR without iqn_net).
    void load_frozen_avg_only(const std::string& avg_path) {
        torch::load(avg_net_, avg_path);
        eta_ = 0.0f;
        sync_fast_weights();
    }

    void save_checkpoint(const std::string& dir) {
        std::system(("mkdir -p " + dir).c_str());
        torch::save(iqn_net_, dir + "/iqn_net.pt");
        torch::save(iqn_target_, dir + "/iqn_target.pt");
        torch::save(avg_net_, dir + "/avg_net.pt");
        torch::save(iqn_optimizer_, dir + "/iqn_opt.pt");
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
        torch::load(iqn_net_, dir + "/iqn_net.pt");
        torch::load(iqn_target_, dir + "/iqn_target.pt");
        torch::load(avg_net_, dir + "/avg_net.pt");
        try { torch::load(iqn_optimizer_, dir + "/iqn_opt.pt"); }
        catch (...) { std::cout << "  (skipped iqn_opt.pt — optimizer reset)" << std::endl; }
        try { torch::load(avg_optimizer_, dir + "/avg_opt.pt"); } catch (...) {}
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

    int br_steps_val() const { return br_steps_; }
    float last_br_loss() const { return last_br_loss_.load(std::memory_order_relaxed); }
    float last_avg_loss() const { return last_avg_loss_.load(std::memory_order_relaxed); }
    int game_steps() const { return game_steps_; }

private:
    float get_epsilon() const {
        int t = std::min(br_steps_, epsilon_decay_duration_);
        return epsilon_end_ + (epsilon_start_ - epsilon_end_) *
               std::exp(-(float)t / epsilon_decay_duration_);
    }

    torch::Tensor distort_taus(torch::Tensor taus) {
        switch (distortion_) {
            case RiskDistortion::CVAR:
                return taus * dist_p1_;
            case RiskDistortion::SEEKING:
                return dist_p1_ + taus * (dist_p2_ - dist_p1_);
            default:
                return taus;
        }
    }

    void compute_mean_q_snapshot(VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS>& target) {
        // Approximate IQN's mean Q-values as a single-layer network for fast worker inference.
        // IQN computes: Q(s) = output_fc(relu(state_fc(s)) * relu(cos_emb(tau)))
        // For mean Q: average over K tau samples → mean_tau_features
        // Then: Q_mean(s) ≈ output_fc(relu(state_fc(s)) * mean_tau)
        // This is a 1-layer network: W_eff = state_W * diag(mean_tau), b_eff = state_b * mean_tau
        torch::NoGradGuard no_grad;

        auto taus = torch::rand({1, K_});
        taus = distort_taus(taus);

        int emb_dim = iqn_net_->embedding_dim_;
        int hid = hidden_sizes_[0];
        auto i_pi = torch::arange(0, emb_dim, torch::kFloat32) * M_PI;
        auto cos_input = taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0);
        auto cos_features = torch::cos(cos_input);
        auto tau_features = torch::relu(iqn_net_->cos_embedding(cos_features));
        auto mean_tau = tau_features.mean(1).squeeze(0);  // [hid]

        auto state_w = iqn_net_->state_fc->weight.data();  // [hid, IN]
        auto state_b = iqn_net_->state_fc->bias.data();    // [hid]
        auto out_w = iqn_net_->output_fc->weight.data();   // [OUT, hid]
        auto out_b = iqn_net_->output_fc->bias.data();     // [OUT]

        auto scaled_w = state_w * mean_tau.unsqueeze(1);   // [hid, IN]
        auto scaled_b = state_b * mean_tau;                 // [hid]

        // Write into the VarFastMLP's single hidden layer
        // target was init'd with {hidden_sizes_[0]} = 1 layer of size hid
        auto& layer = target.hidden_layers[0];
        std::memcpy(layer.w.data(), scaled_w.data_ptr<float>(), hid * INFO_STATE_SIZE * sizeof(float));
        std::memcpy(layer.b.data(), scaled_b.data_ptr<float>(), hid * sizeof(float));
        std::memcpy(target.output_layer.w.data(), out_w.data_ptr<float>(), NUM_ACTIONS * hid * sizeof(float));
        std::memcpy(target.output_layer.b.data(), out_b.data_ptr<float>(), NUM_ACTIONS * sizeof(float));
    }

    void update_iqn(std::mt19937& rng) {
        {
            std::lock_guard<std::mutex> lock(buffer_mtx_);
            if (dqn_buffer_.size() < std::max(batch_size_, min_buffer_size_)) return;
            dqn_buffer_.sample(batch_size_, rng,
                              dqn_s_.data_ptr<float>(), dqn_a_.data_ptr<int64_t>(),
                              dqn_r_.data_ptr<float>(), dqn_ns_.data_ptr<float>(),
                              dqn_d_.data_ptr<float>(), dqn_lm_.data_ptr<float>());
        }

        auto taus = torch::rand({batch_size_, N_});
        auto current_quantiles = iqn_net_->forward(dqn_s_, taus);
        current_quantiles = current_quantiles.gather(
            2, dqn_a_.unsqueeze(1).unsqueeze(2).expand({-1, N_, -1})).squeeze(2);

        torch::Tensor targets;
        {
            torch::NoGradGuard no_grad;
            auto taus_target = torch::rand({batch_size_, N_});
            auto next_quantiles = iqn_target_->forward(dqn_ns_, taus_target);
            auto next_q_avg = next_quantiles.mean(1);
            // Proper masking for illegal actions
            next_q_avg = next_q_avg.masked_fill(dqn_lm_ < 0.5f, -std::numeric_limits<float>::infinity());
            auto best_actions = next_q_avg.argmax(1);
            next_quantiles = next_quantiles.gather(
                2, best_actions.unsqueeze(1).unsqueeze(2).expand({-1, N_, -1})).squeeze(2);
            targets = dqn_r_.unsqueeze(1) + gamma_ * (1.0f - dqn_d_.unsqueeze(1)) * next_quantiles;
        }

        auto td_errors = targets.unsqueeze(1) - current_quantiles.unsqueeze(2);
        auto huber_loss = torch::where(
            td_errors.abs() <= kappa_,
            0.5f * td_errors.pow(2),
            kappa_ * (td_errors.abs() - 0.5f * kappa_));
        auto tau_weights = (taus.unsqueeze(2) - (td_errors.detach() < 0).to(torch::kFloat32)).abs();
        auto loss = (tau_weights * huber_loss).mean(2).mean(1).mean();

        iqn_optimizer_.zero_grad();
        loss.backward();
        iqn_optimizer_.step();

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

        auto logits = avg_net_->forward(avg_s_);
        logits = logits + (avg_lm_ - 1.0f) * 1e38f;
        auto loss = torch::nn::functional::cross_entropy(logits, avg_ap_);

        avg_optimizer_.zero_grad();
        loss.backward();
        avg_optimizer_.step();

        last_avg_loss_.store(loss.item<float>(), std::memory_order_relaxed);
    }

    void soft_update_target() {
        torch::NoGradGuard no_grad;
        auto p1 = iqn_net_->parameters();
        auto p2 = iqn_target_->parameters();
        for (size_t i = 0; i < p1.size(); ++i)
            p2[i].copy_(tau_ * p1[i] + (1.0f - tau_) * p2[i]);
    }

    void sync_fast_weights() {
        compute_mean_q_snapshot(*fast_q_);
        fast_avg_->sync_from(*avg_net_);
    }

    int player_id_;
    float eta_, gamma_;
    int batch_size_, learn_every_, min_buffer_size_;
    int target_update_freq_;
    float tau_, epsilon_start_, epsilon_end_;
    int epsilon_decay_duration_;
    int num_dqn_updates_;
    int N_, K_;
    float kappa_;
    RiskDistortion distortion_;
    float dist_p1_, dist_p2_;
    float variance_penalty_;
    std::vector<int> hidden_sizes_;

    IQNNetwork iqn_net_, iqn_target_;
    VarMLP avg_net_;
    torch::optim::Adam iqn_optimizer_;
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

    mutable std::mutex buffer_mtx_;
};
