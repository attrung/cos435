#pragma once

#include <random>
#include <cmath>
#include <algorithm>
#include <torch/torch.h>
#include "networks.h"
#include "fast_mlp.h"
#include "replay_buffer.h"
#include "leduc_poker.h"

// Risk distortion types
enum class RiskDistortion { NONE, CVAR, SEEKING };

class NFSPIQNAgent {
public:
    NFSPIQNAgent(int player_id, float eta, float dqn_lr, float avg_lr,
                 int hidden_size, int batch_size, int dqn_buffer_size,
                 int reservoir_buffer_size, int learn_every, int min_buffer_size,
                 int target_update_freq, float tau, float epsilon_start,
                 float epsilon_end, int epsilon_decay_duration, float gamma,
                 int iqn_num_quantiles, int iqn_num_quantiles_eval,
                 int iqn_embedding_dim, float iqn_kappa,
                 RiskDistortion distortion, float distortion_param1, float distortion_param2,
                 float variance_penalty)
        : player_id_(player_id), eta_(eta), gamma_(gamma),
          batch_size_(batch_size), learn_every_(learn_every),
          min_buffer_size_(min_buffer_size),
          target_update_freq_(target_update_freq), tau_(tau),
          epsilon_start_(epsilon_start), epsilon_end_(epsilon_end),
          epsilon_decay_duration_(epsilon_decay_duration),
          N_(iqn_num_quantiles), K_(iqn_num_quantiles_eval),
          kappa_(iqn_kappa),
          distortion_(distortion), dist_p1_(distortion_param1), dist_p2_(distortion_param2),
          variance_penalty_(variance_penalty),
          iqn_net_(INFO_STATE_SIZE, hidden_size, NUM_ACTIONS, iqn_embedding_dim),
          iqn_target_(INFO_STATE_SIZE, hidden_size, NUM_ACTIONS, iqn_embedding_dim),
          avg_net_(INFO_STATE_SIZE, hidden_size, NUM_ACTIONS),
          iqn_optimizer_(iqn_net_->parameters(), torch::optim::SGDOptions(dqn_lr).momentum(0.9)),
          avg_optimizer_(avg_net_->parameters(), torch::optim::SGDOptions(avg_lr)),
          dqn_buffer_(dqn_buffer_size),
          reservoir_buffer_(reservoir_buffer_size),
          game_steps_(0), br_steps_(0), dqn_steps_(0),
          mode_(AVG_POLICY), last_br_loss_(0), last_avg_loss_(0) {
        // Copy IQN weights to target
        torch::NoGradGuard no_grad;
        auto p1 = iqn_net_->parameters();
        auto p2 = iqn_target_->parameters();
        for (size_t i = 0; i < p1.size(); ++i) p2[i].copy_(p1[i]);

        // Pre-allocate training tensors
        dqn_s_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        dqn_a_ = torch::zeros({batch_size}, torch::kInt64);
        dqn_r_ = torch::zeros({batch_size});
        dqn_ns_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        dqn_d_ = torch::zeros({batch_size});
        dqn_lm_ = torch::zeros({batch_size, NUM_ACTIONS});
        avg_s_ = torch::zeros({batch_size, INFO_STATE_SIZE});
        avg_ap_ = torch::zeros({batch_size, NUM_ACTIONS});
        avg_lm_ = torch::zeros({batch_size, NUM_ACTIONS});

        fast_avg_ = std::make_unique<FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS>>();
        sync_fast_weights();
    }

    enum Mode { BEST_RESPONSE, AVG_POLICY };

    void sample_episode_mode(std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        mode_ = (dist(rng) < eta_) ? BEST_RESPONSE : AVG_POLICY;
    }

    Mode mode() const { return mode_; }

    // For worker threads: IQN action selection needs torch (quantile sampling)
    // So workers use epsilon-greedy with fast_q_mean_ (pre-computed mean Q)
    // This is an approximation — workers use a snapshot of mean-Q for greedy action,
    // which is equivalent for risk-neutral. For risk-sensitive, we accept the approximation
    // since it only affects exploration policy, not training.
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
        return iqn_action_with_probs(state, legal_actions, legal_mask, rng);
    }

    void add_transition(const std::array<float, INFO_STATE_SIZE>& state, int action,
                        float reward, const std::array<float, INFO_STATE_SIZE>& next_state,
                        float done, const std::array<float, NUM_ACTIONS>& legal_mask) {
        dqn_buffer_.add(state, action, reward, next_state, done, legal_mask);
    }

    void add_reservoir(const std::array<float, INFO_STATE_SIZE>& state,
                       const std::array<float, NUM_ACTIONS>& probs,
                       const std::array<float, NUM_ACTIONS>& legal_mask,
                       std::mt19937& rng) {
        reservoir_buffer_.add(state, probs, legal_mask, rng);
    }

    bool increment_steps(int n, bool is_br_mode, std::mt19937& rng) {
        int old_steps = game_steps_;
        game_steps_ += n;
        int old_br = br_steps_;
        if (is_br_mode) br_steps_ += n;

        bool did_learn = false;
        int old_learn = old_steps / learn_every_;
        int new_learn = game_steps_ / learn_every_;
        for (int i = 0; i < new_learn - old_learn; ++i) {
            update_iqn(rng);
            update_avg_policy(rng);
            did_learn = true;
        }
        // Extra IQN trigger in BR mode (matches OpenSpiel DQN._iteration)
        if (is_br_mode) {
            int old_dqn = dqn_steps_;
            dqn_steps_ += n;
            int old_dl = old_dqn / learn_every_;
            int new_dl = dqn_steps_ / learn_every_;
            for (int i = 0; i < new_dl - old_dl; ++i) {
                update_iqn(rng);
                did_learn = true;
            }
        }
        if (is_br_mode) {
            int old_target = old_br / target_update_freq_;
            int new_target = br_steps_ / target_update_freq_;
            if (new_target > old_target) {
                soft_update_target();
                did_learn = true;
            }
        }
        if (did_learn) sync_fast_weights();
        return did_learn;
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

    void sync_fast_to(FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS>& ext_q,
                      FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS>& ext_avg) {
        // For IQN workers: we compute mean-Q snapshot on main thread
        // and export as a fake MLP. Workers use this for greedy action selection.
        // This reuses the FastMLP infrastructure even though IQN isn't an MLP.
        compute_mean_q_snapshot(ext_q);
        ext_avg.sync_from(*avg_net_);
    }

    void save_weights(const std::string& path) {
        torch::save(avg_net_, path + "_avg.pt");
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
        // Skip iqn_opt.pt — saved with Adam, can't load into SGD.
        // SGD will start fresh momentum, fine-tuning the Adam-found weights.
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
    float last_br_loss() const { return last_br_loss_; }
    float last_avg_loss() const { return last_avg_loss_; }
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
                return taus * dist_p1_;  // tau * alpha
            case RiskDistortion::SEEKING:
                return dist_p1_ + taus * (dist_p2_ - dist_p1_);  // lower + tau*(upper-lower)
            default:
                return taus;
        }
    }

    // Compute mean Q-values for all states and store as MLP-like weights
    // so workers can use FastMLP::forward_argmax for greedy action selection
    void compute_mean_q_snapshot(FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS>& target) {
        // We can't easily represent IQN as a 2-layer MLP.
        // Instead, we compute Q-values on-the-fly in the main thread's iqn_action_with_probs.
        // For workers, we use a linearized approximation: just copy the state_fc and output_fc
        // weights, treating the IQN as if tau_features ≈ 1 (mean embedding).
        // This gives approximate greedy actions for exploration — acceptable.
        torch::NoGradGuard no_grad;

        // Approximate: state_features * mean_tau_features → output
        // mean_tau_features ≈ relu(cos_embedding(mean_cos_input))
        // For uniform tau in [0,1], mean cos(i*pi*tau) over tau ~ analytical but complex.
        // Simpler: just sample K taus, compute mean embedding, use as a fixed multiplier.
        auto taus = torch::rand({1, K_});
        taus = distort_taus(taus);

        // Get cos embedding: (1, K, embedding_dim) -> mean over K -> (1, hidden)
        int emb_dim = iqn_net_->embedding_dim_;
        auto i_pi = torch::arange(0, emb_dim, torch::kFloat32) * M_PI;
        auto cos_input = taus.unsqueeze(-1) * i_pi.unsqueeze(0).unsqueeze(0);
        auto cos_features = torch::cos(cos_input);
        auto tau_features = torch::relu(iqn_net_->cos_embedding(cos_features)); // (1, K, hidden)
        auto mean_tau = tau_features.mean(1).squeeze(0); // (hidden,)

        // Now effective MLP: output = output_fc(relu(state_fc(x)) * mean_tau)
        // = output_fc(diag(mean_tau) @ relu(state_fc(x)))
        // fc1_eff = diag(mean_tau) @ state_fc  (element-wise scale of rows)
        // fc2_eff = output_fc

        auto state_w = iqn_net_->state_fc->weight.data(); // (hidden, input)
        auto state_b = iqn_net_->state_fc->bias.data();   // (hidden,)
        auto out_w = iqn_net_->output_fc->weight.data();   // (output, hidden)
        auto out_b = iqn_net_->output_fc->bias.data();     // (output,)

        // Scale state_fc by mean_tau (element-wise per hidden unit)
        auto scaled_w = state_w * mean_tau.unsqueeze(1);  // (hidden, input)
        auto scaled_b = state_b * mean_tau;                // (hidden,)

        // Copy to FastMLP
        std::memcpy(target.w1, scaled_w.data_ptr<float>(), 128 * INFO_STATE_SIZE * sizeof(float));
        std::memcpy(target.b1, scaled_b.data_ptr<float>(), 128 * sizeof(float));
        std::memcpy(target.w2, out_w.data_ptr<float>(), NUM_ACTIONS * 128 * sizeof(float));
        std::memcpy(target.b2, out_b.data_ptr<float>(), NUM_ACTIONS * sizeof(float));
    }

    std::pair<int, std::array<float, NUM_ACTIONS>>
    iqn_action_with_probs(const std::array<float, INFO_STATE_SIZE>& state,
                          const std::vector<int>& legal_actions,
                          const std::array<float, NUM_ACTIONS>& legal_mask,
                          std::mt19937& rng) {
        std::array<float, NUM_ACTIONS> probs{};
        float epsilon = get_epsilon();
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

        if (uniform(rng) < epsilon) {
            std::uniform_int_distribution<int> adist(0, (int)legal_actions.size() - 1);
            int action = legal_actions[adist(rng)];
            for (int a : legal_actions) probs[a] = 1.0f / legal_actions.size();
            return {action, probs};
        }

        // Full IQN forward on main thread (only called during main-thread episodes, rare)
        torch::NoGradGuard no_grad;
        auto state_t = torch::from_blob(const_cast<float*>(state.data()),
                                        {1, INFO_STATE_SIZE}).clone();
        auto taus = torch::rand({1, K_});
        taus = distort_taus(taus);
        auto quantile_values = iqn_net_->forward(state_t, taus); // (1, K, actions)

        torch::Tensor q_values;
        if (variance_penalty_ > 0) {
            auto q_mean = quantile_values.mean(1).squeeze(0);
            auto q_var = quantile_values.var(1).squeeze(0);
            q_values = q_mean - variance_penalty_ * q_var;
        } else {
            q_values = quantile_values.mean(1).squeeze(0);
        }

        auto mask_t = torch::from_blob(const_cast<float*>(legal_mask.data()), {NUM_ACTIONS}).clone();
        q_values = q_values + (mask_t - 1.0f) * 1e38f;
        int action = q_values.argmax().item<int>();
        probs[action] = 1.0f;
        return {action, probs};
    }

    void update_iqn(std::mt19937& rng) {
        if (dqn_buffer_.size() < std::max(batch_size_, min_buffer_size_)) return;

        dqn_buffer_.sample(batch_size_, rng,
                          dqn_s_.data_ptr<float>(), dqn_a_.data_ptr<int64_t>(),
                          dqn_r_.data_ptr<float>(), dqn_ns_.data_ptr<float>(),
                          dqn_d_.data_ptr<float>(), dqn_lm_.data_ptr<float>());

        auto taus = torch::rand({batch_size_, N_});
        auto current_quantiles = iqn_net_->forward(dqn_s_, taus); // (B, N, A)
        current_quantiles = current_quantiles.gather(
            2, dqn_a_.unsqueeze(1).unsqueeze(2).expand({-1, N_, -1})).squeeze(2); // (B, N)

        torch::Tensor targets;
        {
            torch::NoGradGuard no_grad;
            auto taus_target = torch::rand({batch_size_, N_});
            auto next_quantiles = iqn_target_->forward(dqn_ns_, taus_target); // (B, N, A)
            auto next_q_avg = next_quantiles.mean(1); // (B, A)
            next_q_avg = next_q_avg + (dqn_lm_ - 1.0f) * 1e38f;
            auto best_actions = next_q_avg.argmax(1); // (B,)
            next_quantiles = next_quantiles.gather(
                2, best_actions.unsqueeze(1).unsqueeze(2).expand({-1, N_, -1})).squeeze(2);
            targets = dqn_r_.unsqueeze(1) + gamma_ * (1.0f - dqn_d_.unsqueeze(1)) * next_quantiles;
        }

        // Quantile Huber loss
        auto td_errors = targets.unsqueeze(1) - current_quantiles.unsqueeze(2); // (B, N, N)
        auto huber_loss = torch::where(
            td_errors.abs() <= kappa_,
            0.5f * td_errors.pow(2),
            kappa_ * (td_errors.abs() - 0.5f * kappa_));
        auto tau_weights = (taus.unsqueeze(2) - (td_errors.detach() < 0).to(torch::kFloat32)).abs();
        auto loss = (tau_weights * huber_loss).mean(2).mean(1).mean();

        iqn_optimizer_.zero_grad();
        loss.backward();
        iqn_optimizer_.step();

        last_br_loss_ = loss.item<float>();
    }

    void update_avg_policy(std::mt19937& rng) {
        if (reservoir_buffer_.size() < std::max(batch_size_, min_buffer_size_)) return;

        reservoir_buffer_.sample(batch_size_, rng,
                                avg_s_.data_ptr<float>(), avg_ap_.data_ptr<float>(),
                                avg_lm_.data_ptr<float>());

        auto logits = avg_net_->forward(avg_s_);
        logits = logits + (avg_lm_ - 1.0f) * 1e38f;
        auto loss = torch::nn::functional::cross_entropy(logits, avg_ap_);

        avg_optimizer_.zero_grad();
        loss.backward();
        avg_optimizer_.step();

        last_avg_loss_ = loss.item<float>();
    }

    void soft_update_target() {
        torch::NoGradGuard no_grad;
        auto p1 = iqn_net_->parameters();
        auto p2 = iqn_target_->parameters();
        for (size_t i = 0; i < p1.size(); ++i)
            p2[i].copy_(tau_ * p1[i] + (1.0f - tau_) * p2[i]);
    }

    void sync_fast_weights() {
        fast_avg_->sync_from(*avg_net_);
    }

    int player_id_;
    float eta_, gamma_;
    int batch_size_, learn_every_, min_buffer_size_;
    int target_update_freq_;
    float tau_, epsilon_start_, epsilon_end_;
    int epsilon_decay_duration_;
    int N_, K_;
    float kappa_;
    RiskDistortion distortion_;
    float dist_p1_, dist_p2_;
    float variance_penalty_;

    IQNNetwork iqn_net_, iqn_target_;
    MLP avg_net_;
    torch::optim::SGD iqn_optimizer_;
    torch::optim::SGD avg_optimizer_;

    std::unique_ptr<FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS>> fast_avg_;

    CircularReplayBuffer<INFO_STATE_SIZE, NUM_ACTIONS> dqn_buffer_;
    ReservoirBuffer<INFO_STATE_SIZE, NUM_ACTIONS> reservoir_buffer_;

    int game_steps_, br_steps_, dqn_steps_;
    Mode mode_;
    float last_br_loss_, last_avg_loss_;

    torch::Tensor dqn_s_, dqn_a_, dqn_r_, dqn_ns_, dqn_d_, dqn_lm_;
    torch::Tensor avg_s_, avg_ap_, avg_lm_;
};
