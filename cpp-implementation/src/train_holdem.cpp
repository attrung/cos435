#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <random>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <functional>
#include <sstream>
#include <csignal>

#include "holdem.h"
#include "nfsp_agent_holdem.h"

// Normalize rewards by big blind (100 chips). Paper claims raw rewards + SGD 0.1
// but this diverges empirically. /100 normalization keeps reward scale comparable
// to Leduc while using the paper's lr=0.1.
constexpr float REWARD_SCALE = 1.0f / 100.0f;
#include "nfsp_iqn_agent_holdem.h"

// ── Episode data (fixed-capacity, zero-alloc after init) ──────────────────

// Max transitions per player per episode in Hold'em: ~10 steps total,
// so ~5 per player. Use 12 for safety (4 rounds x 3 raises max = 12 actions).
constexpr int MAX_TRANS_PER_PLAYER = 12;
constexpr int MAX_RESERVOIR_PER_PLAYER = 12;

struct Trans {
    std::array<float, INFO_STATE_SIZE> state;
    int action;
    float reward;
    std::array<float, INFO_STATE_SIZE> next_state;
    float done;
    std::array<float, NUM_ACTIONS> next_legal;
};

struct ReservoirSample {
    std::array<float, INFO_STATE_SIZE> state;
    std::array<float, NUM_ACTIONS> probs;
    std::array<float, NUM_ACTIONS> mask;
};

struct EpisodeResult {
    std::array<Trans, MAX_TRANS_PER_PLAYER> transitions[2];
    int num_transitions[2] = {0, 0};
    std::array<ReservoirSample, MAX_RESERVOIR_PER_PLAYER> reservoir[2];
    int num_reservoir[2] = {0, 0};
    std::array<double, 2> returns;
    std::array<int, 2> modes;  // 0=BR, 1=AVG

    void clear() {
        num_transitions[0] = num_transitions[1] = 0;
        num_reservoir[0] = num_reservoir[1] = 0;
    }
};

constexpr int MAX_BATCH = 16;
struct Batch {
    EpisodeResult episodes[MAX_BATCH];
    int size = 0;
};

// ── Bounded queue ─────────────────────────────────────────────────────────

template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(int cap) : cap_(cap) {}
    void push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_f_.wait(lk, [&]{ return q_.size() < (size_t)cap_ || done_; });
        if (done_) return;
        q_.push(std::move(item));
        cv_e_.notify_one();
    }
    bool pop(T& item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_e_.wait(lk, [&]{ return !q_.empty() || done_; });
        if (q_.empty()) return false;
        item = std::move(q_.front()); q_.pop();
        cv_f_.notify_one();
        return true;
    }
    void shutdown() { std::lock_guard<std::mutex> lk(mu_); done_=true; cv_e_.notify_all(); cv_f_.notify_all(); }
    void drain() { std::lock_guard<std::mutex> lk(mu_); while(!q_.empty()) q_.pop(); cv_f_.notify_all(); }
private:
    int cap_; bool done_=false; std::queue<T> q_; std::mutex mu_;
    std::condition_variable cv_e_, cv_f_;
};

// ── Worker context (shared, lock-free reads) ──────────────────────────────

// Worker-visible FastMLP pair, protected by mutex.
struct FastNetsView {
    VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS> fast_q;
    VarFastMLP<INFO_STATE_SIZE, NUM_ACTIONS> fast_avg;
    std::mutex mtx;
};

struct WorkerContext {
    FastNetsView* views[2];  // one per player, protected by mutex
    float eta;
    float epsilon_start, epsilon_end;
    int epsilon_decay_duration;
    std::atomic<int> br_steps[2];
};

void play_episode_worker(HoldemState& state, WorkerContext& ctx,
                         EpisodeResult& result, std::mt19937& rng) {
    result.clear();
    state.reset();

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int p = 0; p < 2; ++p)
        result.modes[p] = (uniform(rng) < ctx.eta) ? 0 : 1;

    struct PT { std::array<float,INFO_STATE_SIZE> s; int a; std::array<float,NUM_ACTIONS> m; };
    PT pending[2][MAX_TRANS_PER_PLAYER];
    int npending[2] = {0, 0};

    while (!state.is_terminal()) {
        if (state.is_chance_node()) {
            state.apply_action(state.sample_chance_action(rng));
        } else {
            int p = state.current_player();
            auto info = state.info_state_tensor(p);
            auto legal = state.legal_actions();
            auto mask = state.legal_actions_mask();
            int action;
            std::array<float, NUM_ACTIONS> probs{};

            if (result.modes[p] == 0) { // BR
                int t = std::min(ctx.br_steps[p].load(std::memory_order_relaxed),
                                 ctx.epsilon_decay_duration);
                float eps = ctx.epsilon_end + (ctx.epsilon_start - ctx.epsilon_end) *
                            std::exp(-(float)t / ctx.epsilon_decay_duration);
                if (uniform(rng) < eps) {
                    std::uniform_int_distribution<int> d(0, (int)legal.size()-1);
                    action = legal[d(rng)];
                    for (int a : legal) probs[a] = 1.0f / legal.size();
                } else {
                    std::lock_guard<std::mutex> lk(ctx.views[p]->mtx);
                    action = ctx.views[p]->fast_q.forward_argmax(info.data(), mask.data());
                    probs[action] = 1.0f;
                }
                if (result.num_reservoir[p] < MAX_RESERVOIR_PER_PLAYER) {
                    int ri = result.num_reservoir[p]++;
                    result.reservoir[p][ri] = {info, probs, mask};
                }
            } else { // AVG
                std::lock_guard<std::mutex> lk(ctx.views[p]->mtx);
                action = ctx.views[p]->fast_avg.forward_sample(info.data(), mask.data(), uniform(rng));
            }
            if (npending[p] < MAX_TRANS_PER_PLAYER) {
                int pi = npending[p]++;
                pending[p][pi] = {info, action, mask};
            }
            state.apply_action(action);
        }
    }
    result.returns = state.returns();

    static const std::array<float,INFO_STATE_SIZE> zero_s{};
    static const std::array<float,NUM_ACTIONS> ones_m = {1,1,1};
    for (int p = 0; p < 2; ++p) {
        int n = npending[p];
        result.num_transitions[p] = n;
        for (int i = 0; i < n; ++i) {
            bool last = (i == n - 1);
            result.transitions[p][i] = {
                pending[p][i].s, pending[p][i].a,
                last ? (float)result.returns[p] * REWARD_SCALE : 0.0f,
                last ? zero_s : pending[p][i+1].s,
                last ? 1.0f : 0.0f,
                last ? ones_m : pending[p][i+1].m
            };
        }
    }
}

void worker_thread(int id, WorkerContext* ctx, BoundedQueue<Batch>* q,
                   std::atomic<bool>* stop, int bsz) {
    std::mt19937 rng(id * 7919 + 42);
    HoldemState state;
    Batch batch;
    while (!stop->load(std::memory_order_relaxed)) {
        batch.size = bsz;
        for (int i = 0; i < bsz; ++i)
            play_episode_worker(state, *ctx, batch.episodes[i], rng);
        q->push(batch);
    }
}

// ── JSONL Logger ──────────────────────────────────────────────────────────

class JSONLLogger {
public:
    JSONLLogger(const std::string& log_dir, const std::string& experiment_name)
        : experiment_(experiment_name) {
        std::system(("mkdir -p " + log_dir).c_str());
        path_ = log_dir + "/" + experiment_name + ".jsonl";
        std::ofstream(path_, std::ios::trunc).close();
    }

    void log_stats(int episode, double avg_reward, double h2h_winrate,
                   double br_loss, double avg_pol_loss, double eps_per_sec,
                   double elapsed_min) {
        auto ts = timestamp();
        std::ofstream f(path_, std::ios::app);
        f << "{\"episode\": " << episode
          << ", \"avg_reward\": " << avg_reward
          << ", \"h2h_winrate\": " << h2h_winrate
          << ", \"br_loss\": " << br_loss
          << ", \"avg_pol_loss\": " << avg_pol_loss
          << ", \"eps_per_sec\": " << eps_per_sec
          << ", \"elapsed_min\": " << elapsed_min
          << ", \"timestamp\": \"" << ts << "\""
          << ", \"experiment\": \"" << experiment_ << "\"}\n";
    }

    void log_h2h(int episode, double winrate) {
        auto ts = timestamp();
        std::ofstream f(path_, std::ios::app);
        f << "{\"episode\": " << episode
          << ", \"h2h_eval_winrate\": " << winrate
          << ", \"timestamp\": \"" << ts << "\""
          << ", \"experiment\": \"" << experiment_ << "\"}\n";
    }

private:
    std::string timestamp() {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
        return buf;
    }
    std::string path_, experiment_;
};

// ── Head-to-head evaluation ───────────────────────────────────────────────

double compute_h2h(const std::string& weights_dir, int episode,
                   const std::string& eval_script,
                   const std::string& log_dir, const std::string& name) {
    std::string result_file = weights_dir + "/h2h_result.txt";
    std::string cmd = "python3 " + eval_script +
                      " --weights-dir " + weights_dir +
                      " --episode " + std::to_string(episode) +
                      " --result-file " + result_file;
    std::system(cmd.c_str());

    double winrate = -1.0;
    std::ifstream f(result_file);
    if (f.is_open()) f >> winrate;
    return winrate;
}

// ── Agent interface (type-erased) ─────────────────────────────────────────

struct AgentOps {
    std::function<void(std::mt19937&)> sample_mode;
    std::function<void(const std::array<float,INFO_STATE_SIZE>&, int, float,
                       const std::array<float,INFO_STATE_SIZE>&, float,
                       const std::array<float,NUM_ACTIONS>&)> add_transition;
    std::function<void(const std::array<float,INFO_STATE_SIZE>&,
                       const std::array<float,NUM_ACTIONS>&,
                       const std::array<float,NUM_ACTIONS>&, std::mt19937&)> add_reservoir;
    std::function<bool(int, bool, std::mt19937&)> increment_steps;
    // Async version: called by dedicated gradient thread.
    std::function<bool(int, int, std::mt19937&)> catch_up_steps;
    std::function<void(VarFastMLP<INFO_STATE_SIZE,NUM_ACTIONS>&,
                       VarFastMLP<INFO_STATE_SIZE,NUM_ACTIONS>&)> sync_fast_to;
    std::function<int()> br_steps_val;
    std::function<float()> last_br_loss;
    std::function<float()> last_avg_loss;
    std::function<void(const std::string&)> save_weights;
    std::function<void(const std::string&)> save_checkpoint;
    std::function<bool(const std::string&)> load_checkpoint;
};

// ── Per-agent gradient thread ─────────────────────────────────────────────
// Receives (game_steps, br_steps) deltas from main thread and does all
// gradient updates + FastMLP sync on its own thread. Workers' FastMLP reads
// are serialized against syncs via FastNetsView::mtx.
class AgentLearner {
public:
    AgentLearner(int agent_id, AgentOps* ops, FastNetsView* view,
                 std::atomic<int>* br_steps_atomic, int seed)
        : agent_id_(agent_id), ops_(ops), view_(view),
          br_atomic_(br_steps_atomic), rng_(seed + agent_id * 1009),
          pending_g_(0), pending_b_(0), stop_(false) {
        thread_ = std::thread(&AgentLearner::run, this);
    }

    ~AgentLearner() { shutdown(); }

    // Main thread signals: "added new_g game steps (of which new_b were in BR mode)"
    void signal(int new_g, int new_b) {
        {
            std::lock_guard<std::mutex> lock(sig_mtx_);
            pending_g_ += new_g;
            pending_b_ += new_b;
        }
        sig_cv_.notify_one();
    }

    void shutdown() {
        if (stop_.exchange(true)) return;
        sig_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        while (!stop_.load()) {
            int g, b;
            {
                std::unique_lock<std::mutex> lock(sig_mtx_);
                sig_cv_.wait(lock, [&]{ return pending_g_ > 0 || stop_.load(); });
                if (stop_.load() && pending_g_ == 0) break;
                g = pending_g_;  pending_g_ = 0;
                b = pending_b_;  pending_b_ = 0;
            }

            bool learned = ops_->catch_up_steps(g, b, rng_);

            if (learned) {
                std::lock_guard<std::mutex> lock(view_->mtx);
                ops_->sync_fast_to(view_->fast_q, view_->fast_avg);
                br_atomic_->store(ops_->br_steps_val(), std::memory_order_relaxed);
            }
        }
    }

    int agent_id_;
    AgentOps* ops_;
    FastNetsView* view_;
    std::atomic<int>* br_atomic_;
    std::mt19937 rng_;

    std::mutex sig_mtx_;
    std::condition_variable sig_cv_;
    int pending_g_;
    int pending_b_;

    std::atomic<bool> stop_;
    std::thread thread_;
};

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Hold'em defaults — matching Heinrich & Silver 2016 Section 4.3 exactly
    int num_episodes = 5'000'000;  // paper converges by ~3.5M iterations (~3.5M episodes)
    std::string agent_type = "nfsp";
    std::string name = "holdem_baseline";
    float eta = 0.1f;
    float dqn_lr = 0.003f;   // Adam, bumped 3x for smaller [256,128,256,128] net (prior 0.001 with [512,256,512,256] underlearned)
    float avg_lr = 0.0003f;  // Adam, 10x lower than DQN per paper ratio
    int batch_size = 256;     // paper: "mini-batch size 256"
    // Paper: [1024, 512, 1024, 512] — set via hidden_sizes vector below
    int dqn_buf = 600000, res_buf = 30000000;  // paper-matched (125GB RAM allows full 30M reservoir).
    int learn_every = 256, min_buf = 1000;  // paper-matched (reverted from 128 for stability)
    int num_updates_per_learn = 2;  // paper: "2 stochastic gradient updates per network"
    int target_freq = 1000;   // paper: "target network refitted every 1000 updates"
    // NOTE: target_freq here means 1000 learning-update triggers, not game steps.
    // With learn_every=256, each trigger does 2 updates.
    // We track in game steps: 1000 * 256 = 256000 game steps between target updates.
    // But with dual counter, target is triggered by br_steps / target_update_freq.
    // So target_update_freq = 1000 * learn_every = 256000 br game steps.
    // Paper: "target network refitted every 1000 updates"
    // With learn_every=256 and 2 updates per trigger: 1 update per 128 game steps on avg
    // 1000 updates = 128,000 game steps
    int target_update_freq_steps = 128000;
    float tau = 1.0f;         // paper: hard copy (refit), not soft update
    float eps_start = 0.08f, eps_end = 0.001f;  // paper says decay to 0; use 0.001 floor
    int eps_duration = 20000000;  // slow decay (paper emphasizes slower decay than Leduc)
    float gamma = 1.0f;
    int eval_freq = 100000, log_freq = 50000, checkpoint_freq = 500000;
    int seed = 42, num_workers = 1, worker_batch = 4;
    std::string checkpoint_dir = "checkpoints";
    // IQN params
    int iqn_N = 8, iqn_K = 32, iqn_emb = 64;
    float iqn_kappa = 1.0f;
    std::string risk = "none";
    float risk_p1 = 0.25f, risk_p2 = 1.0f;
    float var_penalty = 0.0f;
    std::string eval_script = "eval/eval_holdem_h2h.py";
    std::string hidden_str;  // comma-sep arch override, e.g. "128,64,128,64"

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return argv[++i]; };
        if (a == "--episodes") num_episodes = std::stoi(next());
        else if (a == "--agent") agent_type = next();
        else if (a == "--name") name = next();
        else if (a == "--dqn-lr") dqn_lr = std::stof(next());
        else if (a == "--avg-lr") avg_lr = std::stof(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--eval-freq") eval_freq = std::stoi(next());
        else if (a == "--log-freq") log_freq = std::stoi(next());
        else if (a == "--workers") num_workers = std::stoi(next());
        else if (a == "--risk") risk = next();
        else if (a == "--risk-p1") risk_p1 = std::stof(next());
        else if (a == "--risk-p2") risk_p2 = std::stof(next());
        else if (a == "--var-penalty") var_penalty = std::stof(next());
        else if (a == "--checkpoint-freq") checkpoint_freq = std::stoi(next());
        else if (a == "--checkpoint-dir") checkpoint_dir = next();
        else if (a == "--worker-batch") worker_batch = std::stoi(next());
        else if (a == "--iqn-n") iqn_N = std::stoi(next());
        else if (a == "--target-freq") target_update_freq_steps = std::stoi(next());
        else if (a == "--batch-size") batch_size = std::stoi(next());
        else if (a == "--learn-every") learn_every = std::stoi(next());
        else if (a == "--eps-duration") eps_duration = std::stoi(next());
        else if (a == "--eval-script") eval_script = next();
        else if (a == "--hidden") hidden_str = next();
    }

    torch::set_num_threads(1);
    std::mt19937 rng(seed);

    // Paper: [1024, 512, 1024, 512] — too slow for our compute.
    // Default: [256,128,256,128]. Override with --hidden "a,b,c,d".
    std::vector<int> hidden_sizes = {256, 128, 256, 128};
    if (!hidden_str.empty()) {
        hidden_sizes.clear();
        std::stringstream ss(hidden_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) hidden_sizes.push_back(std::stoi(tok));
    }

    AgentOps ops[2];
    std::unique_ptr<NFSPAgent> dqn_agents[2];
    std::unique_ptr<NFSPIQNAgent> iqn_agents[2];

    RiskDistortion dist = RiskDistortion::NONE;
    if (risk == "cvar") dist = RiskDistortion::CVAR;
    else if (risk == "seeking") dist = RiskDistortion::SEEKING;

    for (int p = 0; p < 2; ++p) {
        if (agent_type == "nfsp") {
            auto ag = std::make_unique<NFSPAgent>(
                p, eta, dqn_lr, avg_lr, hidden_sizes, batch_size, dqn_buf, res_buf,
                learn_every, min_buf, target_update_freq_steps, tau, eps_start, eps_end, eps_duration, gamma,
                num_updates_per_learn);
            auto* ptr = ag.get();
            ops[p] = {
                [ptr](std::mt19937& r){ ptr->sample_episode_mode(r); },
                [ptr](auto& s, int a, float r, auto& ns, float d, auto& m){ ptr->add_transition(s,a,r,ns,d,m); },
                [ptr](auto& s, auto& pr, auto& m, std::mt19937& r){ ptr->add_reservoir(s,pr,m,r); },
                [ptr](int n, bool br, std::mt19937& r){ return ptr->increment_steps(n,br,r); },
                [ptr](int ng, int nb, std::mt19937& r){ return ptr->catch_up_steps(ng,nb,r); },
                [ptr](auto& q, auto& av){ ptr->sync_fast_to(q, av); },
                [ptr]{ return ptr->br_steps_val(); },
                [ptr]{ return ptr->last_br_loss(); },
                [ptr]{ return ptr->last_avg_loss(); },
                [ptr](const std::string& path){ ptr->save_weights(path); },
                [ptr](const std::string& d){ ptr->save_checkpoint(d); },
                [ptr](const std::string& d){ return ptr->load_checkpoint(d); },
            };
            dqn_agents[p] = std::move(ag);
        } else {
            auto ag = std::make_unique<NFSPIQNAgent>(
                p, eta, dqn_lr, avg_lr, hidden_sizes, batch_size, dqn_buf, res_buf,
                learn_every, min_buf, target_update_freq_steps, tau, eps_start, eps_end, eps_duration, gamma,
                iqn_N, iqn_K, iqn_emb, iqn_kappa, dist, risk_p1, risk_p2, var_penalty,
                num_updates_per_learn);
            auto* ptr = ag.get();
            ops[p] = {
                [ptr](std::mt19937& r){ ptr->sample_episode_mode(r); },
                [ptr](auto& s, int a, float r, auto& ns, float d, auto& m){ ptr->add_transition(s,a,r,ns,d,m); },
                [ptr](auto& s, auto& pr, auto& m, std::mt19937& r){ ptr->add_reservoir(s,pr,m,r); },
                [ptr](int n, bool br, std::mt19937& r){ return ptr->increment_steps(n,br,r); },
                [ptr](int ng, int nb, std::mt19937& r){ return ptr->catch_up_steps(ng,nb,r); },
                [ptr](auto& q, auto& av){ ptr->sync_fast_to(q, av); },
                [ptr]{ return ptr->br_steps_val(); },
                [ptr]{ return ptr->last_br_loss(); },
                [ptr]{ return ptr->last_avg_loss(); },
                [ptr](const std::string& path){ ptr->save_weights(path); },
                [ptr](const std::string& d){ ptr->save_checkpoint(d); },
                [ptr](const std::string& d){ return ptr->load_checkpoint(d); },
            };
            iqn_agents[p] = std::move(ag);
        }
    }

    // Try to resume from checkpoint
    int start_episode = 0;
    std::system(("mkdir -p " + checkpoint_dir).c_str());
    {
        int latest = 0;
        for (int ep = checkpoint_freq; ep <= num_episodes; ep += checkpoint_freq) {
            std::string dir = checkpoint_dir + "/" + name + "_ep" + std::to_string(ep) + "_p0";
            std::ifstream test(dir + "/state.bin");
            if (test.is_open()) latest = ep;
        }
        if (latest > 0) {
            for (int p = 0; p < 2; ++p) {
                std::string dir = checkpoint_dir + "/" + name + "_ep" + std::to_string(latest) + "_p" + std::to_string(p);
                ops[p].load_checkpoint(dir);
            }
            start_episode = latest;
            std::cout << "RESUMED from checkpoint at episode " << start_episode << std::endl;
        }
    }

    // Worker context
    WorkerContext wctx;
    wctx.eta = eta;
    wctx.epsilon_start = eps_start;
    wctx.epsilon_end = eps_end;
    wctx.epsilon_decay_duration = eps_duration;
    // FastMLP views (worker-visible, mutex-protected)
    // IQN's mean-Q approximation is 1-layer; baseline uses full architecture
    std::vector<int> fast_q_sizes = (agent_type == "nfsp_iqn")
        ? std::vector<int>{hidden_sizes[0]}
        : hidden_sizes;
    FastNetsView views[2];
    for (int p = 0; p < 2; ++p) {
        views[p].fast_q.init(fast_q_sizes);
        views[p].fast_avg.init(hidden_sizes);
        ops[p].sync_fast_to(views[p].fast_q, views[p].fast_avg);
        wctx.views[p] = &views[p];
        wctx.br_steps[p].store(ops[p].br_steps_val());
    }

    BoundedQueue<Batch> queue(8);
    std::atomic<bool> stop{false};

    // Signal handler for clean Ctrl+C shutdown
    static std::atomic<bool>* stop_ptr = &stop;
    std::signal(SIGINT, [](int) { stop_ptr->store(true); });
    std::signal(SIGTERM, [](int) { stop_ptr->store(true); });

    // Start per-agent gradient threads
    std::unique_ptr<AgentLearner> learners[2];
    for (int p = 0; p < 2; ++p) {
        learners[p] = std::make_unique<AgentLearner>(
            p, &ops[p], &views[p], &wctx.br_steps[p], seed);
    }

    std::vector<std::thread> workers;
    for (int w = 0; w < num_workers; ++w)
        workers.emplace_back(worker_thread, w, &wctx, &queue, &stop, worker_batch);

    double reward_accum = 0;
    float br_loss_accum = 0, avg_loss_accum = 0;
    int loss_count = 0;
    double last_h2h = 0.0;
    std::vector<std::pair<int, double>> h2h_log;
    auto t0 = std::chrono::steady_clock::now();
    auto t_int = t0;

    std::string log_dir = "results/logs";
    std::string weights_dir = "eval_weights_" + name;
    std::system(("mkdir -p " + weights_dir + " " + log_dir + " results/figures").c_str());
    std::string experiment_id = name + "_seed" + std::to_string(seed);
    JSONLLogger logger(log_dir, experiment_id);

    std::string arch_str;
    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        if (i > 0) arch_str += ",";
        arch_str += std::to_string(hidden_sizes[i]);
    }
    std::cout << "NFSP C++ Hold'em " << name << " (" << agent_type << ") | seed=" << seed
              << " | " << num_episodes << " episodes | " << num_workers << " workers"
              << " | arch=[" << arch_str << "] | batch=" << batch_size
              << " | learn_every=" << learn_every << " | target=" << target_update_freq_steps
              << " | dqn_lr=" << dqn_lr << " | sl_lr=" << avg_lr
              << " | dqn_buf=" << dqn_buf << " | res_buf=" << res_buf
              << std::endl;
    std::cout << std::string(90, '=') << std::endl;

    int episode = start_episode;
    Batch batch;

    while (episode < num_episodes && !stop.load()) {
        if (!queue.pop(batch)) break;

        for (int ei = 0; ei < batch.size; ++ei) {
            auto& result = batch.episodes[ei];
            if (++episode > num_episodes) break;

            for (int p = 0; p < 2; ++p) {
                for (int ti = 0; ti < result.num_transitions[p]; ++ti) {
                    auto& t = result.transitions[p][ti];
                    ops[p].add_transition(t.state, t.action, t.reward, t.next_state, t.done, t.next_legal);
                }
                for (int ri = 0; ri < result.num_reservoir[p]; ++ri) {
                    auto& rs = result.reservoir[p][ri];
                    ops[p].add_reservoir(rs.state, rs.probs, rs.mask, rng);
                }
            }
            reward_accum += result.returns[0] * REWARD_SCALE;

            // Signal gradient threads with (game_steps, br_steps) deltas.
            // The async AgentLearner handles all learning + FastMLP sync.
            for (int p = 0; p < 2; ++p) {
                int ns = result.num_transitions[p] + 1;
                bool br = (result.modes[p] == 0);
                learners[p]->signal(ns, br ? ns : 0);
            }

            for (int p = 0; p < 2; ++p) {
                float bl = ops[p].last_br_loss(), al = ops[p].last_avg_loss();
                if (bl > 0) { br_loss_accum += bl; loss_count++; }
                if (al > 0) avg_loss_accum += al;
            }

            if (episode % log_freq == 0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - t0).count();
                double interval = std::chrono::duration<double>(now - t_int).count();
                double eps_sec = log_freq / std::max(interval, 0.001);
                double avg_r = reward_accum / log_freq;
                double avg_br = loss_count > 0 ? br_loss_accum / loss_count : 0.0;
                double avg_ap = loss_count > 0 ? avg_loss_accum / loss_count : 0.0;

                std::cout << "[Ep " << std::setw(9) << episode << "/" << num_episodes << "] "
                          << std::fixed << std::setprecision(3)
                          << "avg_r=" << std::setw(7) << avg_r
                          << " | h2h=" << std::setw(8) << std::setprecision(4) << last_h2h
                          << " | br=" << std::setw(7) << avg_br
                          << " | avg=" << std::setw(7) << avg_ap
                          << " | " << std::setprecision(0) << std::setw(6) << eps_sec << " ep/s"
                          << " | " << std::setprecision(1) << elapsed/60 << "m" << std::endl;

                logger.log_stats(episode, avg_r, last_h2h, avg_br, avg_ap,
                                 eps_sec, elapsed / 60.0);

                reward_accum=0; br_loss_accum=0; avg_loss_accum=0; loss_count=0; t_int=now;
            }

            if (episode % eval_freq == 0) {
                for (int p = 0; p < 2; ++p)
                    ops[p].save_weights(weights_dir + "/p" + std::to_string(p));
                double winrate = compute_h2h(weights_dir, episode,
                    eval_script, log_dir, name);
                if (winrate >= 0) {
                    last_h2h = winrate;
                    h2h_log.push_back({episode, winrate});
                    std::cout << "  >>> H2H WINRATE @ ep " << episode << ": "
                              << std::fixed << std::setprecision(4) << winrate << std::endl;
                    logger.log_h2h(episode, winrate);
                }
            }

            if (episode % checkpoint_freq == 0) {
                for (int p = 0; p < 2; ++p) {
                    std::string dir = checkpoint_dir + "/" + name + "_ep" +
                                     std::to_string(episode) + "_p" + std::to_string(p);
                    ops[p].save_checkpoint(dir);
                }
                int prev_ep = episode - checkpoint_freq;
                if (prev_ep > 0) {
                    std::string cmd = "rm -rf " + checkpoint_dir + "/" + name + "_ep" +
                                     std::to_string(prev_ep) + "_p0 " +
                                     checkpoint_dir + "/" + name + "_ep" +
                                     std::to_string(prev_ep) + "_p1";
                    std::system(cmd.c_str());
                }
                std::cout << "  [CHECKPOINT] saved at ep " << episode << std::endl;
            }
        }
    }

    stop.store(true); queue.shutdown(); queue.drain();
    for (auto& w : workers) w.join();
    // Shut down gradient threads after workers have stopped producing more data
    for (int p = 0; p < 2; ++p) learners[p]->shutdown();

    auto tf = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(tf - t0).count();
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "DONE " << name << " | " << std::fixed << std::setprecision(1)
              << total/60 << "m" << std::endl;

    std::system(("mkdir -p final_weights_" + name).c_str());
    for (int p = 0; p < 2; ++p)
        ops[p].save_weights("final_weights_" + name + "/p" + std::to_string(p));

    // Save h2h log as CSV
    {
        std::string h2h_path = log_dir + "/" + experiment_id + "_h2h.csv";
        std::ofstream f(h2h_path);
        f << "episode,h2h_winrate\n";
        for (auto& [ep, wr] : h2h_log)
            f << ep << "," << std::fixed << std::setprecision(8) << wr << "\n";
        std::cout << "H2H log saved to " << h2h_path << std::endl;
    }

    return 0;
}
