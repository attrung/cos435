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

#include "leduc_poker.h"
#include "nfsp_agent.h"
#include "nfsp_iqn_agent.h"

// ── Episode data (fixed-capacity, zero-alloc after init) ──────────────────

// Max transitions per player per episode in Leduc: 4 (2 rounds × 2 actions max)
constexpr int MAX_TRANS_PER_PLAYER = 4;
constexpr int MAX_RESERVOIR_PER_PLAYER = 4;

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

// Batch: fixed-size array of episodes, pre-allocated once per worker
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

struct WorkerContext {
    FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS> fast_q[2];
    FastMLP<INFO_STATE_SIZE, 128, NUM_ACTIONS> fast_avg[2];
    float eta;
    float epsilon_start, epsilon_end;
    int epsilon_decay_duration;
    std::atomic<int> br_steps[2];
};

void play_episode_worker(LeducState& state, WorkerContext& ctx,
                         EpisodeResult& result, std::mt19937& rng) {
    result.clear();
    state.reset();

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (int p = 0; p < 2; ++p)
        result.modes[p] = (uniform(rng) < ctx.eta) ? 0 : 1;

    // Pending actions per player (stack-allocated, max 4 per player)
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
                    action = ctx.fast_q[p].forward_argmax(info.data(), mask.data());
                    probs[action] = 1.0f;
                }
                int ri = result.num_reservoir[p]++;
                result.reservoir[p][ri] = {info, probs, mask};
            } else { // AVG
                action = ctx.fast_avg[p].forward_sample(info.data(), mask.data(), uniform(rng));
            }
            int pi = npending[p]++;
            pending[p][pi] = {info, action, mask};
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
                last ? (float)result.returns[p] : 0.0f,
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
    LeducState state;
    Batch batch;
    while (!stop->load(std::memory_order_relaxed)) {
        batch.size = bsz;
        for (int i = 0; i < bsz; ++i)
            play_episode_worker(state, *ctx, batch.episodes[i], rng);
        q->push(batch);  // Batch is POD-like, no heap alloc
    }
}

// ── JSONL Logger (matches Python's Logger class) ──────────────────────────

class JSONLLogger {
public:
    JSONLLogger(const std::string& log_dir, const std::string& experiment_name)
        : experiment_(experiment_name) {
        std::system(("mkdir -p " + log_dir).c_str());
        path_ = log_dir + "/" + experiment_name + ".jsonl";
        // Clear existing log
        std::ofstream(path_, std::ios::trunc).close();
    }

    void log_stats(int episode, double avg_reward, double exploitability,
                   double br_loss, double avg_pol_loss, double eps_per_sec,
                   double elapsed_min) {
        auto ts = timestamp();
        std::ofstream f(path_, std::ios::app);
        f << "{\"episode\": " << episode
          << ", \"avg_reward\": " << avg_reward
          << ", \"exploitability\": " << exploitability
          << ", \"br_loss\": " << br_loss
          << ", \"avg_pol_loss\": " << avg_pol_loss
          << ", \"eps_per_sec\": " << eps_per_sec
          << ", \"elapsed_min\": " << elapsed_min
          << ", \"timestamp\": \"" << ts << "\""
          << ", \"experiment\": \"" << experiment_ << "\"}\n";
    }

    void log_exploitability(int episode, double expl) {
        auto ts = timestamp();
        std::ofstream f(path_, std::ios::app);
        f << "{\"episode\": " << episode
          << ", \"exploitability_eval\": " << expl
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

// ── Exploitability ────────────────────────────────────────────────────────

double compute_exploitability(const std::string& weights_dir, int episode,
                              const std::string& eval_script,
                              const std::string& log_dir, const std::string& name) {
    // Save a marker file with the exploitability result
    std::string result_file = weights_dir + "/expl_result.txt";
    std::string cmd = "python3 " + eval_script +
                      " --weights-dir " + weights_dir +
                      " --episode " + std::to_string(episode) +
                      " --result-file " + result_file;
    std::system(cmd.c_str());

    // Read back the result
    double expl = -1.0;
    std::ifstream f(result_file);
    if (f.is_open()) f >> expl;
    return expl;
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
    std::function<void(FastMLP<INFO_STATE_SIZE,128,NUM_ACTIONS>&,
                       FastMLP<INFO_STATE_SIZE,128,NUM_ACTIONS>&)> sync_fast_to;
    std::function<int()> br_steps_val;
    std::function<float()> last_br_loss;
    std::function<float()> last_avg_loss;
    std::function<void(const std::string&)> save_weights;
    std::function<void(const std::string&)> save_checkpoint;
    std::function<bool(const std::string&)> load_checkpoint;
};

// ── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Defaults: baseline config
    int num_episodes = 40'000'000;
    std::string agent_type = "nfsp";
    std::string name = "baseline";
    float eta = 0.1f;
    float dqn_lr = 0.01f;
    float avg_lr = 0.01f;
    int hidden = 128, batch_size = 128;
    int dqn_buf = 200000, res_buf = 2000000;
    int learn_every = 64, min_buf = 1000;
    int target_freq = 19200;
    float tau = 0.995f;
    float eps_start = 0.06f, eps_end = 0.001f;
    int eps_duration = 20000000;
    float gamma = 1.0f;
    int eval_freq = 200000, log_freq = 50000, checkpoint_freq = 1000000;
    int seed = 42, num_workers = 1, worker_batch = 16;
    std::string checkpoint_dir = "checkpoints";
    // IQN params
    int iqn_N = 8, iqn_K = 32, iqn_emb = 64;
    float iqn_kappa = 1.0f;
    std::string risk = "none";
    float risk_p1 = 0.25f, risk_p2 = 1.0f;
    float var_penalty = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]{ return argv[++i]; };
        if (a == "--episodes") num_episodes = std::stoi(next());
        else if (a == "--agent") agent_type = next();
        else if (a == "--name") name = next();
        else if (a == "--dqn-lr") dqn_lr = std::stof(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--eval-freq") eval_freq = std::stoi(next());
        else if (a == "--workers") num_workers = std::stoi(next());
        else if (a == "--risk") risk = next();
        else if (a == "--risk-p1") risk_p1 = std::stof(next());
        else if (a == "--risk-p2") risk_p2 = std::stof(next());
        else if (a == "--var-penalty") var_penalty = std::stof(next());
        else if (a == "--checkpoint-freq") checkpoint_freq = std::stoi(next());
        else if (a == "--checkpoint-dir") checkpoint_dir = next();
        else if (a == "--worker-batch") worker_batch = std::stoi(next());
        else if (a == "--iqn-n") iqn_N = std::stoi(next());
    }

    torch::set_num_threads(1);
    std::mt19937 rng(seed);

    // Create agents
    AgentOps ops[2];
    // We need to keep the actual agent objects alive
    std::unique_ptr<NFSPAgent> dqn_agents[2];
    std::unique_ptr<NFSPIQNAgent> iqn_agents[2];

    RiskDistortion dist = RiskDistortion::NONE;
    if (risk == "cvar") dist = RiskDistortion::CVAR;
    else if (risk == "seeking") dist = RiskDistortion::SEEKING;

    for (int p = 0; p < 2; ++p) {
        if (agent_type == "nfsp") {
            auto ag = std::make_unique<NFSPAgent>(
                p, eta, dqn_lr, avg_lr, hidden, batch_size, dqn_buf, res_buf,
                learn_every, min_buf, target_freq, tau, eps_start, eps_end, eps_duration, gamma);
            auto* ptr = ag.get();
            ops[p] = {
                [ptr](std::mt19937& r){ ptr->sample_episode_mode(r); },
                [ptr](auto& s, int a, float r, auto& ns, float d, auto& m){ ptr->add_transition(s,a,r,ns,d,m); },
                [ptr](auto& s, auto& pr, auto& m, std::mt19937& r){ ptr->add_reservoir(s,pr,m,r); },
                [ptr](int n, bool br, std::mt19937& r){ return ptr->increment_steps(n,br,r); },
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
                p, eta, dqn_lr, avg_lr, hidden, batch_size, dqn_buf, res_buf,
                learn_every, min_buf, target_freq, tau, eps_start, eps_end, eps_duration, gamma,
                iqn_N, iqn_K, iqn_emb, iqn_kappa, dist, risk_p1, risk_p2, var_penalty);
            auto* ptr = ag.get();
            ops[p] = {
                [ptr](std::mt19937& r){ ptr->sample_episode_mode(r); },
                [ptr](auto& s, int a, float r, auto& ns, float d, auto& m){ ptr->add_transition(s,a,r,ns,d,m); },
                [ptr](auto& s, auto& pr, auto& m, std::mt19937& r){ ptr->add_reservoir(s,pr,m,r); },
                [ptr](int n, bool br, std::mt19937& r){ return ptr->increment_steps(n,br,r); },
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
        // Find latest checkpoint
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
    for (int p = 0; p < 2; ++p) {
        ops[p].sync_fast_to(wctx.fast_q[p], wctx.fast_avg[p]);
        wctx.br_steps[p].store(ops[p].br_steps_val());
    }

    BoundedQueue<Batch> queue(2);
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    for (int w = 0; w < num_workers; ++w)
        workers.emplace_back(worker_thread, w, &wctx, &queue, &stop, worker_batch);

    double reward_accum = 0;
    float br_loss_accum = 0, avg_loss_accum = 0;
    int loss_count = 0;
    double last_exp = 999.0;
    std::vector<std::pair<int, double>> exploitability_log;
    auto t0 = std::chrono::steady_clock::now();
    auto t_int = t0;

    std::string log_dir = "results/logs";
    std::string weights_dir = "eval_weights_" + name;
    std::system(("mkdir -p " + weights_dir + " " + log_dir + " results/figures").c_str());
    std::string experiment_id = name + "_seed" + std::to_string(seed);
    JSONLLogger logger(log_dir, experiment_id);

    std::cout << "NFSP C++ " << name << " (" << agent_type << ") | seed=" << seed
              << " | " << num_episodes << " episodes | " << num_workers << " workers" << std::endl;
    std::cout << std::string(90, '=') << std::endl;

    int episode = start_episode;
    Batch batch;

    while (episode < num_episodes) {
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
            reward_accum += result.returns[0];

            for (int p = 0; p < 2; ++p) {
                int ns = result.num_transitions[p] + 1;
                bool br = (result.modes[p] == 0);
                bool learned = ops[p].increment_steps(ns, br, rng);
                if (learned) {
                    ops[p].sync_fast_to(wctx.fast_q[p], wctx.fast_avg[p]);
                    wctx.br_steps[p].store(ops[p].br_steps_val(), std::memory_order_relaxed);
                }
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
                          << " | exploit=" << std::setw(8) << std::setprecision(4) << last_exp
                          << " | br=" << std::setw(7) << avg_br
                          << " | avg=" << std::setw(7) << avg_ap
                          << " | " << std::setprecision(0) << std::setw(6) << eps_sec << " ep/s"
                          << " | " << std::setprecision(1) << elapsed/60 << "m" << std::endl;

                logger.log_stats(episode, avg_r, last_exp, avg_br, avg_ap,
                                 eps_sec, elapsed / 60.0);

                reward_accum=0; br_loss_accum=0; avg_loss_accum=0; loss_count=0; t_int=now;
            }

            if (episode % eval_freq == 0) {
                for (int p = 0; p < 2; ++p)
                    ops[p].save_weights(weights_dir + "/p" + std::to_string(p));
                double expl = compute_exploitability(weights_dir, episode,
                    "eval/compute_exploitability.py", log_dir, name);
                if (expl >= 0) {
                    last_exp = expl;
                    exploitability_log.push_back({episode, expl});
                    std::cout << "  >>> EXPLOITABILITY @ ep " << episode << ": "
                              << std::fixed << std::setprecision(6) << expl << std::endl;
                    logger.log_exploitability(episode, expl);
                }
            }

            if (episode % checkpoint_freq == 0) {
                for (int p = 0; p < 2; ++p) {
                    std::string dir = checkpoint_dir + "/" + name + "_ep" +
                                     std::to_string(episode) + "_p" + std::to_string(p);
                    ops[p].save_checkpoint(dir);
                }
                // Delete previous checkpoint to save disk space
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

    auto tf = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(tf - t0).count();
    std::cout << std::string(90, '=') << std::endl;
    std::cout << "DONE " << name << " | " << std::fixed << std::setprecision(1)
              << total/60 << "m" << std::endl;

    std::system(("mkdir -p final_weights_" + name).c_str());
    for (int p = 0; p < 2; ++p)
        ops[p].save_weights("final_weights_" + name + "/p" + std::to_string(p));

    // Save exploitability log as CSV (easy to load in Python with np.loadtxt)
    {
        std::string expl_path = log_dir + "/" + experiment_id + "_exploitability.csv";
        std::ofstream f(expl_path);
        f << "episode,exploitability\n";
        for (auto& [ep, ex] : exploitability_log)
            f << ep << "," << std::fixed << std::setprecision(8) << ex << "\n";
        std::cout << "Exploitability log saved to " << expl_path << std::endl;
    }

    return 0;
}
