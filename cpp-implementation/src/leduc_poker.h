#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <utility>
#include <random>

// Leduc Poker: 6 cards (3 ranks x 2 suits), 2 players, 2 betting rounds
// Matches OpenSpiel's leduc_poker encoding exactly.
//
// Info state tensor (30 dim):
//   [0:2]   = player id one-hot
//   [2:8]   = private card one-hot (6 cards)
//   [8:14]  = community card one-hot (6 cards, zeros if unrevealed)
//   [14:22] = round 1 betting history (2 bits per action: [call/check, raise])
//   [22:30] = round 2 betting history (same encoding)
//
// Actions: 0=fold, 1=call/check, 2=raise
// Bet sizes: round 1 raise=2, round 2 raise=4, ante=1

constexpr int NUM_CARDS = 6;
constexpr int NUM_RANKS = 3;
constexpr int NUM_ACTIONS = 3;
constexpr int INFO_STATE_SIZE = 30;

// Actions
constexpr int ACTION_FOLD = 0;
constexpr int ACTION_CALL = 1;  // also check
constexpr int ACTION_RAISE = 2;

class LeducState {
public:
    LeducState() { reset(); }

    void reset() {
        phase_ = DEAL_P0;
        p0_card_ = -1;
        p1_card_ = -1;
        community_card_ = -1;
        round_ = 0;
        current_player_ = 0;
        terminal_ = false;
        pot_ = {1, 1};  // ante
        round_raises_ = 0;
        round_actions_ = 0;
        betting_offset_ = {0, 0};
        folded_player_ = -1;
        num_history_ = 0;
    }

    bool is_terminal() const { return terminal_; }
    bool is_chance_node() const { return phase_ == DEAL_P0 || phase_ == DEAL_P1 || phase_ == DEAL_COMMUNITY; }
    int current_player() const { return current_player_; }

    std::array<float, INFO_STATE_SIZE> info_state_tensor(int player) const {
        std::array<float, INFO_STATE_SIZE> t{};
        // Player id
        t[player] = 1.0f;
        // Private card
        int card = (player == 0) ? p0_card_ : p1_card_;
        if (card >= 0) t[2 + card] = 1.0f;
        // Community card
        if (community_card_ >= 0) t[8 + community_card_] = 1.0f;
        // Betting history
        for (int i = 0; i < num_history_; ++i) {
            int round = history_[i].round;
            int action = history_[i].action;
            int idx = history_[i].offset;
            int base = (round == 0) ? 14 : 22;
            if (action == ACTION_CALL) {
                t[base + 2 * idx] = 1.0f;
            } else if (action == ACTION_RAISE) {
                t[base + 2 * idx + 1] = 1.0f;
            }
            // fold doesn't get recorded (game ends)
        }
        return t;
    }

    std::array<float, NUM_ACTIONS> legal_actions_mask() const {
        std::array<float, NUM_ACTIONS> mask{};
        for (int a : legal_actions()) mask[a] = 1.0f;
        return mask;
    }

    std::vector<int> legal_actions() const {
        std::vector<int> actions;
        if (terminal_ || is_chance_node()) return actions;

        actions.push_back(ACTION_CALL);  // always can check/call

        if (round_raises_ > 0) {
            actions.insert(actions.begin(), ACTION_FOLD);  // can fold if facing a bet
        }
        if (round_raises_ < 2) {
            actions.push_back(ACTION_RAISE);  // max 2 raises per round
        }
        return actions;
    }

    // Chance outcomes: (action, probability)
    std::vector<std::pair<int, double>> chance_outcomes() const {
        std::vector<std::pair<int, double>> outcomes;
        std::vector<int> available;

        for (int c = 0; c < NUM_CARDS; ++c) {
            if (c != p0_card_ && c != p1_card_ && c != community_card_) {
                available.push_back(c);
            }
        }
        double prob = 1.0 / available.size();
        for (int c : available) {
            outcomes.push_back({c, prob});
        }
        return outcomes;
    }

    void apply_action(int action) {
        if (phase_ == DEAL_P0) {
            p0_card_ = action;
            phase_ = DEAL_P1;
        } else if (phase_ == DEAL_P1) {
            p1_card_ = action;
            phase_ = PLAY;
            current_player_ = 0;
            round_ = 0;
            round_raises_ = 0;
            round_actions_ = 0;
            betting_offset_ = {0, 0};
        } else if (phase_ == DEAL_COMMUNITY) {
            community_card_ = action;
            phase_ = PLAY;
            current_player_ = 0;
            round_ = 1;
            round_raises_ = 0;
            round_actions_ = 0;
        } else {
            // Player action
            apply_player_action(action);
        }
    }

    // Apply chance action with sampling
    int sample_chance_action(std::mt19937& rng) const {
        auto outcomes = chance_outcomes();
        std::uniform_int_distribution<int> dist(0, (int)outcomes.size() - 1);
        return outcomes[dist(rng)].first;
    }

    std::array<double, 2> returns() const {
        if (folded_player_ >= 0) {
            // Folder loses what they put in
            std::array<double, 2> r{};
            r[folded_player_] = -(double)pot_[folded_player_];
            r[1 - folded_player_] = (double)pot_[folded_player_];
            return r;
        }
        // Showdown
        int winner = showdown_winner();
        if (winner == -1) {
            // Tie
            return {0.0, 0.0};
        }
        int loser = 1 - winner;
        std::array<double, 2> r{};
        r[winner] = (double)pot_[loser];
        r[loser] = -(double)pot_[loser];
        return r;
    }

private:
    enum Phase { DEAL_P0, DEAL_P1, PLAY, DEAL_COMMUNITY };

    struct HistoryEntry {
        int round;
        int action;
        int offset;  // action index within the round
    };

    void apply_player_action(int action) {
        int offset = betting_offset_[round_];
        history_[num_history_++] = {round_, action, offset};
        betting_offset_[round_]++;

        if (action == ACTION_FOLD) {
            folded_player_ = current_player_;
            terminal_ = true;
            return;
        }

        int raise_amount = (round_ == 0) ? 2 : 4;

        if (action == ACTION_RAISE) {
            // Match current bet + raise
            int to_call = pot_[1 - current_player_] - pot_[current_player_];
            pot_[current_player_] += to_call + raise_amount;
            round_raises_++;
            round_actions_++;
            current_player_ = 1 - current_player_;
        } else {  // CALL / CHECK
            int to_call = pot_[1 - current_player_] - pot_[current_player_];
            pot_[current_player_] += to_call;
            round_actions_++;

            // Check if round is over
            // Round ends when: both players have acted at least once AND last action was call/check
            if (round_actions_ >= 2) {
                end_round();
                return;
            }
            current_player_ = 1 - current_player_;
        }
    }

    void end_round() {
        if (round_ == 0) {
            // Go to community card deal
            phase_ = DEAL_COMMUNITY;
        } else {
            // Showdown
            terminal_ = true;
        }
    }

    int showdown_winner() const {
        int p0_rank = p0_card_ / 2;
        int p1_rank = p1_card_ / 2;
        int comm_rank = community_card_ / 2;

        bool p0_pair = (p0_rank == comm_rank);
        bool p1_pair = (p1_rank == comm_rank);

        if (p0_pair && !p1_pair) return 0;
        if (p1_pair && !p0_pair) return 1;

        // Both pair or neither pair — compare private card rank
        if (p0_rank > p1_rank) return 0;
        if (p1_rank > p0_rank) return 1;
        return -1;  // tie
    }

    Phase phase_;
    int p0_card_, p1_card_, community_card_;
    int round_;
    int current_player_;
    bool terminal_;
    std::array<int, 2> pot_;
    int round_raises_;
    int round_actions_;
    std::array<int, 2> betting_offset_;  // per-round action count for encoding
    int folded_player_;
    HistoryEntry history_[8];  // max 8 actions per game (4 per round × 2 rounds)
    int num_history_;
};
