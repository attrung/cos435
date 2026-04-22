#pragma once

// Limit Texas Hold'em wrapper using OpenSpiel's C++ API.
// Wraps open_spiel::universal_poker to match the interface used by our
// NFSP agents (same API shape as LeducState).
//
// Game: 2-player Limit Hold'em
//   - 52 cards, 4 betting rounds (preflop, flop, turn, river)
//   - info_state_size = 208
//   - num_actions = 3 (fold, call/check, raise)
//   - blinds: 50/100, raise sizes: 100/100/200/200, max 3 raises/round

#include <array>
#include <vector>
#include <random>
#include <memory>
#include <string>
#include <cstdlib>
#include <iostream>

#include <open_spiel/spiel.h>

// Game constants for Hold'em — must be defined before including agent headers.
// These shadow the Leduc constants when building train_holdem.
constexpr int NUM_ACTIONS = 3;
constexpr int INFO_STATE_SIZE = 208;

// Action constants (same as Leduc: fold=0, call=1, raise=2)
constexpr int ACTION_FOLD = 0;
constexpr int ACTION_CALL = 1;
constexpr int ACTION_RAISE = 2;

class HoldemState {
public:
    static constexpr int INFO_SIZE = 208;
    static constexpr int NUM_ACT = 3;

    // Game string for 2-player Limit Hold'em matching OpenSpiel's encoding
    static const std::string& game_string() {
        static const std::string s =
            "universal_poker(betting=limit,numPlayers=2,numRounds=4,"
            "blind=50 100,firstPlayer=2 1 1 1,numSuits=4,numRanks=13,"
            "numHoleCards=2,numBoardCards=0 3 1 1,"
            "raiseSize=100 100 200 200,maxRaises=3 3 3 3)";
        return s;
    }

    HoldemState() {
        game_ = open_spiel::LoadGame(game_string());
        // Verify dimensions
        int expected_info = game_->InformationStateTensorSize();
        int expected_actions = game_->NumDistinctActions();
        if (expected_info != INFO_SIZE) {
            std::cerr << "FATAL: info_state_size mismatch: expected "
                      << INFO_SIZE << ", got " << expected_info << std::endl;
            std::abort();
        }
        if (expected_actions != NUM_ACT) {
            std::cerr << "FATAL: num_actions mismatch: expected "
                      << NUM_ACT << ", got " << expected_actions << std::endl;
            std::abort();
        }
        state_ = game_->NewInitialState();
    }

    void reset() {
        state_ = game_->NewInitialState();
    }

    bool is_terminal() const {
        return state_->IsTerminal();
    }

    bool is_chance_node() const {
        return state_->IsChanceNode();
    }

    int current_player() const {
        return state_->CurrentPlayer();
    }

    std::vector<int> legal_actions() const {
        auto actions = state_->LegalActions();
        return std::vector<int>(actions.begin(), actions.end());
    }

    std::array<float, INFO_SIZE> info_state_tensor(int player) const {
        std::array<float, INFO_SIZE> result{};
        std::vector<float> tensor = state_->InformationStateTensor(player);
        for (int i = 0; i < INFO_SIZE && i < (int)tensor.size(); ++i) {
            result[i] = tensor[i];
        }
        return result;
    }

    std::array<float, NUM_ACT> legal_actions_mask() const {
        std::array<float, NUM_ACT> mask{};
        auto actions = state_->LegalActions();
        for (auto a : actions) {
            if (a >= 0 && a < NUM_ACT) mask[a] = 1.0f;
        }
        return mask;
    }

    void apply_action(int action) {
        state_->ApplyAction(action);
    }

    int sample_chance_action(std::mt19937& rng) const {
        auto outcomes = state_->ChanceOutcomes();
        // Build cumulative distribution
        std::vector<double> cum_probs;
        cum_probs.reserve(outcomes.size());
        double cumsum = 0.0;
        for (auto& [action, prob] : outcomes) {
            cumsum += prob;
            cum_probs.push_back(cumsum);
        }
        std::uniform_real_distribution<double> dist(0.0, cumsum);
        double r = dist(rng);
        for (size_t i = 0; i < outcomes.size(); ++i) {
            if (r <= cum_probs[i]) {
                return outcomes[i].first;
            }
        }
        return outcomes.back().first;
    }

    std::array<double, 2> returns() const {
        auto r = state_->Returns();
        return {r[0], r[1]};
    }

private:
    std::shared_ptr<const open_spiel::Game> game_;
    std::unique_ptr<open_spiel::State> state_;
};
