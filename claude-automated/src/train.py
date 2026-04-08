#!/usr/bin/env python3
"""Main training script for NFSP on Leduc poker.

Single-process training matching Heinrich & Silver (2016):
  for each episode:
      play episode with latest weights
      update Q-network (1 step per player)
      update avg policy (1 step per player)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)

import torch
torch.set_num_threads(1)  # tiny networks: multi-threaded ops add overhead, not speed

import argparse
import time
import io
import contextlib
import numpy as np

# Suppress open_spiel's noisy print about optional modules
with contextlib.redirect_stderr(io.StringIO()):
    with contextlib.redirect_stdout(io.StringIO()):
        import pyspiel

from tqdm import tqdm

from src.utils import load_config, set_seed, save_checkpoint, Logger
from src.nfsp_agent import NFSPAgent
from src.nfsp_iqn_agent import NFSPIQNAgent
from src.evaluate import compute_exploitability


def create_agents(config, state_size, num_actions, device):
    """Create pair of NFSP agents (DQN or IQN best-response)."""
    agent_type = config.get('agent_type', 'nfsp')
    agents = []
    for player_id in range(2):
        if agent_type == 'nfsp':
            agent = NFSPAgent(player_id, state_size, num_actions, config, device)
        elif agent_type == 'nfsp_iqn':
            agent = NFSPIQNAgent(player_id, state_size, num_actions, config, device)
        else:
            raise ValueError(f'Unknown agent type: {agent_type}')
        agents.append(agent)
    return agents


def play_episode(game, agents, num_actions, state_size):
    """Play one Leduc poker episode. Returns transitions, reservoir samples, returns."""
    state = game.new_initial_state()
    player_transitions = {0: [], 1: []}
    reservoir_samples = {0: [], 1: []}

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            player = state.current_player()
            info_state = np.array(state.information_state_tensor(player), dtype=np.float32)
            legal_actions = state.legal_actions()
            legal_mask = np.zeros(num_actions, dtype=np.float32)
            for a in legal_actions:
                legal_mask[a] = 1.0

            action, mode = agents[player].step(info_state, legal_actions)

            if mode == 'best_response':
                reservoir_samples[player].append((info_state.copy(), action))

            player_transitions[player].append({
                'state': info_state.copy(),
                'action': action,
                'legal_mask': legal_mask.copy(),
            })
            state.apply_action(action)

    returns = state.returns()

    # Convert to buffer-ready tuples
    processed = {0: [], 1: []}
    for player in range(2):
        transitions = player_transitions[player]
        for i, t in enumerate(transitions):
            is_last = (i == len(transitions) - 1)
            reward = returns[player] if is_last else 0.0
            next_state = np.zeros(state_size, dtype=np.float32) if is_last else transitions[i + 1]['state']
            next_legal = np.ones(num_actions, dtype=np.float32) if is_last else transitions[i + 1]['legal_mask']
            done = 1.0 if is_last else 0.0
            processed[player].append((t['state'], t['action'], reward, next_state, done, next_legal))

    return processed, reservoir_samples, returns


def _write_progress(progress_file, episodes, total):
    """Write current progress to file for external monitoring."""
    if progress_file:
        try:
            with open(progress_file + '.tmp', 'w') as f:
                f.write(f'{episodes} {total}')
            os.replace(progress_file + '.tmp', progress_file)
        except OSError:
            pass


def train(config, seed, device, max_episodes=None, progress_file=None):
    """NFSP training matching Heinrich & Silver (2016) exactly.

    Each episode: play with latest weights, then 1 gradient step per player.
    """
    experiment_name = config.get('experiment_name', 'nfsp')
    num_episodes = max_episodes or config.get('num_episodes', 3000000)
    eval_freq = config.get('eval_freq', 50000)
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    log_dir = config.get('log_dir', 'results/logs')
    checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
    log_freq = 50000

    set_seed(seed)

    game = pyspiel.load_game('leduc_poker')
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

    agents = create_agents(config, state_size, num_actions, device)
    logger = Logger(log_dir, f'{experiment_name}_seed{seed}')

    exploitability_log = []
    last_exp = float('inf')
    reward_accum = 0.0
    br_loss_accum = 0.0
    avg_loss_accum = 0.0
    loss_count = 0

    start_time = time.time()
    interval_start = time.time()

    print(f'Training {experiment_name} | seed={seed} | {num_episodes:,} episodes')
    print(f'Eval every {eval_freq:,} | Log every {log_freq:,} | Device: {device}')
    print('=' * 95)

    pbar = tqdm(range(1, num_episodes + 1), desc=experiment_name, unit='ep',
                smoothing=0.05, mininterval=2.0)

    for episode in pbar:
        # LR decay: linear anneal based on progress
        progress = episode / num_episodes
        for player in range(2):
            agents[player].set_lr(progress)

        transitions, reservoir_samples, returns = play_episode(game, agents, num_actions, state_size)

        for player in range(2):
            for t in transitions[player]:
                agents[player].add_transition(*t)
            for s, a in reservoir_samples[player]:
                agents[player].add_reservoir(s, a)

        reward_accum += returns[0]

        # Update both agents (1 gradient step per player per episode)
        for player in range(2):
            losses = agents[player].update()
            if losses[0] is not None:
                br_loss_accum += losses[0]
                loss_count += 1
            if losses[1] is not None:
                avg_loss_accum += losses[1]

        # Progress file update (frequent, for external monitoring)
        if episode % 1000 == 0:
            _write_progress(progress_file, episode, num_episodes)

        # Periodic logging
        if episode % log_freq == 0:
            elapsed = time.time() - start_time
            interval = time.time() - interval_start
            eps_sec = log_freq / max(interval, 0.001)
            avg_r = reward_accum / log_freq
            avg_br = br_loss_accum / max(loss_count, 1)
            avg_ap = avg_loss_accum / max(loss_count, 1)

            msg = (f'[Ep {episode:>9,}/{num_episodes:,}] '
                   f'avg_reward={avg_r:>7.3f} | '
                   f'exploit={last_exp:>8.4f} | '
                   f'br_loss={avg_br:>7.4f} | '
                   f'avg_pol={avg_ap:>7.4f} | '
                   f'eps/s={eps_sec:>7.1f} | '
                   f'{elapsed / 60:>5.1f}m')
            tqdm.write(msg)

            logger.log({
                'episode': episode, 'avg_reward': avg_r, 'exploitability': last_exp,
                'br_loss': avg_br, 'avg_pol_loss': avg_ap, 'eps_per_sec': eps_sec,
                'elapsed_min': elapsed / 60,
            })

            reward_accum = 0.0
            br_loss_accum = 0.0
            avg_loss_accum = 0.0
            loss_count = 0
            interval_start = time.time()

        # Exploitability evaluation
        if episode % eval_freq == 0:
            exp = compute_exploitability(game, agents, device)
            exploitability_log.append((episode, exp))
            last_exp = exp
            tqdm.write(f'  >>> EXPLOITABILITY @ ep {episode:,}: {exp:.6f}')
            logger.log({'episode': episode, 'exploitability_eval': exp})

        # Checkpoint
        if episode % checkpoint_freq == 0:
            for p in range(2):
                path = os.path.join(checkpoint_dir, f'{experiment_name}_seed{seed}_p{p}_ep{episode}.pt')
                save_checkpoint(path, agents[p], episode, exploitability_log)

    # Final save
    elapsed = time.time() - start_time
    final_exp = exploitability_log[-1][1] if exploitability_log else float('inf')

    for p in range(2):
        path = os.path.join(checkpoint_dir, f'{experiment_name}_seed{seed}_p{p}_final.pt')
        save_checkpoint(path, agents[p], episode, exploitability_log)

    os.makedirs(log_dir, exist_ok=True)
    np.save(os.path.join(log_dir, f'{experiment_name}_seed{seed}_exploitability.npy'),
            np.array(exploitability_log) if exploitability_log else np.empty((0, 2)))

    print(f'\n{"=" * 95}')
    print(f'DONE: {experiment_name} seed={seed} | Final exploit={final_exp:.6f} | Time={elapsed / 60:.1f}m ({elapsed / 3600:.2f}h)')
    print(f'{"=" * 95}\n')

    return exploitability_log


def main():
    parser = argparse.ArgumentParser(description='NFSP Training for Leduc Poker')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=None, help='Override max episodes')
    parser.add_argument('--progress-file', type=str, default=None,
                        help='File to write progress for external monitoring')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')
    train(config, args.seed, device, max_episodes=args.max_episodes,
          progress_file=args.progress_file)


if __name__ == '__main__':
    main()
