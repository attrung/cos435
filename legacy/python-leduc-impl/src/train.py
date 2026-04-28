#!/usr/bin/env python3
"""Main training script for NFSP on Leduc poker.

Parallel architecture with shared memory:
  - Model weights shared via torch.share_memory_() (zero-copy reads)
  - Workers play episodes, push transitions via queue (batched to reduce IPC)
  - Main thread processes transitions, runs gradient updates
  - Workers see updated weights immediately (shared memory, no pickling)
  - Bounded queue ensures workers can't run more than ~1 learning cycle ahead
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
torch.set_num_threads(1)
# Use fork so children inherit shared memory tensors
import multiprocessing as mp
try:
    mp.set_start_method('fork')
except RuntimeError:
    pass  # already set

import argparse
import time
import io
import contextlib
import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    with contextlib.redirect_stdout(io.StringIO()):
        import pyspiel

from tqdm import tqdm

from src.utils import load_config, set_seed, save_checkpoint, load_checkpoint, Logger
from src.nfsp_agent import NFSPAgent
from src.nfsp_iqn_agent import NFSPIQNAgent
from src.evaluate import compute_exploitability


def create_agents(config, state_size, num_actions, device):
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
    """Play one episode. Returns (transitions, reservoir_samples, returns, modes)."""
    state = game.new_initial_state()

    #create containers for transitions and reservoir samples for each player
    player_transitions = {0: [], 1: []} 
    reservoir_samples = {0: [], 1: []} 

    # record each player’s current mode (best_response or average_policy)
    modes = {p: agents[p]._mode for p in range(2)}

    while not state.is_terminal():
        # in OpenSpiel, the game is a tree
        # a chance node is a step where neither player chooses, the environment samples randomness according to the game rules
        # if the state is a chance node (dealing cards, burning cards)
        # get all chance outcome, sample an action and apply that action
        if state.is_chance_node():
            outcomes = state.chance_outcomes()         
            action_list, prob_list = zip(*outcomes) 
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)

        # otherwise, a player acts
        else:
            player = state.current_player() #get the current player (0 or 1 in a 2-player game)

            # get the observation that player is allowed to see (imperfect information)
            info_state = np.array(state.information_state_tensor(player), dtype=np.float32)

            # which actions are allowed; the mask is a fixed-size vector for the neural nets
            legal_actions = state.legal_actions() 
            legal_mask = np.zeros(num_actions, dtype=np.float32)
            for a in legal_actions:
                legal_mask[a] = 1.0

            # the NFSP agent picks an action (and returns mode/probs for logging/training).
            action, mode, probs = agents[player].step(info_state, legal_actions)

            if mode == 'best_response':
                # reservoir samples are stored only in best-response mode
                reservoir_samples[player].append((info_state.copy(), probs.copy(), legal_mask.copy())) 

            # stores what we need for learning
            player_transitions[player].append({
                'state': info_state.copy(),
                'action': action,
                'legal_mask': legal_mask.copy(),
            })
            state.apply_action(action) #apply action

    returns = state.returns()

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

    return processed, reservoir_samples, returns, modes


def _write_progress(progress_file, episodes, total):
    if progress_file:
        try:
            with open(progress_file + '.tmp', 'w') as f:
                f.write(f'{episodes} {total}')
            os.replace(progress_file + '.tmp', progress_file)
        except OSError:
            pass


def _worker_process(worker_id, agents, state_size, num_actions, result_queue,
                    stop_event, batch_size, game_name='leduc_poker'):
    """Worker process: plays episodes using shared model weights.

    Model weights are in shared memory (via torch.share_memory_()).
    Workers read weights directly — no pickling, no IPC for weights.
    Only transition data goes through the queue.
    """
    np.random.seed(worker_id * 7919 + int(time.time()) % 100000)
    torch.set_num_threads(1)

    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.redirect_stdout(io.StringIO()):
            import pyspiel

    game = pyspiel.load_game(game_name)

    while not stop_event.is_set():
        # Play a batch of episodes using shared model weights directly
        batch = []
        for _ in range(batch_size):
            for player in range(2):
                agents[player].sample_episode_mode()
            transitions, reservoir_samples, returns, modes = play_episode(
                game, agents, num_actions, state_size)
            batch.append((transitions, reservoir_samples, returns, modes))

        # BLOCKS if queue full → prevents running ahead of learning
        try:
            result_queue.put(batch, timeout=5.0)
        except Exception:
            if stop_event.is_set():
                break


def train(config, seed, device, max_episodes=None, progress_file=None, num_workers=5):
    """NFSP training with shared-memory parallel episode generation."""
    experiment_name = config.get('experiment_name', 'nfsp')
    num_episodes = max_episodes or config.get('num_episodes', 20000000)
    eval_freq = config.get('eval_freq', 10000)
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    log_dir = config.get('log_dir', 'results/logs')
    checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
    log_freq = 50000
    learn_every = config.get('learn_every', 64)

    set_seed(seed)

    game_name = config.get('game_name', 'leduc_poker')
    game = pyspiel.load_game(game_name)
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

    agents = create_agents(config, state_size, num_actions, device)

    # Try to resume from latest checkpoint
    start_episode = 0
    exploitability_log = []
    last_exp = float('inf')

    latest_ep = 0
    for p in range(2):
        import glob
        pattern = os.path.join(checkpoint_dir, f'{experiment_name}_seed{seed}_p{p}_ep*.pt')
        files = glob.glob(pattern)
        for f in files:
            # Extract episode number from filename
            base = os.path.basename(f)
            try:
                ep = int(base.split('_ep')[1].split('.pt')[0])
                latest_ep = max(latest_ep, ep)
            except (ValueError, IndexError):
                pass

    if latest_ep > 0:
        # Load both agents from latest checkpoint
        for p in range(2):
            path = os.path.join(checkpoint_dir,
                                f'{experiment_name}_seed{seed}_p{p}_ep{latest_ep}.pt')
            if os.path.exists(path):
                ep, exp_log = load_checkpoint(path, agents[p])
                if p == 0:
                    exploitability_log = exp_log
                    if exploitability_log:
                        last_exp = exploitability_log[-1][1]
        start_episode = latest_ep
        print(f'RESUMED from checkpoint at episode {start_episode:,} (exploit={last_exp:.6f})')

    # Move model weights to shared memory — workers read them directly
    for agent in agents:
        if hasattr(agent, 'q_network'):
            agent.q_network.share_memory()
            agent.avg_policy_network.share_memory()
        if hasattr(agent, 'iqn'):
            agent.iqn.network.share_memory()
            agent.avg_policy_network.share_memory()
        # Share _br_steps tensor so workers see updated epsilon
        if isinstance(agent._br_steps, torch.Tensor):
            agent._br_steps.share_memory_()

    logger = Logger(log_dir, f'{experiment_name}_seed{seed}')
    reward_accum = 0.0
    br_loss_accum = 0.0
    avg_loss_accum = 0.0
    loss_count = 0

    start_time = time.time()
    interval_start = time.time()

    # Batch ≈ learn_every game steps. Queue holds 2 batches max.
    worker_batch_size = max(1, learn_every // 4)
    queue_maxsize = 2

    remaining = num_episodes - start_episode
    print(f'Training {experiment_name} | seed={seed} | {num_episodes:,} episodes | {num_workers} workers')
    print(f'learn_every={learn_every} | batch={worker_batch_size} | shared_memory=True')
    if start_episode > 0:
        print(f'Resuming from episode {start_episode:,} ({remaining:,} remaining)')
    print('=' * 95)

    result_queue = mp.Queue(maxsize=queue_maxsize)
    stop_event = mp.Event()
    workers = []

    for w in range(num_workers):
        p = mp.Process(
            target=_worker_process,
            args=(w, agents, state_size, num_actions, result_queue,
                  stop_event, worker_batch_size, game_name),
            daemon=True)
        p.start()
        workers.append(p)

    pbar = tqdm(total=num_episodes, initial=start_episode, desc=experiment_name,
                unit='ep', smoothing=0.05, mininterval=2.0)

    episode = start_episode
    try:
        while episode < num_episodes:
            batch = result_queue.get()

            for transitions, reservoir_samples, returns, modes in batch:
                episode += 1
                if episode > num_episodes:
                    break

                pbar.update(1)

                for player in range(2):
                    for t in transitions[player]:
                        agents[player].add_transition(*t)
                    for s, probs, legal_mask in reservoir_samples[player]:
                        agents[player].add_reservoir(s, probs, legal_mask)

                reward_accum += returns[0]

                # Increment steps — triggers learning when crossing learn_every boundaries
                # +1 for terminal step (matching OpenSpiel which calls step() at terminal)
                # is_br_mode separates epsilon/target counters from learning counters
                for player in range(2):
                    num_steps = len(transitions[player]) + 1  # +1 terminal
                    is_br = (modes[player] == 'best_response')
                    agents[player].increment_steps(num_steps, is_br)

                for player in range(2):
                    br_loss, avg_loss = agents[player].losses
                    if br_loss is not None:
                        br_loss_accum += br_loss
                        loss_count += 1
                    if avg_loss is not None:
                        avg_loss_accum += avg_loss

                if episode % 1000 == 0:
                    _write_progress(progress_file, episode, num_episodes)

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

                if episode % eval_freq == 0:
                    skip_exploit = config.get('skip_exploitability', False)
                    if not skip_exploit:
                        exp = compute_exploitability(game, agents, device)
                        exploitability_log.append((episode, exp))
                        last_exp = exp
                        tqdm.write(f'  >>> EXPLOITABILITY @ ep {episode:,}: {exp:.6f}')
                        logger.log({'episode': episode, 'exploitability_eval': exp})
                    else:
                        tqdm.write(f'  [eval @ ep {episode:,}] (exploitability skipped — use head-to-head)')
                        logger.log({'episode': episode, 'eval_checkpoint': True})

                if episode % checkpoint_freq == 0:
                    for p in range(2):
                        path = os.path.join(checkpoint_dir,
                                            f'{experiment_name}_seed{seed}_p{p}_ep{episode}.pt')
                        save_checkpoint(path, agents[p], episode, exploitability_log)
                    # Delete previous checkpoint to save disk
                    prev_ep = episode - checkpoint_freq
                    if prev_ep > 0:
                        for p in range(2):
                            prev_path = os.path.join(checkpoint_dir,
                                                     f'{experiment_name}_seed{seed}_p{p}_ep{prev_ep}.pt')
                            if os.path.exists(prev_path):
                                os.remove(prev_path)

    finally:
        stop_event.set()
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except Exception:
                break
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

    pbar.close()

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
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of parallel episode worker processes')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')
    train(config, args.seed, device, max_episodes=args.max_episodes,
          progress_file=args.progress_file, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
