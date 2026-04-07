#!/usr/bin/env python3
"""Main training script for NFSP on Leduc poker.

Supports single-process and parallel (15 workers) training.
CPU-only. Rich logging every 1000 episodes.
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
torch.set_num_threads(18)

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
import multiprocessing as mp
import threading
import queue as queue_mod

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
    """Play one Leduc poker episode. Returns processed transitions, reservoir samples, returns."""
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


# =============================================================================
# SINGLE-PROCESS TRAINING
# =============================================================================

def train_single(config, seed, device, max_episodes=None):
    """Single-process NFSP training with rich logging."""
    experiment_name = config.get('experiment_name', 'nfsp')
    num_episodes = max_episodes or config.get('num_episodes', 10000000)
    eval_freq = config.get('eval_freq', 50000)
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    log_dir = config.get('log_dir', 'results/logs')
    checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
    log_freq = 100000

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

    print(f'Training {experiment_name} | seed={seed} | {num_episodes:,} episodes | mode=single')
    print(f'Eval every {eval_freq:,} | Log every {log_freq:,} | Device: {device}')
    print('=' * 95)

    pbar = tqdm(range(1, num_episodes + 1), desc=experiment_name, unit='ep',
                smoothing=0.05, mininterval=2.0)

    for episode in pbar:
        transitions, reservoir_samples, returns = play_episode(game, agents, num_actions, state_size)

        for player in range(2):
            for t in transitions[player]:
                agents[player].add_transition(*t)
            for s, a in reservoir_samples[player]:
                agents[player].add_reservoir(s, a)

        reward_accum += returns[0]

        # Update both agents
        for player in range(2):
            losses = agents[player].update()
            if losses[0] is not None:
                br_loss_accum += losses[0]
                loss_count += 1
            if losses[1] is not None:
                avg_loss_accum += losses[1]

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


# =============================================================================
# PARALLEL TRAINING (18 workers)
# =============================================================================

def _pack_transitions(transitions, reservoir):
    """Pack transitions into compact numpy arrays for efficient queue transfer."""
    packed = {}
    for p in range(2):
        if transitions[p]:
            states, actions, rewards, next_states, dones, legal_masks = zip(*transitions[p])
            packed[f't_states_{p}'] = np.array(states, dtype=np.float32)
            packed[f't_actions_{p}'] = np.array(actions, dtype=np.int64)
            packed[f't_rewards_{p}'] = np.array(rewards, dtype=np.float32)
            packed[f't_next_{p}'] = np.array(next_states, dtype=np.float32)
            packed[f't_dones_{p}'] = np.array(dones, dtype=np.float32)
            packed[f't_legal_{p}'] = np.array(legal_masks, dtype=np.float32)
        else:
            packed[f't_states_{p}'] = None

        if reservoir[p]:
            r_states, r_actions = zip(*reservoir[p])
            packed[f'r_states_{p}'] = np.array(r_states, dtype=np.float32)
            packed[f'r_actions_{p}'] = np.array(r_actions, dtype=np.int64)
        else:
            packed[f'r_states_{p}'] = None
    return packed


def _worker_loop(worker_id, game_name, config, state_size, num_actions,
                 result_queue, stop_event, episodes_per_batch, weight_path):
    """Worker: play episodes, send packed transitions back. Reads weights from file."""
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    np.random.seed(worker_id * 7919 + int(time.time() * 1000) % 100000)
    import random
    random.seed(worker_id * 7919 + int(time.time() * 1000) % 100000)

    game = pyspiel.load_game(game_name)
    device = torch.device('cpu')
    agents = create_agents(config, state_size, num_actions, device)

    last_weight_mtime = 0.0

    while not stop_event.is_set():
        # Check for new weights on disk (atomic file, no FD leaks)
        try:
            mtime = os.path.getmtime(weight_path)
            if mtime > last_weight_mtime:
                weights = torch.load(weight_path, map_location='cpu', weights_only=False)
                for p in range(2):
                    agents[p].load_state_dict(weights[p])
                last_weight_mtime = mtime
        except Exception:
            pass  # file being written, skip this cycle

        # Play batch of episodes
        all_transitions = {0: [], 1: []}
        all_reservoir = {0: [], 1: []}
        total_reward = 0.0

        for _ in range(episodes_per_batch):
            trans, res, rets = play_episode(game, agents, num_actions, state_size)
            for p in range(2):
                all_transitions[p].extend(trans[p])
                all_reservoir[p].extend(res[p])
            total_reward += rets[0]

        # Pack into numpy arrays for efficient pickling (no torch tensors = no FD leaks)
        packed = _pack_transitions(all_transitions, all_reservoir)
        packed['reward_sum'] = total_reward
        packed['num_episodes'] = episodes_per_batch

        if stop_event.is_set():
            break
        try:
            result_queue.put(packed)
        except Exception:
            break  # queue closed or broken, exit cleanly


def _unpack_and_feed(packed, agents):
    """Unpack numpy arrays and feed to agent buffers using batch insertion."""
    for p in range(2):
        if packed[f't_states_{p}'] is not None:
            agents[p].add_transition_batch(
                packed[f't_states_{p}'], packed[f't_actions_{p}'],
                packed[f't_rewards_{p}'], packed[f't_next_{p}'],
                packed[f't_dones_{p}'], packed[f't_legal_{p}'],
            )

        if packed[f'r_states_{p}'] is not None:
            agents[p].add_reservoir_batch(
                packed[f'r_states_{p}'], packed[f'r_actions_{p}'],
            )


def _player_gradient_thread_fn(player_id, agent, grad_queue, grad_stop,
                               loss_accumulator, loss_lock, pause_event,
                               pause_ack):
    """Gradient thread for a single player. Two of these run in parallel.

    Player 0 and Player 1 have independent networks/buffers, so their
    gradient updates can run truly concurrently (torch releases GIL).
    """
    while not grad_stop.is_set():
        # Pause during exploitability eval / checkpointing for consistent reads
        if not pause_event.is_set():
            pause_ack.set()  # signal that we have stopped
            pause_event.wait(timeout=0.5)
            continue
        pause_ack.clear()  # we are running

        # Drain queue but cap at 20 updates to avoid unbounded backlog
        num_updates = 0
        try:
            while num_updates < 20:
                num_updates += grad_queue.get_nowait()
        except queue_mod.Empty:
            pass
        if num_updates == 0:
            num_updates = 2  # opportunistic updates

        for _ in range(num_updates):
            if grad_stop.is_set() or not pause_event.is_set():
                break
            losses = agent.update()
            if losses[0] is not None:
                with loss_lock:
                    loss_accumulator[f'br_loss_{player_id}'] += losses[0]
                    loss_accumulator[f'count_{player_id}'] += 1
            if losses[1] is not None:
                with loss_lock:
                    loss_accumulator[f'avg_loss_{player_id}'] += losses[1]

        with loss_lock:
            loss_accumulator[f'updates_{player_id}'] += num_updates


def _weight_sync_thread_fn(agents, grad_stop, loss_accumulator, loss_lock,
                           weight_path, weight_sync_freq, grad_pause):
    """Periodically writes weights to disk for workers to pick up."""
    last_sync = 0
    while not grad_stop.is_set():
        time.sleep(0.1)  # check every 100ms
        with loss_lock:
            total_updates = (loss_accumulator.get('updates_0', 0) +
                             loss_accumulator.get('updates_1', 0))
        if total_updates - last_sync >= weight_sync_freq * 2:
            # Only sync when gradient threads are running (not paused for eval)
            if not grad_pause.is_set():
                continue
            try:
                weights = {p: agents[p].get_state_dict() for p in range(2)}
                tmp_path = weight_path + '.tmp'
                torch.save(weights, tmp_path)
                os.replace(tmp_path, weight_path)
                last_sync = total_updates
            except Exception:
                pass


def train_parallel(config, seed, device, max_episodes=None, num_workers=18):
    """Parallel NFSP training with threaded gradient updates.

    Architecture:
    - 18 worker PROCESSES play episodes, send numpy-packed data via Queue
    - Main THREAD drains queue and inserts data into replay buffers
    - Gradient THREAD continuously samples from buffers and does SGD updates
    - Torch releases the GIL during forward/backward, so both threads run concurrently
    """
    experiment_name = config.get('experiment_name', 'nfsp')
    num_episodes = max_episodes or config.get('num_episodes', 10000000)
    eval_freq = config.get('eval_freq', 50000)
    checkpoint_freq = config.get('checkpoint_freq', 100000)
    log_dir = config.get('log_dir', 'results/logs')
    checkpoint_dir = config.get('checkpoint_dir', 'results/checkpoints')
    log_freq = 100000
    episodes_per_batch = 50
    updates_per_collect = 4  # gradient updates requested per drained batch
    weight_sync_freq = 100  # gradient thread syncs weights every N updates

    set_seed(seed)
    torch.set_num_threads(4)

    game = pyspiel.load_game('leduc_poker')
    state_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()

    agents = create_agents(config, state_size, num_actions, device)
    logger = Logger(log_dir, f'{experiment_name}_seed{seed}')

    # --- Worker processes ---
    stop_event = mp.Event()
    result_queue = mp.Queue()

    os.makedirs(checkpoint_dir, exist_ok=True)
    weight_path = os.path.join(checkpoint_dir, f'{experiment_name}_latest_weights.pt')
    initial_weights = {p: agents[p].get_state_dict() for p in range(2)}
    torch.save(initial_weights, weight_path)

    workers = []
    for i in range(num_workers):
        w = mp.Process(
            target=_worker_loop,
            args=(i, 'leduc_poker', config, state_size, num_actions,
                  result_queue, stop_event, episodes_per_batch, weight_path),
            daemon=True,
        )
        w.start()
        workers.append(w)

    # --- Two gradient threads (one per player) + weight sync thread ---
    # Player 0 and Player 1 have independent networks/buffers, so their
    # gradient updates run truly concurrently (torch releases GIL).
    grad_queues = [queue_mod.Queue() for _ in range(2)]
    grad_stop = threading.Event()
    grad_pause = threading.Event()
    grad_pause.set()  # start unpaused (set = running, clear = paused)
    grad_acks = [threading.Event() for _ in range(2)]  # gradient threads signal when paused
    loss_lock = threading.Lock()
    loss_accumulator = {
        'br_loss_0': 0.0, 'avg_loss_0': 0.0, 'count_0': 0, 'updates_0': 0,
        'br_loss_1': 0.0, 'avg_loss_1': 0.0, 'count_1': 0, 'updates_1': 0,
    }

    grad_threads = []
    for p in range(2):
        t = threading.Thread(
            target=_player_gradient_thread_fn,
            args=(p, agents[p], grad_queues[p], grad_stop,
                  loss_accumulator, loss_lock, grad_pause, grad_acks[p]),
            daemon=True,
        )
        grad_threads.append(t)

    sync_thread = threading.Thread(
        target=_weight_sync_thread_fn,
        args=(agents, grad_stop, loss_accumulator, loss_lock,
              weight_path, weight_sync_freq, grad_pause),
        daemon=True,
    )

    # Tracking
    exploitability_log = []
    last_exp = float('inf')
    total_episodes = 0
    reward_accum = 0.0
    ep_since_log = 0

    start_time = time.time()
    interval_start = time.time()

    print(f'Training {experiment_name} | seed={seed} | {num_episodes:,} episodes | mode=parallel ({num_workers} workers + 2 grad threads)')
    print(f'Eval every {eval_freq:,} | Log every {log_freq:,} | Device: {device}')
    print('=' * 95)

    pbar = tqdm(total=num_episodes, desc=experiment_name, unit='ep',
                smoothing=0.05, mininterval=2.0)

    # Start gradient threads after buffers have some data
    grad_threads_started = False

    try:
        while total_episodes < num_episodes:
            # --- Wait for data with proper timeout (no unreliable empty()) ---
            try:
                first_result = result_queue.get(timeout=60)
            except queue_mod.Empty:
                alive = sum(1 for w in workers if w.is_alive())
                if alive == 0:
                    raise RuntimeError(
                        f'All {num_workers} workers died. Check logs for errors.')
                tqdm.write(f'  [WARN] No data for 60s ({alive}/{num_workers} workers alive). Waiting...')
                continue

            # Drain remaining available results
            results_batch = [first_result]
            while True:
                try:
                    results_batch.append(result_queue.get_nowait())
                except queue_mod.Empty:
                    break

            prev_total = total_episodes

            # Insert data into buffers (main thread — fast numpy ops)
            for res in results_batch:
                _unpack_and_feed(res, agents)
                total_episodes += res['num_episodes']
                reward_accum += res['reward_sum']
                ep_since_log += res['num_episodes']

            pbar.update(total_episodes - prev_total)

            # Start gradient threads once we have enough data
            if not grad_threads_started and total_episodes > 500:
                for t in grad_threads:
                    t.start()
                sync_thread.start()
                grad_threads_started = True

            # Signal both gradient threads: process this many updates each
            if grad_threads_started:
                n = updates_per_collect * len(results_batch)
                for q in grad_queues:
                    q.put(n)

            # Periodic logging
            if ep_since_log >= log_freq:
                elapsed = time.time() - start_time
                interval = time.time() - interval_start
                eps_sec = ep_since_log / max(interval, 0.001)
                avg_r = reward_accum / max(ep_since_log, 1)
                with loss_lock:
                    cnt = max(loss_accumulator['count_0'] + loss_accumulator['count_1'], 1)
                    avg_br = (loss_accumulator['br_loss_0'] + loss_accumulator['br_loss_1']) / cnt
                    avg_ap = (loss_accumulator['avg_loss_0'] + loss_accumulator['avg_loss_1']) / cnt
                    grad_updates = loss_accumulator['updates_0'] + loss_accumulator['updates_1']

                    # Reset loss accumulators (not updates — those are total counters)
                    for key in loss_accumulator:
                        if key.startswith(('br_loss', 'avg_loss', 'count')):
                            loss_accumulator[key] = 0.0 if 'loss' in key else 0

                msg = (f'[Ep {total_episodes:>9,}/{num_episodes:,}] '
                       f'avg_reward={avg_r:>7.3f} | '
                       f'exploit={last_exp:>8.4f} | '
                       f'br_loss={avg_br:>7.4f} | '
                       f'avg_pol={avg_ap:>7.4f} | '
                       f'eps/s={eps_sec:>7.1f} | '
                       f'grad={grad_updates:,} | '
                       f'{elapsed / 60:>5.1f}m')
                tqdm.write(msg)

                logger.log({
                    'episode': total_episodes, 'avg_reward': avg_r, 'exploitability': last_exp,
                    'br_loss': avg_br, 'avg_pol_loss': avg_ap, 'eps_per_sec': eps_sec,
                    'elapsed_min': elapsed / 60, 'grad_updates': grad_updates,
                })

                reward_accum = 0.0
                ep_since_log = 0
                interval_start = time.time()

            # Exploitability / Checkpoint — pause gradient threads for consistent reads
            need_eval = prev_total // eval_freq < total_episodes // eval_freq
            need_ckpt = prev_total // checkpoint_freq < total_episodes // checkpoint_freq

            if need_eval or need_ckpt:
                if grad_threads_started:
                    grad_pause.clear()  # ask gradient threads to pause
                    # Wait for both gradient threads to acknowledge they've stopped
                    for ack in grad_acks:
                        ack.wait(timeout=5.0)

                if need_eval:
                    exp = compute_exploitability(game, agents, device)
                    exploitability_log.append((total_episodes, exp))
                    last_exp = exp
                    tqdm.write(f'  >>> EXPLOITABILITY @ ep {total_episodes:,}: {exp:.6f}')
                    logger.log({'episode': total_episodes, 'exploitability_eval': exp})

                if need_ckpt:
                    for p in range(2):
                        path = os.path.join(checkpoint_dir,
                                            f'{experiment_name}_seed{seed}_p{p}_ep{total_episodes}.pt')
                        save_checkpoint(path, agents[p], total_episodes, exploitability_log)

                if grad_threads_started:
                    grad_pause.set()  # resume gradient threads

    finally:
        # Stop gradient threads first
        grad_stop.set()
        grad_pause.set()  # unblock any waiting gradient threads
        if grad_threads_started:
            for t in grad_threads:
                t.join(timeout=5)
            sync_thread.join(timeout=5)

        # Stop workers
        stop_event.set()
        pbar.close()
        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

        # Final save (inside finally so it runs even on interrupt)
        elapsed = time.time() - start_time
        final_exp = exploitability_log[-1][1] if exploitability_log else float('inf')

        for p in range(2):
            path = os.path.join(checkpoint_dir, f'{experiment_name}_seed{seed}_p{p}_final.pt')
            save_checkpoint(path, agents[p], total_episodes, exploitability_log)

        os.makedirs(log_dir, exist_ok=True)
        np.save(os.path.join(log_dir, f'{experiment_name}_seed{seed}_exploitability.npy'),
                np.array(exploitability_log) if exploitability_log else np.empty((0, 2)))

        # Clean up temporary weight file
        try:
            os.remove(weight_path)
        except OSError:
            pass

        print(f'\n{"=" * 95}')
        print(f'DONE: {experiment_name} seed={seed} | Final exploit={final_exp:.6f} | Time={elapsed / 60:.1f}m ({elapsed / 3600:.2f}h)')
        print(f'{"=" * 95}\n')

    return exploitability_log


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='NFSP Training for Leduc Poker')
    parser.add_argument('--config', type=str, required=True, help='Config YAML file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_episodes', type=int, default=None, help='Override max episodes from config')
    parser.add_argument('--parallel', action='store_true', help='Use parallel training (15 workers)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cpu')

    if args.parallel:
        train_parallel(config, args.seed, device, max_episodes=args.max_episodes)
    else:
        train_single(config, args.seed, device, max_episodes=args.max_episodes)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
