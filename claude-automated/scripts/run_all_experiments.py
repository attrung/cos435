#!/usr/bin/env python3
"""Launch 4 NFSP experiments with live progress bars and training metrics."""

import os
import sys
import signal
import subprocess
import time
from tqdm import tqdm

EXPERIMENTS = [
    ('configs/nfsp_baseline.yaml',     'nfsp_baseline',     'BASELINE'),
    ('configs/nfsp_iqn_neutral.yaml',  'nfsp_iqn_neutral',  'IQN-NEUTRAL'),
    ('configs/nfsp_iqn_mean_var.yaml',    'nfsp_iqn_mean_var',    'IQN-MV-0.1'),
    ('configs/nfsp_iqn_mean_var_05.yaml', 'nfsp_iqn_mean_var_05', 'IQN-MV-0.5'),
    ('configs/nfsp_iqn_averse.yaml',      'nfsp_iqn_averse',      'IQN-AVERSE'),
]

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
os.makedirs('results/logs', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

PROGRESS_DIR = 'results/.progress'
os.makedirs(PROGRESS_DIR, exist_ok=True)

all_procs = []
log_files = []


def cleanup(signum=None, frame=None):
    for p in all_procs:
        try:
            p.terminate()
        except (OSError, ProcessLookupError):
            pass
    deadline = time.time() + 3
    for p in all_procs:
        remaining = max(0, deadline - time.time())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except (OSError, ProcessLookupError):
                pass
    for f in log_files:
        try:
            f.close()
        except Exception:
            pass
    print('\n  All processes stopped.')
    sys.exit(1)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def clean_experiment(name):
    for path in [
        f'results/logs/{name}_seed42.jsonl',
        f'results/logs/{name}_seed42_exploitability.npy',
        f'results/logs/{name}.log',
    ]:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        os.remove(os.path.join(PROGRESS_DIR, name))
    except OSError:
        pass


def read_progress(name):
    path = os.path.join(PROGRESS_DIR, name)
    try:
        with open(path) as f:
            parts = f.read().strip().split()
            return int(parts[0]), int(parts[1])
    except (OSError, ValueError, IndexError):
        return 0, 0


def tail_log_lines(log_path, last_pos):
    """Read new lines from a log file since last_pos. Returns (new_lines, new_pos)."""
    lines = []
    try:
        with open(log_path, 'r') as f:
            f.seek(last_pos)
            for line in f:
                stripped = line.strip()
                # Only show important lines: exploitability evals and periodic stats
                if 'EXPLOITABILITY' in stripped or stripped.startswith('[Ep'):
                    lines.append(stripped)
            new_pos = f.tell()
        return lines, new_pos
    except OSError:
        return [], last_pos


def main():
    print('=' * 60)
    print(f'  NFSP Leduc Poker — {len(EXPERIMENTS)} experiments')
    print(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 60)
    print()

    procs = []
    names = []

    for config, name, label in EXPERIMENTS:
        clean_experiment(name)
        progress_file = os.path.join(PROGRESS_DIR, name)
        lf = open(f'results/logs/{name}.log', 'w')
        log_files.append(lf)
        cmd = [
            sys.executable, '-u', 'src/train.py',
            '--config', config,
            '--progress-file', progress_file,
        ]
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(proc)
        names.append(name)
        all_procs.append(proc)
        print(f'  [{label}] PID {proc.pid}')

    print()

    # Progress bars — one per experiment, positioned below each other
    bars = []
    for i, name in enumerate(names):
        bar = tqdm(
            total=1, desc=name, unit='ep', position=i,
            smoothing=0.05, mininterval=1.0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        )
        bars.append(bar)

    # Track log file read positions for tailing
    log_positions = {name: 0 for name in names}

    try:
        while True:
            all_done = True
            new_log_lines = []

            for i, (proc, name) in enumerate(zip(procs, names)):
                # Update progress bar
                current, total = read_progress(name)
                if total > 0 and bars[i].total != total:
                    bars[i].total = total
                    bars[i].refresh()
                if current > bars[i].n:
                    bars[i].update(current - bars[i].n)
                if proc.poll() is None:
                    all_done = False

                # Tail log file for new training metrics
                log_path = f'results/logs/{name}.log'
                lines, new_pos = tail_log_lines(log_path, log_positions[name])
                log_positions[name] = new_pos
                for line in lines:
                    new_log_lines.append(f'  {name}: {line}')

            # Print new log lines above the progress bars
            if new_log_lines:
                for line in new_log_lines:
                    tqdm.write(line)

            if all_done:
                # Final progress update
                for i, name in enumerate(names):
                    current, total = read_progress(name)
                    if current > bars[i].n:
                        bars[i].update(current - bars[i].n)
                break

            time.sleep(1.0)
    finally:
        for bar in bars:
            bar.close()

    print('\n' * len(bars), end='')

    # Check exit codes
    failed = 0
    for proc, name in zip(procs, names):
        proc.wait()
        if proc.returncode != 0:
            print(f'  FAILED: {name} (exit code {proc.returncode}) — check results/logs/{name}.log')
            failed += 1
        else:
            print(f'  DONE: {name}')

    for f in log_files:
        f.close()

    print()
    if failed == 0:
        print('=== GENERATING RESULTS ===')
        subprocess.run([sys.executable, 'scripts/generate_results.py'], check=False)
        print()
        print('=' * 60)
        print(f'  ALL DONE: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        print('  Logs:    results/logs/')
        print('  Figures: results/figures/')
        print('=' * 60)
    else:
        print(f'  {failed} experiment(s) failed. Check logs.')
        sys.exit(1)


if __name__ == '__main__':
    main()
