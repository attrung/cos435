#!/usr/bin/env python3
"""Generate comparison plots and summary from training logs."""

import os
import sys
import glob
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('WARNING: matplotlib not available, skipping plots')

LOG_DIR = 'results/logs'
FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_exploitability_logs():
    """Load all exploitability data. Tries .npy first, falls back to .jsonl."""
    results = {}
    # Try npy files first
    for path in sorted(glob.glob(os.path.join(LOG_DIR, '*_exploitability.npy'))):
        name = os.path.basename(path).replace('_exploitability.npy', '')
        data = np.load(path)
        if data.size > 0 and data.ndim == 2:
            results[name] = data  # shape (N, 2): [episode, exploitability]

    # Fall back to jsonl for experiments missing from npy
    for path in sorted(glob.glob(os.path.join(LOG_DIR, '*.jsonl'))):
        name = os.path.basename(path).replace('.jsonl', '')
        if name in results:
            continue
        points = []
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    if 'exploitability_eval' in e:
                        points.append([e['episode'], e['exploitability_eval']])
                except json.JSONDecodeError:
                    pass
        if points:
            results[name] = np.array(points)
            print(f'  Loaded {name} from jsonl ({len(points)} points)')

    return results


def load_jsonl_logs():
    """Load JSONL log files for speed/loss stats."""
    results = {}
    for path in sorted(glob.glob(os.path.join(LOG_DIR, '*.jsonl'))):
        name = os.path.basename(path).replace('.jsonl', '')
        entries = []
        with open(path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        if entries:
            results[name] = entries
    return results


def plot_exploitability(exp_data):
    """Plot exploitability curves for all experiments."""
    if not HAS_MPL or not exp_data:
        return

    plt.figure(figsize=(12, 7))

    colors = {
        'baseline': '#2196F3',
        'iqn_neutral': '#4CAF50',
        'iqn_mean_var': '#9C27B0',
        'iqn_averse': '#FF9800',
        'iqn_seeking': '#F44336',
    }

    for name, data in sorted(exp_data.items()):
        episodes = data[:, 0]
        exploit = data[:, 1]
        # Pick color by matching keywords
        color = '#888888'
        label = name
        for key, c in colors.items():
            if key in name:
                color = c
                break
        plt.plot(episodes, exploit, label=label, color=color, linewidth=1.5)

    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Exploitability (NashConv)', fontsize=12)
    plt.title('NFSP Exploitability: DQN vs IQN Variants on Leduc Poker', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exploitability_comparison.png'), dpi=150)
    plt.close()
    print(f'  Saved: {FIG_DIR}/exploitability_comparison.png')

    # Linear scale version
    plt.figure(figsize=(12, 7))
    for name, data in sorted(exp_data.items()):
        episodes = data[:, 0]
        exploit = data[:, 1]
        color = '#888888'
        for key, c in colors.items():
            if key in name:
                color = c
                break
        plt.plot(episodes, exploit, label=name, color=color, linewidth=1.5)

    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Exploitability (NashConv)', fontsize=12)
    plt.title('NFSP Exploitability (Linear Scale)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exploitability_linear.png'), dpi=150)
    plt.close()
    print(f'  Saved: {FIG_DIR}/exploitability_linear.png')


def generate_summary(exp_data, jsonl_data):
    """Generate full_comparison.md."""
    lines = ['# NFSP Experiment Results — Leduc Poker\n']
    lines.append(f'Generated automatically.\n')

    # Exploitability summary
    lines.append('## Exploitability Summary\n')
    lines.append('| Experiment | Final Exploitability | Min Exploitability | Episodes |')
    lines.append('|---|---|---|---|')

    for name, data in sorted(exp_data.items()):
        final = data[-1, 1]
        best = data[:, 1].min()
        eps = int(data[-1, 0])
        lines.append(f'| {name} | {final:.6f} | {best:.6f} | {eps:,} |')

    lines.append('')

    # Speed summary from JSONL logs
    lines.append('## Training Speed\n')
    for name, entries in sorted(jsonl_data.items()):
        speed_entries = [e for e in entries if 'eps_per_sec' in e]
        if speed_entries:
            speeds = [e['eps_per_sec'] for e in speed_entries]
            avg_speed = np.mean(speeds)
            elapsed_entries = [e for e in entries if 'elapsed_min' in e]
            total_min = elapsed_entries[-1]['elapsed_min'] if elapsed_entries else 0
            lines.append(f'- **{name}**: avg {avg_speed:.0f} eps/sec, total {total_min:.1f} min')

    lines.append('')
    lines.append('## Figures\n')
    lines.append('- `results/figures/exploitability_comparison.png` — log-scale comparison')
    lines.append('- `results/figures/exploitability_linear.png` — linear-scale comparison')
    lines.append('')

    summary = '\n'.join(lines)

    with open('results/full_comparison.md', 'w') as f:
        f.write(summary)
    print(f'  Saved: results/full_comparison.md')

    # Also print to stdout
    print('\n' + summary)


def update_speedup_notes():
    """Extract speedup measurements from logs."""
    single_log = os.path.join(LOG_DIR, 'speedup_single.log')
    parallel_log = os.path.join(LOG_DIR, 'speedup_parallel.log')

    if not (os.path.exists(single_log) and os.path.exists(parallel_log)):
        return

    def extract_final_speed(log_path):
        """Get average eps/sec from log file."""
        speeds = []
        with open(log_path) as f:
            for line in f:
                if 'eps/s=' in line:
                    try:
                        part = line.split('eps/s=')[1].split('|')[0].strip()
                        speeds.append(float(part))
                    except (ValueError, IndexError):
                        pass
        return np.mean(speeds) if speeds else 0

    single_speed = extract_final_speed(single_log)
    parallel_speed = extract_final_speed(parallel_log)
    speedup = parallel_speed / max(single_speed, 1)

    notes_path = 'notes/speedup_notes.md'
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            content = f.read()

        # Update the measurement section
        measurement = f"""
## Measured Results

- Single-process: {single_speed:.0f} eps/sec
- Parallel (15 workers): {parallel_speed:.0f} eps/sec
- **Speedup: {speedup:.1f}x**
"""
        if '## Measured Results' in content:
            # Replace existing section
            before = content.split('## Measured Results')[0]
            after_parts = content.split('## Measured Results')[1].split('\n## ')
            if len(after_parts) > 1:
                after = '\n## ' + '\n## '.join(after_parts[1:])
            else:
                after = ''
            content = before + measurement + after
        else:
            content += '\n' + measurement

        with open(notes_path, 'w') as f:
            f.write(content)
        print(f'  Updated: {notes_path} with speedup={speedup:.1f}x')


if __name__ == '__main__':
    print('Loading experiment data...')
    exp_data = load_exploitability_logs()
    jsonl_data = load_jsonl_logs()

    if not exp_data:
        print('No exploitability data found in results/logs/. Run experiments first.')
        sys.exit(0)

    print(f'Found {len(exp_data)} experiments: {list(exp_data.keys())}')
    print()

    print('Generating plots...')
    plot_exploitability(exp_data)

    print('Generating summary...')
    generate_summary(exp_data, jsonl_data)

    print('Updating speedup notes...')
    update_speedup_notes()

    print('\nDone!')
