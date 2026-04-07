"""Utility functions: config loading, seeding, logging, checkpointing."""

import os
import yaml
import torch
import numpy as np
import random
import json
from datetime import datetime


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_gpu():
    """Set up GPU with 30% memory cap."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.30, device=0)
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def save_checkpoint(path, agent, episode, exploitability_log):
    """Save agent checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'episode': episode,
        'exploitability_log': exploitability_log,
    }
    checkpoint.update(agent.get_state_dict())
    torch.save(checkpoint, path)


def load_checkpoint(path, agent):
    """Load agent checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    agent.load_state_dict(checkpoint)
    return checkpoint['episode'], checkpoint['exploitability_log']


class Logger:
    """Simple logger that writes to file and stdout."""

    def __init__(self, log_dir, experiment_name):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f'{experiment_name}.jsonl')
        self.experiment_name = experiment_name

    def log(self, data):
        """Log a dictionary as a JSONL line."""
        data['timestamp'] = datetime.now().isoformat()
        data['experiment'] = self.experiment_name
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def log_exploitability(self, episode, exploitability):
        self.log({'episode': episode, 'exploitability': exploitability})
