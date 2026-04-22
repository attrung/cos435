"""Utility to load C++ trained models (any architecture) into Python."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VarMLP(nn.Module):
    """Matches C++ VarMLPImpl — per-layer hidden sizes."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        in_sz = input_size
        for h in hidden_sizes:
            self.layers.append(nn.Linear(in_sz, h))
            in_sz = h
        self.output = nn.Linear(in_sz, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)


def load_models(weights_dir, input_size=208, num_actions=3):
    """Auto-detect architecture (per-layer hidden sizes) from saved weights."""
    models = []
    for p in range(2):
        path = f'{weights_dir}/p{p}_avg.pt'
        jm = torch.jit.load(path, map_location='cpu')
        sd = jm.state_dict()

        # Keys like: fc0.weight, fc1.weight, ..., output.weight
        # Discover per-layer hidden sizes (output dim of each fc layer)
        fc_indices = sorted(
            int(k.split('.')[0][2:]) for k in sd.keys()
            if k.startswith('fc') and k.endswith('.weight')
        )
        hidden_sizes = [sd[f'fc{i}.weight'].shape[0] for i in fc_indices]

        model = VarMLP(input_size, hidden_sizes, num_actions)
        new_sd = {}
        for k, v in sd.items():
            if k.startswith('fc'):
                idx = k.split('.')[0][2:]  # "fc0.weight" -> "0"
                rest = k.split('.', 1)[1]   # "weight" or "bias"
                new_sd[f'layers.{idx}.{rest}'] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd)
        model.eval()
        models.append(model)
    return models
