"""Risk distortion functions for IQN action selection.

These functions distort the quantile levels tau during action selection
to implement risk-sensitive behavior. Training always uses uniform tau.
"""

import torch


def identity(tau):
    """Risk-neutral: no distortion."""
    return tau


def cvar(tau, alpha=0.25):
    """CVaR (Conditional Value at Risk) distortion for risk-averse behavior.

    Maps tau in [0,1] to tau' in [0, alpha], focusing on worst-case quantiles.
    tau' = tau * alpha
    """
    return tau * alpha


def risk_seeking(tau, lower=0.75, upper=1.0):
    """Risk-seeking distortion: map tau to upper quantiles.

    Maps tau in [0,1] to tau' in [lower, upper], focusing on best-case quantiles.
    tau' = lower + tau * (upper - lower)
    """
    return lower + tau * (upper - lower)


def get_distortion_fn(config):
    """Get distortion function from config."""
    distortion_type = config.get('risk_distortion', 'none')

    if distortion_type == 'none' or distortion_type is None:
        return identity
    elif distortion_type == 'cvar':
        alpha = config.get('cvar_alpha', 0.25)
        return lambda tau: cvar(tau, alpha=alpha)
    elif distortion_type == 'seeking':
        lower = config.get('seeking_lower', 0.75)
        upper = config.get('seeking_upper', 1.0)
        return lambda tau: risk_seeking(tau, lower=lower, upper=upper)
    else:
        raise ValueError(f'Unknown distortion type: {distortion_type}')
