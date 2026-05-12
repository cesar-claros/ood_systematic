"""Tunable entropy-family scoring functions.

These are the underlying scoring functions wrapped by the `EntropyScores`
class (in `src/csfs/entropy.py`) for hyperparameter-tuned use (M, gamma
chosen by Bayesian optimization). They can also be called directly with
default arguments to act as atomic base detectors.

Each takes a softmax tensor of shape (N, C) and returns a (N,)-shaped score
tensor. Higher = more confident (more likely in-distribution).
"""

import torch

from src.csfs._validators import ArrayType, validate_softmax
from src.csfs.base_detectors import predictive_collision_entropy, predictive_entropy


@validate_softmax
def generalized_entropy(softmax: ArrayType, gamma: float = 0.1, M: int | None = None) -> ArrayType:
    """Generalized entropy CSF (GEN). Returns -sum((p_top_M * (1-p_top_M))^gamma)."""
    if M is None:
        M = softmax.shape[1]
    softmax = softmax.clone()
    sorted_softmax = torch.sort(softmax, descending=True, dim=1).values
    sorted_softmax = sorted_softmax[:, :M]
    gen_ent = torch.sum((sorted_softmax * (1 - sorted_softmax)) ** gamma, dim=1)
    return -gen_ent


@validate_softmax
def renyi_entropy(softmax: ArrayType, gamma: float = 0.1, M: int | None = None) -> ArrayType:
    """Renyi entropy CSF (REN). Reduces to PE at gamma=1, PCE at gamma=2."""
    if M is None:
        M = softmax.shape[1]
    epsilon = torch.finfo(softmax.dtype).eps
    softmax = softmax.clone()
    sorted_softmax = torch.sort(softmax, descending=True, dim=1).values
    sorted_softmax = sorted_softmax[:, :M]
    if gamma == 1.0:
        return predictive_entropy(sorted_softmax)
    elif gamma == 2.0:
        return predictive_collision_entropy(sorted_softmax)
    else:
        ren_ent = (gamma / (1 - gamma)) * torch.log(
            torch.norm(sorted_softmax, p=gamma, dim=1) + epsilon
        )
        return -ren_ent


@validate_softmax
def tsallis_entropy(softmax: ArrayType, gamma: float = 0.1, M: int | None = None) -> ArrayType:
    """Tsallis entropy CSF. Reduces to PE at gamma=1."""
    if M is None:
        M = softmax.shape[1]
    softmax = softmax.clone()
    sorted_softmax = torch.sort(softmax, descending=True, dim=1).values
    sorted_softmax = sorted_softmax[:, :M]
    if gamma == 1.0:
        return predictive_entropy(sorted_softmax)
    else:
        tsa_ent = (1 / (gamma - 1)) * (1 - torch.sum(sorted_softmax ** gamma, dim=1))
        return -tsa_ent
