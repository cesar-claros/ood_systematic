"""Atomic base-detector CSFs operating on softmax/logits.

These are stateless functions returning a (N,)-shaped tensor of confidence
scores. They have no fit step (no hyperparameters to tune) and no save/load
state — `csf_fit.py` doesn't touch them; they appear in the confids dict in
`csf_pipeline.stats()` via direct call.

Convention: higher score = higher predicted confidence (more likely
in-distribution). Functions that naturally produce a "spread" score
(entropy, energy) are negated so the convention holds uniformly.
"""

import torch

from src.csfs._validators import ArrayType, validate_logit, validate_softmax


# MSR
def maximum_softmax_response(softmax: ArrayType) -> ArrayType:
    """Maximum softmax probability CSF (MSR)."""
    softmax = softmax.clone()
    msr = torch.max(softmax, dim=1).values
    return msr


# PE
@validate_softmax
def predictive_entropy(softmax: ArrayType) -> ArrayType:
    """Predictive entropy CSF (PE). Returns -H(p); higher = more confident."""
    softmax = softmax.clone()
    epsilon = torch.finfo(softmax.dtype).eps
    pred_ent = -torch.sum((softmax * (torch.log(softmax + epsilon))), dim=1)
    return -pred_ent


# MLS
@validate_logit
def maximum_logit_score(logit: ArrayType, temperature: float = 1) -> ArrayType:
    """Maximum logit score CSF (MLS)."""
    logit = logit.clone()
    mls = torch.max(logit / temperature, dim=1).values
    return mls


# MCS
@validate_logit
def maximum_cosine_similarity(similarity: ArrayType) -> ArrayType:
    """Maximum cosine similarity CSF (MCS)."""
    similarity = similarity.clone()
    mcs = torch.max(similarity, dim=1).values
    return mcs


# PCE
@validate_softmax
def predictive_collision_entropy(softmax: ArrayType) -> ArrayType:
    """Predictive collision entropy CSF (PCE). Returns -log(sum(p^2))."""
    epsilon = torch.finfo(softmax.dtype).eps
    softmax = softmax.clone()
    pred_col_ent = -torch.log(torch.sum(torch.square(softmax), dim=1) + epsilon)
    return -pred_col_ent


# GE
@validate_softmax
def guessing_entropy(softmax: ArrayType, M: int | None = None) -> ArrayType:
    """Guessing entropy CSF (GE). Sum of top-M sorted softmax weighted by rank."""
    if M is None:
        M = softmax.shape[1]
    softmax = softmax.clone()
    k_guesses = torch.tile(
        torch.tensor([i + 1 for i in range(softmax.shape[1])]),
        (softmax.shape[0], 1),
    )
    sorted_softmax = torch.sort(softmax, descending=True, dim=1).values
    sorted_softmax = sorted_softmax[:, :M]
    k_guesses = k_guesses[:, :M]
    guess_ent = torch.sum(k_guesses * sorted_softmax, dim=1)
    return -guess_ent


# Energy
@validate_logit
def energy(logit: ArrayType, temperature: float = 1) -> ArrayType:
    """Energy CSF: -T * logsumexp(logit / T). Returned negated (higher = more ID)."""
    energy_score = -temperature * torch.logsumexp(logit / temperature, dim=1)
    return -energy_score
