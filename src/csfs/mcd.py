"""Monte Carlo dropout aggregation helpers and MCD-specific scoring functions.

Two roles in this module:

  1. Generic MCD adapters. `mcd_function` and `mcd_expected_function` are
     operators on CSF scoring functions: they take a base-detector callable
     and apply MCD aggregation (mean-then-apply vs apply-per-sample-then-mean).
     They are not CSFs themselves.

  2. MCD-specific CSFs. `mcd_softmax_variance`, `mcd_watanabe_aic`, and
     `mcd_mutual_information` produce confidence scores that genuinely
     require an MCD distribution (shape (N, C, M)) and don't reduce to a
     base detector.
"""

import torch

from src.csfs._validators import ArrayType, validate_softmax_logit_distribution
from src.csfs.base_detectors import predictive_entropy


@validate_softmax_logit_distribution
def mcd_function(func, logit_softmax_distribution: ArrayType, **kwargs):
    """Reduce the MCD distribution to its mean, then call `func` on the result."""
    mean_logit_softmax_distribution = logit_softmax_distribution.mean(dim=2)
    if 'temperature' in kwargs.keys():
        temperature = kwargs['temperature']
        mcd_score = func(mean_logit_softmax_distribution, temperature=temperature)
    elif 'similarity' in kwargs.keys():
        similarity = kwargs['similarity']
        mcd_score = func(mean_logit_softmax_distribution, similarity=similarity)
    elif 'logits_eval' in kwargs.keys():
        logits_eval = kwargs['logits_eval']
        mean_logit_eval_distribution = logits_eval.mean(dim=2)
        mcd_score = func(mean_logit_softmax_distribution, logits_eval=mean_logit_eval_distribution)
    elif 'use_cuda' in kwargs.keys():
        use_cuda = kwargs['use_cuda']
        mcd_score = func(mean_logit_softmax_distribution, use_cuda=use_cuda)
    elif 'combine' in kwargs.keys():
        combine = kwargs['combine']
        mcd_score = func(mean_logit_softmax_distribution, combine=combine)
    elif ('predictions_eval' in kwargs.keys()) and ('X_back_projected_eval' in kwargs.keys()):
        X_back_projected_eval = kwargs['X_back_projected_eval']
        predictions_eval = kwargs['predictions_eval']
        mcd_score = func(
            mean_logit_softmax_distribution,
            predictions_eval=predictions_eval,
            X_back_projected_eval=X_back_projected_eval,
        )
    elif 'X_back_projected_eval' in kwargs.keys():
        X_back_projected_eval = kwargs['X_back_projected_eval']
        mcd_score = func(mean_logit_softmax_distribution, X_back_projected_eval=X_back_projected_eval)
    elif 'predictions_eval' in kwargs.keys():
        predictions_eval = kwargs['predictions_eval']
        mcd_score = func(mean_logit_softmax_distribution, predictions_eval=predictions_eval)
    elif ('use_cuda' in kwargs.keys()) and ('temperature' in kwargs.keys()):
        use_cuda = kwargs['use_cuda']
        temperature = kwargs['temperature']
        mcd_score = func(mean_logit_softmax_distribution, use_cuda=use_cuda, temperature=temperature)
    else:
        mcd_score = func(mean_logit_softmax_distribution)
    return mcd_score


@validate_softmax_logit_distribution
def mcd_expected_function(func, logit_softmax_distribution: ArrayType, **kwargs):
    """Apply `func` to each MCD sample independently, then average the resulting scores."""
    mcd_repetitions = logit_softmax_distribution.shape[2]
    if 'temperature' in kwargs.keys():
        temperature = kwargs['temperature']
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j], temperature=temperature)
            for j in range(mcd_repetitions)
        ])
    elif 'similarity' in kwargs.keys():
        similarity = kwargs['similarity']
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j], similarity=similarity)
            for j in range(mcd_repetitions)
        ])
    elif 'logits_eval' in kwargs.keys():
        logits_eval = kwargs['logits_eval']
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j], logits_eval=logits_eval[:, :, j])
            for j in range(mcd_repetitions)
        ])
    elif 'predictions_eval' in kwargs.keys():
        predictions_eval = kwargs['predictions_eval']
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j], predictions_eval=predictions_eval[:, j])
            for j in range(mcd_repetitions)
        ])
    elif 'use_cuda' in kwargs.keys():
        use_cuda = kwargs['use_cuda']
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j], use_cuda=use_cuda)
            for j in range(mcd_repetitions)
        ])
    else:
        mcd_dist = torch.vstack([
            func(logit_softmax_distribution[:, :, j])
            for j in range(mcd_repetitions)
        ])
    return mcd_dist.mean(dim=0)


@validate_softmax_logit_distribution
def mcd_softmax_variance(softmax_distribution: ArrayType) -> ArrayType:
    """Mean per-class variance of log-softmax across MCD samples (negated)."""
    var_log_softmax_distribution = (torch.log(softmax_distribution)).var(dim=2)
    return torch.mean(-var_log_softmax_distribution, dim=1)


@validate_softmax_logit_distribution
def mcd_watanabe_aic(softmax_distribution: ArrayType) -> ArrayType:
    """Watanabe AIC under MCD."""
    mean_softmax_distribution = softmax_distribution.mean(dim=2)
    var_log_softmax_distribution = (torch.log(softmax_distribution)).var(dim=2)
    waic = torch.mean(
        torch.log(mean_softmax_distribution) - var_log_softmax_distribution,
        dim=1,
    )
    return -waic


@validate_softmax_logit_distribution
def mcd_mutual_information(softmax_distribution: ArrayType) -> ArrayType:
    """Mutual information under MCD (PE of mean - mean of PE)."""
    mcd_pe = -mcd_function(predictive_entropy, softmax_distribution)
    mcd_ee = -mcd_expected_function(predictive_entropy, softmax_distribution)
    mutual_information = mcd_pe - mcd_ee
    return -mutual_information
