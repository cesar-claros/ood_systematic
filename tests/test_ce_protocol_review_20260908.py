"""Outcome-free arithmetic checks and counterexamples for the CE protocol review.

Run from code/: .venv/bin/python -m pytest -q -s \
    tests/test_ce_protocol_review_20260908.py
No checkpoint, geometry cache, or target-outcome artifact is read.
"""

from itertools import combinations_with_replacement
from math import factorial
from pathlib import Path

import numpy as np
import orjson
from numpy.testing import assert_allclose
from scipy.stats import rankdata

from mc_phase_audit import BASE, build_config_model
from pilot0.scores import auroc, ctm
from src.rc_stats_utils import generalized_risk_stats


def test_augrc_identity_with_ties() -> None:
    """Check the repository's trapezoidal AUGRC against pairwise failure AUROC."""
    rng = np.random.default_rng(20260908)
    for sample_count in (7, 20, 101):
        residuals = np.arange(sample_count) % 2
        prevalence = residuals.mean()
        for _ in range(20):
            scores = rng.integers(0, 4, sample_count).astype(float)
            curve = generalized_risk_stats(confids=scores, residuals=residuals)
            area = -np.trapz(curve["risks"], curve["coverages"])
            failure_auc = auroc(scores[residuals == 0], scores[residuals == 1])
            expected = prevalence**2 / 2 + prevalence * (1 - prevalence) * (
                1 - failure_auc
            )
            assert_allclose(area, expected, atol=1e-14)


def test_nc1_low_rank_cutoff_and_etf_dictionary() -> None:
    """Verify the squared-singular-value cutoff and isotropic ETF identity."""
    mean_matrix = np.diag([1.0, 0.002, 0.0005])
    covariance = np.diag([1.0, 2.0, 3.0])
    class_count = 3
    between = mean_matrix @ mean_matrix.T / class_count
    direct = (
        np.trace(covariance @ np.linalg.pinv(between, rcond=1e-6, hermitian=True))
        / class_count
    )
    left, singular, _ = np.linalg.svd(mean_matrix, full_matrices=False)
    retained = singular > np.sqrt(1e-6) * singular[0]
    projected = np.diag(left[:, retained].T @ covariance @ left[:, retained])
    assert retained.sum() == 2
    assert_allclose(direct, np.sum(projected / singular[retained] ** 2))

    class_count, dimension, radius, sigma = 4, 6, 3.0, 0.7
    means = np.zeros((class_count, dimension))
    means[:, :class_count] = (
        radius
        * np.sqrt(class_count / (class_count - 1))
        * (np.eye(class_count) - 1 / class_count)
    )
    between = means.T @ means / class_count
    within = sigma**2 * np.eye(dimension)
    nc1 = np.trace(within @ np.linalg.pinv(between, hermitian=True)) / class_count
    assert_allclose(nc1, (class_count - 1) ** 2 / (class_count * (radius / sigma) ** 2))


def test_smoothed_shift_gate_accepts_single_shift_signal() -> None:
    """Demonstrate that fitted B checks can pass with one informative shift."""
    ranks = np.linspace(-0.4, 0.4, 5)
    severity = np.array([-1.5, -0.5, 0.5, 1.5]) / np.sqrt(1.25)
    actual_slopes = np.array([0.0, 0.0, -0.16, 0.0])
    outcomes = ranks[:, None] * actual_slopes[None, :]
    design = np.column_stack([np.repeat(ranks, 4), (ranks[:, None] * severity).ravel()])
    beta, interaction = np.linalg.lstsq(design, outcomes.ravel(), rcond=None)[0]
    fitted_b = -0.8 * (beta + interaction * severity)
    assert_allclose(fitted_b, [0.0128, 0.0256, 0.0384, 0.0512], atol=1e-14)
    assert np.count_nonzero(fitted_b > 0) == 4
    assert np.all(fitted_b > -0.01)
    assert np.count_nonzero(actual_slopes) == 1
    assert_allclose(outcomes[:, [0, 1, 3]], 0)


def test_leave_source_gate_accepts_single_source_signal() -> None:
    """Demonstrate that three positive source deletions permit one-source signal."""
    source_contrasts = np.array([0.08, 0.0, 0.0, 0.0])
    deletions = (source_contrasts.sum() - source_contrasts) / 3
    assert np.count_nonzero(deletions > 0) == 3
    assert np.all(deletions > -0.01)
    assert np.count_nonzero(source_contrasts > 0) == 1


def test_ctm_translation_reverses_winner_with_same_centered_coordinates() -> None:
    """Show the centered dictionary omits a translation relevant to raw CTM."""
    prototypes = np.array([[-1.0, 0.0], [1.0, 0.0]])
    id_features = np.array([[-1.0, 0.1]])
    ood_features = np.array([[1.0, 0.2]])
    translation = np.array([1.1, 0.0])
    weights = prototypes.copy()
    bias = np.zeros(2)
    translated_bias = bias - weights @ translation
    before = auroc(ctm(id_features, prototypes), ctm(ood_features, prototypes))
    after = auroc(
        ctm(id_features + translation, prototypes + translation),
        ctm(ood_features + translation, prototypes + translation),
    )
    assert before == 1.0
    assert after == 0.0
    assert_allclose(
        id_features @ weights.T + bias,
        (id_features + translation) @ weights.T + translated_bias,
    )
    assert_allclose(translated_bias.mean(), bias.mean())
    assert_allclose(
        prototypes - prototypes.mean(0),
        prototypes + translation - (prototypes + translation).mean(0),
    )


def test_failure_auc_depends_on_failure_composition() -> None:
    """Removing pi(1-pi) does not remove the ID-error mixture dependence."""
    correct_ood_gap = 0.10
    correct_error_gap = -0.40
    failure_ood_weights = np.array([0.90, 0.60])
    gaps = (
        failure_ood_weights * correct_ood_gap
        + (1 - failure_ood_weights) * correct_error_gap
    )
    assert_allclose(gaps, [0.05, -0.10])


def test_frozen_generator_does_not_recover_named_coordinates() -> None:
    """Expose intended-versus-realized alignment, tilt, and scale mismatches."""
    config = dict(BASE, s=10.0, theta_deg=60.0, eta_std=0.0, a=0.001, ga=0.001)
    model = build_config_model(10, 32, config, seed=0)
    directions = model["means"] / np.linalg.norm(model["means"], axis=1, keepdims=True)
    ood_direction = model["m_ood"] / np.linalg.norm(model["m_ood"])
    realized_alignment = (directions @ ood_direction).max()
    assert realized_alignment > 10 * config["a"]
    assert_allclose(directions[0] @ ood_direction, config["a"], atol=1e-14)

    head_directions = model["w"] / np.linalg.norm(model["w"], axis=1, keepdims=True)
    angle = np.deg2rad(config["theta_deg"])
    perturbations = (head_directions - np.cos(angle) * directions) / np.sin(angle)
    assert np.abs(perturbations @ directions.T).max() > 0.1
    mean_target_logit = np.einsum("cd,cd->c", model["w"], model["means"]).mean()
    assert_allclose(mean_target_logit, config["logit_target"] * np.cos(angle))


def test_log_tail_magnitude_is_not_auc_gap_magnitude() -> None:
    """A larger error-log-ratio can accompany a much smaller AUROC gap."""
    log_errors = np.array([[-1.0, -2.0], [-100.0, -110.0]])
    margins = log_errors[:, 0] - log_errors[:, 1]
    auc_gaps = np.exp(log_errors[:, 0]) - np.exp(log_errors[:, 1])
    assert margins[1] > margins[0]
    assert auc_gaps[1] < auc_gaps[0]


def test_five_family_bootstrap_null_calibration() -> None:
    """Measure exact-bootstrap null behavior for a simple five-family mean.

    Enumerates all 126 count vectors with their multinomial probabilities.
    Uses 20,000 independent Gaussian null datasets and writes synthetic-only
    audit results to the existing ignored regression_outputs directory.
    This is an inference diagnostic, not a simulation of the complete CE gate.
    """
    sample_count = 5
    counts = np.array(
        [
            np.bincount(draw, minlength=sample_count)
            for draw in combinations_with_replacement(range(sample_count), sample_count)
        ]
    )
    masses = np.array(
        [
            factorial(sample_count)
            / np.prod([factorial(int(value)) for value in count])
            / sample_count**sample_count
            for count in counts
        ]
    )
    assert_allclose(masses.sum(), 1.0)
    rng = np.random.default_rng(20260908)
    trials = 20000
    positive_lower = 0
    centered_rejections = 0
    for _ in range(trials // 500):
        observations = rng.normal(size=(500, sample_count))
        estimates = observations.mean(axis=1)
        bootstrap = observations @ (counts / sample_count).T
        order = np.argsort(bootstrap, axis=1)
        sorted_bootstrap = np.take_along_axis(bootstrap, order, axis=1)
        cumulative = np.cumsum(masses[order], axis=1)
        low_indices = np.argmax(cumulative >= 0.025, axis=1)
        lower = sorted_bootstrap[np.arange(len(observations)), low_indices]
        positive_lower += np.count_nonzero(lower > 0)
        p_values = (
            (bootstrap - estimates[:, None] >= estimates[:, None]) * masses
        ).sum(axis=1)
        centered_rejections += np.count_nonzero(p_values < 0.05)
    output = {
        "simulation_seed": 20260908,
        "trials": trials,
        "independent_families": sample_count,
        "bootstrap_count_vectors": len(counts),
        "bootstrap_ordered_draws": sample_count**sample_count,
        "positive_95pct_lower_bound_rate": positive_lower / trials,
        "nominal_positive_tail_rate": 0.025,
        "centered_one_sided_p_lt_005_rate": centered_rejections / trials,
        "nominal_one_sided_rate": 0.05,
    }
    output_directory = Path("regression_outputs/ce_protocol_review_20260908")
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "synthetic_inference.json").write_bytes(
        orjson.dumps(output, option=orjson.OPT_INDENT_2)
    )
    assert positive_lower / trials > 0.04
    assert centered_rejections / trials > 0.07


def test_reranking_bootstrap_changes_the_estimator() -> None:
    """Document a protocol ambiguity: carrying ranks differs from recomputing."""
    frozen_ranks = np.linspace(-0.4, 0.4, 5)
    draw = np.array([0, 0, 1, 2, 4])
    reranked = (rankdata(frozen_ranks[draw]) - 0.5) / 5 - 0.5
    assert not np.allclose(frozen_ranks[draw], reranked)
