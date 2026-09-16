"""Known-answer and invariance checks before an adaptation training pilot."""

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from x8_pool_a.descriptors_v3 import (
    mean_span_projector,
    paired_linear_cka,
    residue_energy_projector,
)


@pytest.fixture
def rank_one_panel() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Three collinear class means with perpendicular within-class variation."""
    labels = np.repeat(np.arange(3, dtype=np.int64), 2)
    means = np.array([[-1.0, 0, 0], [0, 0, 0], [1, 0, 0]])
    noise = np.tile([[0.0, 1, 0], [0, -1, 0]], (3, 1))
    return means[labels] + noise, labels


def test_rank_deficiency_uses_actual_span(
    rank_one_panel: tuple[NDArray[np.float64], NDArray[np.int64]],
) -> None:
    """A C-1 QR truncation would add a spurious second direction here."""
    features, labels = rank_one_panel
    projector, _ = mean_span_projector(features, labels, 3)
    np.testing.assert_allclose(projector, np.diag([1, 0, 0]), atol=1e-14)
    assert residue_energy_projector(features, labels, 3) == pytest.approx(1.0)


def test_coincident_class_means_have_empty_span() -> None:
    """An empty class-mean span leaves all nonzero within-class energy outside."""
    features = np.tile([[0.0, 1, 0], [0, -1, 0]], (3, 1))
    labels = np.repeat(np.arange(3), 2)
    projector, _ = mean_span_projector(features, labels, 3)
    np.testing.assert_array_equal(projector, np.zeros((3, 3)))
    assert residue_energy_projector(features, labels, 3) == pytest.approx(1.0)


def test_zero_within_class_variance_is_undefined() -> None:
    """A zero denominator must not masquerade as maximal residue energy."""
    features = np.repeat([[-1.0, 0], [0, 0], [1, 0]], 2, axis=0)
    with pytest.raises(ValueError, match="zero within-class"):
        residue_energy_projector(features, np.repeat(np.arange(3), 2), 3)


def test_constant_decimal_features_do_not_create_spurious_variance() -> None:
    """Mean rounding with uneven class sizes must not invent a valid fraction."""
    features = np.full((17, 3), 0.1)
    labels = np.array([0] * 3 + [1] * 14)
    with pytest.raises(ValueError, match="zero within-class"):
        residue_energy_projector(features, labels, 2)
    ids = np.arange(len(features))
    with pytest.raises(ValueError, match="constant"):
        paired_linear_cka(features, features, ids, ids)


@pytest.mark.parametrize("direction, expected", [(0, 0.0), (2, 1.0)])
def test_full_rank_span_energy_endpoints(direction: int, expected: float) -> None:
    """Variance in the mean plane and its orthogonal axis have known fractions."""
    labels = np.repeat(np.arange(3), 2)
    means = np.array([[-1.0, 0, 0], [0, 1, 0], [1, -1, 0]])
    noise = np.zeros((6, 3))
    noise[:, direction] = np.tile([1, -1], 3)
    assert residue_energy_projector(means[labels] + noise, labels, 3) == pytest.approx(
        expected, abs=1e-14
    )


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_residue_scale_rotation_and_label_invariance(scale: float) -> None:
    """Equivalent coordinates and class names preserve the defined energy."""
    rng = np.random.default_rng(31)
    features = rng.normal(size=(36, 7))
    labels = np.repeat(np.arange(3), 12)
    rotation = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    expected = residue_energy_projector(features, labels, 3)
    actual = residue_energy_projector(
        scale * (features @ rotation), np.array([2, 0, 1])[labels], 3
    )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_projector_matches_trace_definition() -> None:
    """Compare the efficient energy calculation with the explicit covariance."""
    rng = np.random.default_rng(45)
    features = rng.normal(size=(40, 9))
    labels = np.repeat(np.arange(4), 10)
    projector, means = mean_span_projector(features, labels, 4)
    centered_means = means - means.mean(axis=0)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    np.testing.assert_allclose(projector, projector.T, atol=1e-12)
    np.testing.assert_allclose(centered_means @ projector, centered_means, atol=1e-12)
    within = features - means[labels]
    covariance = within.T @ within / len(within)
    expected = 1 - np.trace(projector @ covariance @ projector) / np.trace(covariance)
    assert residue_energy_projector(features, labels, 4) == pytest.approx(expected)


@pytest.mark.parametrize(
    "labels, classes",
    [
        ([0, 0, 2, 2], 3),
        ([0, 0, 1, 1], 0),
        ([0, -1, 1, 1], 2),
        ([0.0, 0, 1, 1], 2),
        ([0, 1], 2),
    ],
)
def test_invalid_labels_are_rejected(labels: ArrayLike, classes: int) -> None:
    """Missing classes and malformed labels cannot define this measurement."""
    with pytest.raises(ValueError):
        residue_energy_projector(np.ones((4, 3)), labels, classes)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_features_are_rejected(bad_value: float) -> None:
    """Do not emit a numerical descriptor from invalid features."""
    features = np.ones((4, 2))
    features[0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        mean_span_projector(features, [0, 0, 1, 1], 2)


def test_explicit_rank_tolerance() -> None:
    """The numerical rank convention is visible and controllable."""
    features = np.array([[-1.0, 1e-8], [0, -2e-8], [1, 1e-8]])
    labels = np.arange(3)
    full, _ = mean_span_projector(features, labels, 3, rank_rtol=1e-12)
    reduced, _ = mean_span_projector(features, labels, 3, rank_rtol=1e-6)
    assert np.trace(full) == pytest.approx(2)
    assert np.trace(reduced) == pytest.approx(1)
    with pytest.raises(ValueError, match="rank_rtol"):
        mean_span_projector(features, labels, 3, rank_rtol=-1)


def test_cka_joint_coordinate_and_scale_controls() -> None:
    """Paired shape similarity stays one under rotations, scales and offsets."""
    rng = np.random.default_rng(76)
    reference = rng.normal(size=(36, 5))
    rotation = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    ids = np.arange(len(reference))
    adapted = 3 * reference @ rotation + 8
    assert paired_linear_cka(reference, adapted, ids, ids) == pytest.approx(1.0)


def test_cka_uses_pairing_beyond_endpoint_covariances() -> None:
    """Identical endpoint covariances can conceal different sample pairings."""
    rng = np.random.default_rng(86)
    reference = rng.normal(size=(40, 4))
    permutation = rng.permutation(len(reference))
    adapted = reference[permutation]
    np.testing.assert_allclose(np.cov(reference.T), np.cov(adapted.T), atol=1e-12)
    ids = np.arange(len(reference))
    assert paired_linear_cka(reference, adapted, ids, ids) < 0.9
    with pytest.raises(ValueError, match="aligned"):
        paired_linear_cka(reference, adapted, ids, ids[permutation])


@pytest.mark.parametrize("samples, width", [(6, 11), (30, 4)])
def test_cka_matches_independent_sample_gram_formula(samples: int, width: int) -> None:
    """Both computational branches match centered sample-Gram alignment."""
    rng = np.random.default_rng(91)
    reference = rng.normal(size=(samples, width))
    adapted = rng.normal(size=(samples, width + 1))
    center = np.eye(samples) - np.ones((samples, samples)) / samples
    first = center @ reference @ reference.T @ center
    second = center @ adapted @ adapted.T @ center
    expected = np.sum(first * second) / (np.linalg.norm(first) * np.linalg.norm(second))
    ids = np.arange(samples)
    assert paired_linear_cka(reference, adapted, ids, ids) == pytest.approx(expected)


@pytest.mark.parametrize("case", ["duplicate", "few", "constant"])
def test_invalid_pair_measurements_fail_explicitly(case: str) -> None:
    """Undefined similarities and invalid pair IDs cannot silently pass."""
    reference = np.arange(12, dtype=float).reshape(4, 3)
    ids = np.arange(4)
    if case == "duplicate":
        ids[1] = ids[0]
    elif case == "few":
        reference, ids = reference[:2], ids[:2]
    else:
        reference[:] = 1
    with pytest.raises(ValueError):
        paired_linear_cka(reference, reference, ids, ids)


def test_joint_rotation_preserves_logit_and_mahalanobis_scores() -> None:
    """Large coordinate drift can leave the actual detector scores unchanged."""
    rng = np.random.default_rng(101)
    features = rng.normal(size=(40, 6))
    weights = rng.normal(size=(3, 6))
    rotation = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    rotated = features @ rotation
    np.testing.assert_allclose(features @ weights.T, rotated @ (weights @ rotation).T)
    covariance = np.cov(features.T) + 0.05 * np.eye(6)
    first = np.einsum("ni,ij,nj->n", features, np.linalg.inv(covariance), features)
    second = np.einsum(
        "ni,ij,nj->n",
        rotated,
        np.linalg.inv(rotation.T @ covariance @ rotation),
        rotated,
    )
    np.testing.assert_allclose(first, second)
    assert np.linalg.norm(rotated - features) / np.linalg.norm(features) > 0.5
