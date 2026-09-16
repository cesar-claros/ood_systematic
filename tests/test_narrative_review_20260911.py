"""Independent mathematical checks for the September 11 narrative review."""

import numpy as np


def test_failure_and_ood_auroc_can_reverse_detector_preference() -> None:
    """The AUGRC bridge does not identify all-ID-versus-OOD AUROC."""
    energy_id = np.array([2.0, 2.0, 2.0, 5.0])
    energy_ood = np.full(4, 3.0)
    cosine_id = np.array([4.0, 4.0, 4.0, 0.0])
    cosine_ood = np.array([3.0, 6.0, 6.0, 6.0])
    ood_auroc_energy = np.mean(energy_id[:, None] > energy_ood)
    ood_auroc_cosine = np.mean(cosine_id[:, None] > cosine_ood)
    energy_failures = np.concatenate([energy_id[-1:], energy_ood])
    cosine_failures = np.concatenate([cosine_id[-1:], cosine_ood])
    failure_auroc_energy = np.mean(energy_id[:3, None] > energy_failures)
    failure_auroc_cosine = np.mean(cosine_id[:3, None] > cosine_failures)
    assert ood_auroc_energy == 0.25 > ood_auroc_cosine == 0.1875
    assert failure_auroc_energy == 0.0 < failure_auroc_cosine == 0.4


def test_canonical_crossing_matches_entire_logit_distribution() -> None:
    """Canonical OOD and class-one logits have identical Gaussian laws."""
    class_count, dimension = 4, 32
    directions = np.zeros((class_count, dimension))
    directions[:, :class_count] = (np.eye(class_count) - 1.0 / class_count) * np.sqrt(
        class_count / (class_count - 1)
    )
    radius, alignment = 2.0, 0.7
    complement = np.eye(dimension)[class_count]
    ood_direction = alignment * directions[0] + np.sqrt(1 - alignment**2) * complement
    head = 3.0 * directions
    id_means = radius * directions
    ood_mean = radius / alignment * ood_direction
    np.testing.assert_allclose(head @ ood_mean, head @ id_means[0])
    id_logit_covariance = head @ head.T
    ood_logit_covariance = head @ np.eye(dimension) @ head.T
    np.testing.assert_allclose(id_logit_covariance, ood_logit_covariance)
    # The other ID classes permute the same means and exchangeable covariance.
    for class_mean in id_means:
        np.testing.assert_allclose(np.sort(head @ class_mean), np.sort(head @ ood_mean))


def test_rotated_head_covariance_requires_complement_gram() -> None:
    """Orthonormal head complements change off-diagonal logit correlation."""
    class_count, dimension = 3, 32
    directions = np.zeros((class_count, dimension))
    directions[:, :class_count] = (np.eye(class_count) - 1.0 / class_count) * np.sqrt(
        class_count / (class_count - 1)
    )
    complements = np.eye(dimension)[class_count : 2 * class_count]
    theta = np.pi / 3
    head = np.cos(theta) * directions + np.sin(theta) * complements
    actual = head @ head.T
    simplex_gram = directions @ directions.T
    expected = np.cos(theta) ** 2 * simplex_gram + np.sin(theta) ** 2 * np.eye(
        class_count
    )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual[0, 1], -0.125)
    np.testing.assert_allclose(simplex_gram[0, 1], -0.5)
    assert not np.allclose(actual, simplex_gram)


def test_translation_preserves_logits_only_with_transformed_bias() -> None:
    """Changing feature origin requires b_new = b + W @ origin."""
    rng = np.random.default_rng(911)
    head = rng.standard_normal((3, 8))
    bias = rng.standard_normal(3)
    features = rng.standard_normal((12, 8))
    origin = rng.standard_normal(8)
    original = features @ head.T + bias
    translated = (features - origin) @ head.T + bias + head @ origin
    np.testing.assert_allclose(translated, original, atol=1e-12)
    assert not np.allclose((features - origin) @ head.T + bias, original)
