"""Known-answer and invariance tests for x8_pool_a/retention.py."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "x8_pool_a"))
import retention as rt  # noqa: E402

C, D, N = 5, 16, 400


def panel(seed=0, mean_scale=3.0):
    rng = np.random.default_rng(seed); y = rng.integers(0, C, N); M = rng.standard_normal((C, D)) * mean_scale
    return M[y] + rng.standard_normal((N, D)), y, np.arange(N)


def test_identity_is_one():
    H, y, ids = panel(); r = rt.retention_record(H, y, ids, H, y, ids, C)
    assert all(abs(v - 1.0) < 1e-9 for v in r.values())


def test_joint_rotation_scaling_translation_leaves_cka_measures_unchanged_but_moves_subspace_overlap():
    H, y, ids = panel(); rng = np.random.default_rng(1); Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    Ht = 2.5 * (H @ Q) + 7.0
    r = rt.retention_record(H, y, ids, Ht, y, ids, C)
    assert abs(r["cm_cka_ref"] - 1.0) < 1e-9 and abs(r["cc_cka_ref"] - 1.0) < 1e-9
    assert r["bc_overlap_ref"] < 0.9   # coordinate-dependent by design


def test_class_mean_relation_change_is_detected_with_within_class_fixed():
    H, y, ids = panel(); M = rt.centred_class_means(H, y, C); rng = np.random.default_rng(2)
    Mnew = rng.standard_normal((C, D)) * 3.0; Ht = H - M[y] + (Mnew - Mnew.mean(0))[y]   # replace class means, keep residuals
    r = rt.retention_record(H, y, ids, Ht, y, ids, C)
    assert r["cm_cka_ref"] < 0.9 and abs(r["cc_cka_ref"] - 1.0) < 1e-6   # random 5-class Gram matrices share some structure by chance; 0.9 still separates from identity


def test_within_class_change_is_detected_with_class_means_fixed():
    H, y, ids = panel(); M = rt.centred_class_means(H, y, C); rng = np.random.default_rng(3)
    R = rng.standard_normal((N, D)); R -= np.stack([R[y == c].mean(0) for c in range(C)])[y]   # fresh residuals with exact zero class means
    Ht = M[y] + H.mean(0) + R
    r = rt.retention_record(H, y, ids, Ht, y, ids, C)
    assert abs(r["cm_cka_ref"] - 1.0) < 1e-6 and r["cc_cka_ref"] < 0.2


def test_rank_deficient_means_and_bad_inputs():
    rng = np.random.default_rng(4); y = np.repeat(np.arange(3), 50); means = np.array([[-1., 0, 0], [0, 0, 0], [1., 0, 0]])
    noise = rng.standard_normal((150, 3)) * 0.1; noise -= np.stack([noise[y == c].mean(0) for c in range(3)])[y]
    H = means[y] + noise; ov = rt.between_class_subspace_overlap(H, y, H, y, 3); assert abs(ov - 1.0) < 1e-9
    with pytest.raises(ValueError): rt.class_centred_paired_cka(H, y, np.arange(150), H, y, np.arange(150)[::-1], 3)
    with pytest.raises(ValueError): rt.centred_class_means(H, np.zeros(150, dtype=int), 3)
