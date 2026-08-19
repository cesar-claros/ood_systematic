"""Tests for the Pilot 1 manipulation-report gate logic and Papyan mirror."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nc_csf_predictivity.interventions.manipulation_report import (
    ARM_ORDER,
    DOSE_RANK,
    evaluate,
    render,
)
from pilot0.geometry import fit_feature_model, papyan_metrics


def _records(sd_by_arm: dict[str, list[float]],
             acc_by_arm: dict[str, list[float]]) -> list[dict]:
    rng = np.random.default_rng(0)
    records = []
    for lam in ARM_ORDER:
        for run, (sd, acc) in enumerate(zip(sd_by_arm[lam],
                                            acc_by_arm[lam]), 1):
            rec = {"experiment": f"g/{lam}_run{run}", "lam": lam,
                   "run": run, "dose_rank": DOSE_RANK[lam],
                   "self_duality": sd, "val_acc": acc,
                   "train_acc": acc + 0.05, "head_residual_fraction": 0.05,
                   "logit_scale": 30.0}
            for coord in ("var_collapse", "equinorm_uc", "equinorm_wc",
                          "equiangular_uc", "equiangular_wc",
                          "max_equiangular_uc", "max_equiangular_wc"):
                rec[coord] = 0.05 + rng.normal(0, 0.001)
            records.append(rec)
    return records


def _theory_true() -> list[dict]:
    sd = {"-0.1": [0.30, 0.31, 0.29], "0.0": [0.080, 0.085, 0.075],
          "0.3": [0.040, 0.042, 0.038], "1.0": [0.020, 0.021, 0.019],
          "hard": [0.002, 0.002, 0.002]}
    acc = {lam: [0.70, 0.71, 0.69] for lam in ARM_ORDER}
    acc["hard"] = [0.695, 0.70, 0.69]
    return _records(sd, acc)


def test_gates_pass_on_theory_true_ordering():
    ev = evaluate(_theory_true())
    assert ev["M1_strength"] and ev["M1_monotone"] and ev["M1"]
    assert ev["spearman"] < -0.9
    assert ev["M2"]
    assert ev["exits_span"]["-0.1"] and ev["exits_span"]["hard"]


def test_monotone_gate_fails_on_scrambled_doses():
    recs = _theory_true()
    values = [r["self_duality"] for r in recs]
    rng = np.random.default_rng(3)
    rng.shuffle(values)
    for r, v in zip(recs, values):
        r["self_duality"] = v
    ev = evaluate(recs)
    assert not ev["M1_monotone"]
    assert not ev["M1"]


def test_m2_fails_on_accuracy_collapse():
    recs = _theory_true()
    for r in recs:
        if r["lam"] != "0.0":
            r["val_acc"] -= 0.10
    ev = evaluate(recs)
    assert not ev["M2"]


def test_selectivity_flags_comoving_coordinate():
    recs = _theory_true()
    for r in recs:
        if r["lam"] == "1.0":
            r["equinorm_uc"] += 0.05
    ev = evaluate(recs)
    assert ev["selectivity"]["1.0"]["equinorm_uc"]["ratio_to_target"] > 0.25
    assert ev["selectivity"]["0.3"]["equinorm_uc"]["ratio_to_target"] <= 0.25


def test_render_contains_gates():
    recs = _theory_true()
    text = render(recs, evaluate(recs))
    assert "M1 manipulation" in text and "PASS" in text
    assert "Selectivity" in text


def test_papyan_metrics_at_exact_collapse():
    rng = np.random.default_rng(1)
    n_classes, dim, radius, sigma = 10, 64, 1.0, 0.05
    m = (np.eye(n_classes) - 1.0 / n_classes) * np.sqrt(
        n_classes / (n_classes - 1)) * radius
    q = np.linalg.qr(rng.standard_normal((dim, n_classes)))[0]
    mu = m @ q.T
    y = rng.integers(0, n_classes, 30_000)
    h = mu[y] + rng.standard_normal((30_000, dim)) * sigma
    model = fit_feature_model(h, y, n_classes)
    metrics = papyan_metrics(3.0 * mu, model)
    assert metrics["equinorm_uc"] < 0.02
    assert metrics["equinorm_wc"] < 1e-9
    assert metrics["equiangular_uc"] < 0.02
    assert metrics["max_equiangular_uc"] < 0.02
    assert metrics["self_duality"] < 1e-3
    expected_nc1 = (n_classes - 1) ** 2 / (n_classes * (radius / sigma) ** 2)
    assert abs(metrics["var_collapse"] - expected_nc1) / expected_nc1 < 0.1
