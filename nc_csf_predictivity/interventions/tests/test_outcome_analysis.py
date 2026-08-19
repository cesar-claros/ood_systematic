"""Tests for the registered outcome analysis (E1-E5, X-a/X-f, Holm)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nc_csf_predictivity.interventions.outcome_analysis import (
    ARM_LAMS,
    MODE_TO_SET,
    PAIRS,
    STATS_TEMPLATE,
    load_long,
    run_analysis,
)

RUNS = [1, 2, 3, 4]
SETS = ["ood_a", "ood_b", "ood_c", "ood_d"]
METHODS = ["MSR", "MLS", "Energy", "CTM", "CTM_mean", "Maha", "fDBD",
           "PCA_RecError", "Residual"]
BASE = 0.80
SEED_EPS = {1: 0.0010, 2: -0.0010, 3: 0.0016, 4: -0.0016}
HEAD_SPREAD = {1: 0.020, 2: -0.020, 3: 0.030, 4: -0.030}

# Noiseless arm effects on AUROC_f: (lam, method) -> {set: delta_auroc}.
EFFECTS = {
    ("0.3", "MLS"): {"ood_a": -0.010, "ood_b": -0.010},
    ("1.0", "MLS"): {"ood_a": -0.015, "ood_b": -0.015},
    ("hard", "Maha"): {s: -0.050 for s in SETS},
    ("hard", "CTM_mean"): {s: -0.020 for s in SETS},
}


def _auroc(lam: str, run: int, set_name: str, method: str) -> float:
    value = BASE + SEED_EPS[run]
    value += EFFECTS.get((lam, method), {}).get(set_name, 0.0)
    if lam == "-0.1" and method in ("MSR", "MLS", "Energy", "CTM", "fDBD"):
        value += HEAD_SPREAD[run]
    return value


def _table() -> pd.DataFrame:
    rows = []
    for lam in ["0.0", *ARM_LAMS.values()]:
        for run in RUNS:
            for s in SETS:
                for m in METHODS:
                    a = _auroc(lam, run, s, m)
                    rows.append({"name": f"x_run{run}_lam{lam}",
                                 "lam": lam, "run": run, "set_name": s,
                                 "method": m, "auroc_f": a,
                                 "augrc": (1.0 - a) * 0.2})
    return pd.DataFrame(rows)


def _committed(flip: bool = False) -> dict:
    committed: dict = {}
    for endpoint, pair in PAIRS.items():
        committed[endpoint] = {}
        for label, lam in ARM_LAMS.items():
            cells = {}
            for s in SETS:
                delta = ((-EFFECTS.get((lam, pair[0]), {}).get(s, 0.0))
                         - (-EFFECTS.get((lam, pair[1]), {}).get(s, 0.0)))
                sign = int(np.sign(delta)) * (-1 if flip else 1)
                cells[s] = {"sign": sign if sign else 1,
                            "material": abs(delta) >= 0.004}
            committed[endpoint][label] = {"cells": cells}
    return committed


def test_theory_true_outcomes_agree_and_significant():
    result = run_analysis(_table(), _committed(), scale="auroc_f")
    for endpoint in ("E1", "E2", "E4"):
        assert result["pairs"][endpoint]["agreement_overall"] == 1.0
    assert result["pairs"]["E1"]["p_one_sided"] < 0.05
    assert result["holm"]["E1"] < 0.2
    # E5: the A1- head-side seed spread must be detected.
    assert result["E5"]["median_sd_ratio"] > 3.0
    assert result["E5"]["pooled_p_one_sided"] < 0.05
    # X-f: A2 has no inflated spread in this construction.
    assert result["X_f"]["median_sd_ratio"] < 2.0


def test_e3_nulls_and_violation_detection():
    result = run_analysis(_table(), _committed(), scale="auroc_f")
    # PCA_RecError / Residual are bit-identical to baseline per seed.
    for score in ("PCA_RecError", "Residual"):
        for label in ARM_LAMS:
            rec = result["E3"][score][label]
            assert rec["n_equivalent"] == rec["n_sets"]
    # Maha moves hugely under A2: TOST must NOT declare equivalence.
    assert result["E3"]["Maha"]["A2"]["n_equivalent"] == 0


def test_flipped_committed_signs_yield_zero_agreement():
    result = run_analysis(_table(), _committed(flip=True), scale="auroc_f")
    assert result["pairs"]["E1"]["agreement_overall"] == 0.0
    assert result["pairs"]["E1"]["p_one_sided"] > 0.5


def test_loader_roundtrip(tmp_path):
    exp = tmp_path / "etfreg_bbvgg13_do0_run1_lam0.0" / "analysis"
    exp.mkdir(parents=True)
    for mode in MODE_TO_SET:
        df = pd.DataFrame(
            {"AUGRC": [40.0, 55.0, 60.0], "AURC": [50.0, 60.0, 70.0],
             "AUROC_f": [0.9, 0.85, 0.8], "FPR@95TPR": [0.4, 0.5, 0.6],
             "ECE": [0.1, 0.1, 0.1], "MCE": [0.2, 0.2, 0.2],
             "AP_ferr": [0.9, 0.9, 0.9], "AP_fsuc": [0.5, 0.5, 0.5]},
            index=["MLS", "Maha", "PCA_RecError_global"])
        df.to_csv(exp / STATS_TEMPLATE.format(mode=mode))
    table = load_long(tmp_path)
    assert len(table) == len(MODE_TO_SET) * 3
    row = table[(table.method == "MLS")
                & (table.set_name == "ood_sncs")].iloc[0]
    assert row["lam"] == "0.0" and row["run"] == 1
    assert row["auroc_f"] == 0.9 and abs(row["augrc"] - 0.040) < 1e-12
    # PCA_RecError only exists as the global variant in the stats CSVs;
    # the loader canonicalizes it to the registered E3 name.
    assert "PCA_RecError" in set(table.method)
    assert "PCA_RecError_global" not in set(table.method)


def test_missing_null_score_reported_not_fatal():
    table = _table()
    table = table[table.method != "PCA_RecError"].reset_index(drop=True)
    result = run_analysis(table, _committed(), scale="auroc_f")
    assert result["E3"]["PCA_RecError"] == {"status": "missing"}
    # The other registered nulls and the pair endpoints are unaffected.
    assert result["E3"]["Residual"]["A1+"]["n_equivalent"] == 4
    assert result["pairs"]["E1"]["agreement_overall"] == 1.0


def test_loader_missing_mode_raises(tmp_path):
    exp = tmp_path / "etfreg_bbvgg13_do0_run1_lam0.0" / "analysis"
    exp.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_long(tmp_path)
