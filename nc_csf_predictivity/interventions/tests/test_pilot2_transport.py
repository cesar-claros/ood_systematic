"""Tests for the Pilot 2 transport test (manifest section 8)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nc_csf_predictivity.interventions.pilot2_transport as p2
from nc_csf_predictivity.interventions.outcome_analysis import (
    MODE_TO_SET,
    STATS_TEMPLATE,
)

LAMS = ["0.0", "-0.1", "0.3", "1.0", "hard"]
RUNS = [1, 2, 3, 4]
SETS = sorted(MODE_TO_SET.values())
SCORES = ["MLS", "Maha", "CTM_head", "CTM_mean", "Energy", "MSR"]
BASE = {"MLS": 0.90, "Maha": 0.85, "CTM_head": 0.88, "CTM_mean": 0.86,
        "Energy": 0.895, "MSR": 0.80}
STATS_METHOD = {"CTM_head": "CTM"}
TRUE_SLOPE = 1.2


def pred_auroc(score: str, lam: str, set_name: str) -> float:
    i_set = SETS.index(set_name)
    j = SCORES.index(score)
    value = BASE[score] + 0.004 * ((i_set * (j + 1)) % 5 - 2)
    d = p2.DOSE.get(lam)
    if d is not None:
        value += {"MLS": 0.015, "Maha": -0.010}.get(score, 0.0) * d
    else:  # A2: joint mechanism, far outside the A1 range
        value += {"Maha": -0.25, "MLS": 0.03, "CTM_head": 0.02,
                  "Energy": 0.025}.get(score, 0.0)
    return value


def obs_loss(score: str, lam: str, run: int, set_name: str,
             coupled: bool = True) -> float:
    eps = 1e-4 * (((run * 7 + SCORES.index(score) * 3
                    + SETS.index(set_name)) % 9) - 4)
    if coupled:
        return 0.002 + TRUE_SLOPE * (1.0 - pred_auroc(score, lam,
                                                      set_name)) + eps
    # No-signal world: outcomes ignore the intervention entirely.
    return 0.002 + TRUE_SLOPE * (1.0 - pred_auroc(score, "0.0",
                                                  set_name)) + eps


def make_table(coupled: bool = True) -> pd.DataFrame:
    rows = []
    for lam in LAMS:
        for run in RUNS:
            for s in SETS:
                for score in SCORES:
                    if score == "MSR":
                        continue
                    loss = obs_loss(score, lam, run, s, coupled)
                    rows.append({
                        "name": f"x_run{run}_lam{lam}", "lam": lam,
                        "run": run, "set_name": s,
                        "method": STATS_METHOD.get(score, score),
                        "auroc_f": 1.0 - loss, "augrc": loss * 0.2})
    return pd.DataFrame(rows)


def make_stage2b() -> dict:
    out = {}
    for lam in LAMS:
        for run in RUNS:
            sets = {}
            for s in SETS:
                i_set = SETS.index(s)
                h = {"gamma": 0.5 + 0.01 * i_set,
                     "a": 0.6, "rho": 1.1 + (0.5 if lam == "hard" else 0.0),
                     "w_perp": 0.2, "top2_gap": 0.1}
                sets[s] = {"n_ood": 2000, "h_coords": h,
                           "preds": {"emp": {sc: pred_auroc(sc, lam, s)
                                             for sc in SCORES}}}
            out[(lam, run)] = {"lam": lam, "run": run, "sets": sets}
    return out


def make_geometry() -> dict:
    rng = np.random.default_rng(7)
    out = {}
    for lam in LAMS:
        for run in RUNS:
            d = p2.DOSE.get(lam, 2.0)
            rec = {"var_collapse": 0.12 - 0.01 * d,
                   "self_duality": 0.014 - 0.005 * d,
                   "equinorm_uc": 0.1, "equinorm_wc": 0.1,
                   "equiangular_uc": 0.05, "equiangular_wc": 0.05,
                   "max_equiangular_uc": 0.06, "max_equiangular_wc": 0.06,
                   "log_radius": 1.0, "log_sigma": -0.5,
                   "eig_max_over_mean": 20.0,
                   "head_residual_fraction": 0.05 + 0.02 * d,
                   "logit_scale": 25.0,
                   "class_mean_radius": np.e, "sigma_iso": 0.6}
            # Nuisances: pure noise, uncorrelated with the outcomes.
            rec.update({f: float(rng.normal()) for f in p2.Q_FIELDS})
            out[(lam, run)] = rec
    return out


def run_pipeline(coupled: bool = True) -> dict:
    table, s2b, geo = make_table(coupled), make_stage2b(), make_geometry()
    cells = p2.build_cells(table, s2b, "auroc_f")
    margin = p2.bootstrap_margin(cells, p2.TRAIN_LAMS, p2.HOLDOUT_LAMS,
                                 b=500)
    return p2.evaluate(cells, geo, s2b, margin, p2.TRAIN_LAMS,
                       p2.HOLDOUT_LAMS)


def test_theory_true_transport_passes():
    result = run_pipeline(coupled=True)
    assert result["verdict_margin"] == "PASS"
    assert result["sign"]["pass"] and result["sign"]["n_material"] > 0
    assert result["registered_verdict"] == "PASS"
    # The recovered calibration matches the construction.
    assert abs(result["fits"]["plugin"]["beta"] - TRUE_SLOPE) < 0.05
    assert result["mae"]["plugin"]["pooled"] < 0.002
    for comp in ("nuisance_pc", "dose", "cellmean", "nc_pc"):
        assert result["mae"][comp]["pooled"] > result["mae"]["plugin"]["pooled"]
    # A2 sits off-support in the registered coordinates.
    assert result["support"]["n_off"] == result["n_holdout_cells"]


def test_no_signal_world_does_not_pass():
    result = run_pipeline(coupled=False)
    assert result["registered_verdict"] != "PASS"


def test_dose_arm_degenerates_to_cell_mean_on_holdout():
    table, s2b, geo = make_table(), make_stage2b(), make_geometry()
    cells = p2.build_cells(table, s2b, "auroc_f")
    preds, _ = p2.arm_predictions(cells, geo, p2.TRAIN_LAMS)
    hold = preds[preds.lam == "hard"]
    assert np.allclose(hold["pred_dose"], hold["pred_cellmean"])
    train = preds[preds.lam != "hard"]
    assert not np.allclose(train["pred_dose"], train["pred_cellmean"])


def test_bootstrap_margin_deterministic():
    cells = p2.build_cells(make_table(), make_stage2b(), "auroc_f")
    m1 = p2.bootstrap_margin(cells, p2.TRAIN_LAMS, p2.HOLDOUT_LAMS, b=200)
    m2 = p2.bootstrap_margin(cells, p2.TRAIN_LAMS, p2.HOLDOUT_LAMS, b=200)
    assert m1 == m2 and m1 > 0


def test_pc1_handles_constant_columns():
    train = np.column_stack([np.ones(6), np.arange(6.0)])
    full = np.column_stack([np.ones(8), np.arange(8.0)])
    scores = p2.pc1_scores(train, full)
    assert scores.shape == (8,)
    assert np.all(np.diff(scores) > 0)  # monotone in the informative column


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    stats_root = tmp_path / "stats"
    s2b_dir = tmp_path / "stage2b"
    geo_dir = tmp_path / "geometry"
    s2b_dir.mkdir()
    geo_dir.mkdir()
    table = make_table()
    for name, sub in table.groupby("name"):
        exp = stats_root / str(name).replace(
            "x_", "etfreg_bbvgg13_do0_") / "analysis"
        exp.mkdir(parents=True)
        for mode, set_name in MODE_TO_SET.items():
            block = sub[sub.set_name == set_name]
            df = pd.DataFrame(
                {"AUGRC": block["augrc"].values * 1000.0,
                 "AURC": 50.0, "AUROC_f": block["auroc_f"].values,
                 "FPR@95TPR": 0.4, "ECE": 0.1, "MCE": 0.2,
                 "AP_ferr": 0.9, "AP_fsuc": 0.5},
                index=block["method"].values)
            df.to_csv(exp / STATS_TEMPLATE.format(mode=mode))
    for (lam, run), rec in make_stage2b().items():
        (s2b_dir / f"etfreg_bbvgg13_do0_run{run}_lam{lam}.json").write_text(
            json.dumps(rec))
    for (lam, run), rec in make_geometry().items():
        rec = dict(rec, lam=lam, run=run)
        rec.pop("log_radius"), rec.pop("log_sigma")  # loader derives these
        (geo_dir /
         f"etfreg_bbvgg13_do0_run{run}_lam{lam}__last.json").write_text(
            json.dumps(rec))
    return stats_root, s2b_dir, geo_dir


def test_cli_roundtrip_and_margin_ordering(tmp_path, monkeypatch):
    stats_root, s2b_dir, geo_dir = _write_fixture(tmp_path)
    margin_file = tmp_path / "pilot2_margin.json"
    out = tmp_path / "pilot2_report.md"
    common = ["pilot2_transport.py", "--stats_root", str(stats_root),
              "--stage2b_dir", str(s2b_dir), "--geometry_dir", str(geo_dir),
              "--margin_file", str(margin_file), "--out", str(out)]

    # full before margin must refuse (ordering enforcement).
    monkeypatch.setattr(sys, "argv", common + ["--stage", "full"])
    with pytest.raises(SystemExit):
        p2.main()

    monkeypatch.setattr(p2, "BOOT_B", 300)
    monkeypatch.setattr(sys, "argv", common + ["--stage", "margin"])
    p2.main()
    frozen = json.loads(margin_file.read_text())
    assert set(frozen["margins"]) == {"auroc_f", "augrc"}
    assert "Deviation note" in frozen["note"]

    # Re-freezing without --overwrite_margin must refuse.
    with pytest.raises(SystemExit):
        p2.main()

    monkeypatch.setattr(sys, "argv", common + ["--stage", "full"])
    p2.main()
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["auroc_f"]["registered_verdict"] == "PASS"
    assert report["augrc"]["verdict_margin"] == "PASS"
    text = out.read_text()
    assert "Registered Pilot 2 verdict: PASS" in text
    assert "AUGRC (secondary)" in text
