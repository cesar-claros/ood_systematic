"""Sensitivity of the headline predictor and the Mantel tests to the NC1 definition.

The paper uses the legacy NC1 (Papyan's NC1 divided by C under the legacy
pseudoinverse). This script reruns, with only the NC1 input changed:
  - the headline predictor: regime-free NC-only heads, 22-CSF labels,
    VGG13 -> ResNet18, the 66 (regime, side, baseline) comparisons and the
    size-matched random-set control of `newcsf_heads.py`;
  - the global Mantel tests and the partial Mantel controls of
    `mantel_partial_convergence.py` (four strata).
Variants: L-raw (paper), L-log (legacy, log10; transform control), C-raw
(corrected NC1 of `nc1_corrected_panel.py`), C-log (corrected, log10). ViT
corrected values are the approximate legacy x C (they enter the Mantel ViT
stratum only).

Decision rule (fixed before the run; documentation checklist, 2026-10-30):
  P1  joint-side mean regret below Always-CTM in all three regimes, with the
      mean test significant (Holm within cell) in all three.
  P2  at least 60 of 66 Wilcoxon comparisons significant (Holm within cell),
      none significantly worse.
  M1  all four Mantel strata p <= 0.05.
  Verdict from C-raw and C-log: ROBUST if P1, P2, M1 hold for both; NOT
  ROBUST if P1 fails for both; PARTLY ROBUST otherwise. L-log is a control.

Gates (abort before any variant number is computed):
  GS  L-raw reproduces the stored 22-CSF ResNet18 predictions exactly and the
      stored 65/0 and 65/0 comparison counts.
  GM  L-raw reproduces the stored global Mantel r and p of every stratum.
  GC  every VGG13, ResNet18, and ViT model gets a corrected NC1.

Run from `code/`. Outputs: outputs/nc1_sensitivity/{tests,summary,mantel}.csv,
outputs/55_nc1_sensitivity.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parents[1]
for path in (EVAL_DIR.parent / "ablations", EVAL_DIR, CODE_DIR):
    sys.path.insert(0, str(path))
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    standardize_per_architecture,
)
from calibration_regime_free import run_config_variant  # noqa: E402
from mantel_partial_convergence import STRATA, run_stratum  # noqa: E402
from nc_regime_analysis import load_nc_metrics  # noqa: E402
from newcsf_heads import (  # noqa: E402
    ALWAYS_BASELINES,
    NEW_CSFS,
    SPLIT,
    baseline_rows,
    labels_22,
    prepare_pool,
)
from regimefree_wilcoxon import (  # noqa: E402
    N_FLIPS,
    SEED,
    arm_rows,
    load_parquet,
    random_set_control,
    wilcoxon_cells,
)

CONFIG, VARIANT = "none_nr", "marginal"
VARIANTS = {"L-raw": (False, False), "L-log": (False, True),
            "C-raw": (True, False), "C-log": (True, True)}
COUNTS = ["reject_p", "reject_p_worse", "reject_p_mean", "reject_p_mean_worse"]
REGIMES = ["near", "mid", "far"]
STORED_COUNTS = {"reject_p": 65, "reject_p_worse": 0, "reject_p_mean": 65,
                 "reject_p_mean_worse": 0}
NC1_SOURCES = {"supercifar": "supercifar100"}
TOL = 1e-12


def nc1_values(nc1: pd.Series, corrected: pd.Series, use_corrected: bool,
               log10: bool) -> pd.Series:
    """The NC1 input of one variant.

    Raises:
        RuntimeError: If a model has no corrected value (gate GC).
    """
    values = nc1
    if use_corrected:
        values = corrected
        if values.isna().any():
            raise RuntimeError(f"GC failed: {int(values.isna().sum())} models without "
                               "a corrected NC1")
    return np.log10(values) if log10 else values


def predictor_variant(name: str, long_df: pd.DataFrame, corrected: pd.Series,
                      labels: pd.DataFrame, pool: dict, out_root: Path,
                      flips: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tests and random-set control of the NC-only heads for one NC1 variant."""
    use_corrected, log10 = VARIANTS[name]
    long_v = long_df.assign(var_collapse=nc1_values(
        long_df["var_collapse"], long_df["model_id"].map(corrected), use_corrected, log10))
    root = f"nc1_sensitivity/{name}"
    run_config_variant(standardize_per_architecture(long_v, NC_PRIMARY), labels, SPLIT,
                       out_root, CONFIG, VARIANT, ablation_root=root)
    preds = load_parquet(out_root / root / "track1" / SPLIT / f"{CONFIG}_{VARIANT}"
                         / "preds.parquet")
    if name == "L-raw":
        stored = load_parquet(out_root / "newcsf_heads" / "track1" / SPLIT
                              / f"{CONFIG}_{VARIANT}" / "preds.parquet")
        both = stored.merge(preds, on=["model_id", "regime", "csf"],
                            suffixes=("", "_new"))
        dev = float((both["p_competitive"] - both["p_competitive_new"]).abs().max())
        if len(both) != len(stored) or dev > TOL:
            raise RuntimeError(f"GS failed: max |dp| {dev:.3g}")
    rows = arm_rows(preds, pool["eval_test"], pool["oracle_test"])
    tests = wilcoxon_cells(rows, pool["baselines"], flips).assign(variant=name)
    control = random_set_control(rows, pool["eval_test"], flips).assign(variant=name)
    if name == "L-raw":
        counts = {c: int(tests[c].sum()) for c in COUNTS}
        if counts != STORED_COUNTS:
            raise RuntimeError(f"GS failed: counts {counts}")
        print("GS: L-raw reproduces the stored predictions and the 65/0, 65/0 counts")
    return tests, control


def mantel_variant(name: str, nc: pd.DataFrame, corrected: pd.Series,
                   stored: pd.DataFrame) -> pd.DataFrame:
    """Global and partial Mantel rows of every stratum for one NC1 variant.

    Raises:
        RuntimeError: If L-raw differs from the stored Mantel results (gate GM).
    """
    use_corrected, log10 = VARIANTS[name]
    key = (nc["architecture"] + "|" + nc["study"].replace({"vit": "modelvit"}) + "|"
           + nc["dataset"].replace(NC1_SOURCES) + "|" + nc["run"].astype(str) + "|"
           + nc["dropout"].map({"True": "1", "False": "0"}) + "|"
           + nc["reward"].map(lambda r: f"{r:g}"))
    in_paper = nc["architecture"].isin(["VGG13", "ViT"])
    value = nc1_values(nc.loc[in_paper, "var_collapse"],
                       key[in_paper].map(corrected), use_corrected, log10)
    nc_v = nc.copy()
    nc_v.loc[in_paper, "var_collapse"] = value
    rows = pd.DataFrame([r for backbone, study in STRATA
                         for r in run_stratum(backbone, study, nc_v)]).assign(variant=name)
    if name == "L-raw":
        both = stored.merge(rows, on=["backbone", "study", "control"],
                            suffixes=("", "_new"))
        dev = float((both[["r", "p"]].to_numpy() - both[["r_new", "p_new"]].to_numpy())
                    .__abs__().max())
        if len(both) != len(stored) or dev > TOL:
            raise RuntimeError(f"GM failed: {len(both)} rows, max |d| {dev:.3g}")
        print("GM: L-raw reproduces the stored global and partial Mantel results")
    return rows


def criteria(tests: pd.DataFrame, mantel: pd.DataFrame) -> pd.DataFrame:
    """P1, P2, and M1 per variant, with the joint-side numbers behind them."""
    recs = []
    for name in VARIANTS:
        t = tests[tests["variant"] == name]
        ctm = t[(t["side"] == "all") & (t["comparator"] == "Always-CTM")].set_index("regime")
        full = mantel[(mantel["variant"] == name)
                      & (mantel["control"] == "(none, replication)")]
        rec = {"variant": name,
               **{f"pred_{r}": ctm.loc[r, "pred_mean"] for r in REGIMES},
               **{f"ctm_{r}": ctm.loc[r, "baseline_mean"] for r in REGIMES},
               **{c: int(t[c].sum()) for c in COUNTS},
               **{f"mantel_r_{s}": v for s, v in zip(full["study"], full["r"])},
               "mantel_p_max": full["p"].max()}
        rec["P1"] = bool(all(ctm.loc[r, "pred_mean"] < ctm.loc[r, "baseline_mean"]
                             and ctm.loc[r, "reject_p_mean"] for r in REGIMES))
        rec["P2"] = rec["reject_p"] >= 60 and rec["reject_p_worse"] == 0
        rec["M1"] = bool(full["p"].max() <= 0.05)
        recs.append(rec)
    return pd.DataFrame(recs)


def verdict(table: pd.DataFrame) -> str:
    """The pre-stated verdict from the two corrected variants."""
    c = table[table["variant"].isin(["C-raw", "C-log"])].set_index("variant")
    if c[["P1", "P2", "M1"]].all(axis=None):
        return "ROBUST"
    if not c["P1"].any():
        return "NOT ROBUST"
    return "PARTLY ROBUST"


def main() -> None:
    """Gates, four NC1 variants, criteria, verdict, and report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    long_df, train_ids, test_ids, eval22, oracle22, _ = prepare_pool(out_root)
    corrected = (pd.read_csv(out_root / "nc1_corrected" / "nc1_corrected.csv")
                 .query("pool in ['VGG13', 'ResNet18', 'ViT']")
                 .set_index("model_id")["nc1_corrected"])
    eval_test = eval22[eval22["model_id"].isin(test_ids)]
    oracle_test = oracle22[oracle22["model_id"].isin(test_ids)]
    pool = {"eval_test": eval_test, "oracle_test": oracle_test,
            "baselines": baseline_rows(eval_test, oracle_test,
                                       eval22[eval22["model_id"].isin(train_ids)],
                                       ALWAYS_BASELINES + NEW_CSFS)}
    labels = labels_22(out_root)
    flips = np.random.default_rng(SEED).choice(np.array([-1.0, 1.0]),
                                               size=(N_FLIPS, len(test_ids)))
    nc = load_nc_metrics(str(CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv"))
    stored_mantel = pd.read_csv(CODE_DIR / "mantel_partial_outputs"
                                / "partial_mantel_summary.csv")
    tests, controls, mantel = [], [], []
    for name in VARIANTS:
        t, c = predictor_variant(name, long_df, corrected, labels, pool, out_root, flips)
        tests.append(t)
        controls.append(c)
        mantel.append(mantel_variant(name, nc, corrected, stored_mantel))
    tests, controls, mantel = map(pd.concat, (tests, controls, mantel))
    table = criteria(tests, mantel)
    result = verdict(table)

    csv_dir = out_root / "nc1_sensitivity"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(csv_dir / "tests.csv", index=False)
    table.to_csv(csv_dir / "summary.csv", index=False)
    mantel.to_csv(csv_dir / "mantel.csv", index=False)
    joint = controls[controls["side"] == "all"][["variant", "regime", "pred_mean",
                                                  "random_mean", "set_size", "p_mean_holm"]]
    partial = mantel[mantel["control"].isin(["(none, replication)", "iid_msr", "dataset"])]
    lines = ["# NC1 sensitivity of the headline predictor and the Mantel tests\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/nc1_sensitivity.py` (decision "
             "rule in the docstring and the camera-ready checklist, fixed before the "
             "run). Gates GS, GM, GC passed.\n\n",
             f"**Verdict: {result}.**\n\n## Criteria\n\n```\n",
             table.round(3).to_string(index=False), "\n```\n\n",
             "## Joint side: predictor vs size-matched random set\n\n```\n",
             joint.round(3).to_string(index=False), "\n```\n\n",
             "## Mantel: no control, ID performance, source identity\n\n```\n",
             partial.pivot_table(index=["backbone", "study", "control"], columns="variant",
                                 values="r")[list(VARIANTS)].round(3).to_string(),
             "\n```\n"]
    (out_root / "55_nc1_sensitivity.md").write_text("".join(lines))
    print(f"verdict {result}; wrote {out_root / '55_nc1_sensitivity.md'}")


if __name__ == "__main__":
    main()
