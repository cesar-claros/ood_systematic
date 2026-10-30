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

Added after the first run: K-log, log10 of kappa = C NC1 / (C - 1)^2 with the
corrected NC1. On a simplex ETF with within-class covariance sigma^2 I and
class-mean norm R, kappa = (sigma / R)^2 at any C. K-log does not enter the
verdict above; it is judged on its own by the same P1, P2, and M1.

Gates (abort before any variant number is computed):
  GS  L-raw reproduces the stored 22-CSF ResNet18 predictions exactly and the
      stored 65/0 and 65/0 comparison counts.
  GM  L-raw reproduces the stored global Mantel r and p of every stratum.
  GC  every VGG13, ResNet18, and ViT model gets a corrected NC1.
  GV  the four earlier variants reproduce the stored summary.csv (checked
      before the outputs are overwritten).

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
VARIANTS = {"L-raw": ("legacy", False), "L-log": ("legacy", True),
            "C-raw": ("papyan", False), "C-log": ("papyan", True),
            "K-log": ("kappa", True)}
EARLIER_VARIANTS = ["L-raw", "L-log", "C-raw", "C-log"]
COUNTS = ["reject_p", "reject_p_worse", "reject_p_mean", "reject_p_mean_worse"]
REGIMES = ["near", "mid", "far"]
STORED_COUNTS = {"reject_p": 65, "reject_p_worse": 0, "reject_p_mean": 65,
                 "reject_p_mean_worse": 0}
NC1_SOURCES = {"supercifar": "supercifar100"}
TOL = 1e-12


def nc1_values(legacy: pd.Series, model_ids: pd.Series, name: str,
               inputs: dict[str, pd.Series]) -> pd.Series:
    """The NC1 input of one variant.

    Args:
        legacy: Stored NC1 values.
        model_ids: Model id of each value.
        name: Variant name of VARIANTS.
        inputs: Per-model values of the non-legacy NC1 definitions.

    Raises:
        RuntimeError: If a model has no corrected value (gate GC).
    """
    source, log10 = VARIANTS[name]
    values = legacy
    if source != "legacy":
        values = model_ids.map(inputs[source])
        if values.isna().any():
            raise RuntimeError(f"GC failed: {int(values.isna().sum())} models without "
                               "a corrected NC1")
    return np.log10(values) if log10 else values


def predictor_variant(name: str, long_df: pd.DataFrame, inputs: dict[str, pd.Series],
                      labels: pd.DataFrame, pool: dict, out_root: Path,
                      flips: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tests and random-set control of the NC-only heads for one NC1 variant."""
    long_v = long_df.assign(var_collapse=nc1_values(
        long_df["var_collapse"], long_df["model_id"], name, inputs))
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


def mantel_variant(name: str, nc: pd.DataFrame, inputs: dict[str, pd.Series],
                   stored: pd.DataFrame) -> pd.DataFrame:
    """Global and partial Mantel rows of every stratum for one NC1 variant.

    Raises:
        RuntimeError: If L-raw differs from the stored Mantel results (gate GM).
    """
    key = (nc["architecture"] + "|" + nc["study"].replace({"vit": "modelvit"}) + "|"
           + nc["dataset"].replace(NC1_SOURCES) + "|" + nc["run"].astype(str) + "|"
           + nc["dropout"].map({"True": "1", "False": "0"}) + "|"
           + nc["reward"].map(lambda r: f"{r:g}"))
    in_paper = nc["architecture"].isin(["VGG13", "ViT"])
    value = nc1_values(nc.loc[in_paper, "var_collapse"], key[in_paper], name, inputs)
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


def gate_earlier(table: pd.DataFrame, path: Path) -> None:
    """GV: the four earlier variants reproduce the stored summary.

    Raises:
        RuntimeError: If a stored value differs.
    """
    stored = pd.read_csv(path).set_index("variant").loc[EARLIER_VARIANTS]
    new = table.set_index("variant").loc[EARLIER_VARIANTS, stored.columns]
    numeric = stored.select_dtypes("number").columns
    dev = float((stored[numeric] - new[numeric]).abs().max().max())
    flags = ["P1", "P2", "M1"]
    if dev > 1e-9 or not stored[flags].astype(bool).equals(new[flags].astype(bool)):
        raise RuntimeError(f"GV failed: max |d| {dev:.3g}")
    print(f"GV: the four earlier variants reproduce the stored summary (max |d| {dev:.1e})")


def main() -> None:
    """Gates, five NC1 variants, criteria, verdict, and report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    long_df, train_ids, test_ids, eval22, oracle22, _ = prepare_pool(out_root)
    panel = (pd.read_csv(out_root / "nc1_corrected" / "nc1_corrected.csv")
             .query("pool in ['VGG13', 'ResNet18', 'ViT']").set_index("model_id"))
    inputs = {"papyan": panel["nc1_corrected"],
              "kappa": (panel["nc1_corrected"] * panel["n_classes"]
                        / (panel["n_classes"] - 1) ** 2)}
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
        t, c = predictor_variant(name, long_df, inputs, labels, pool, out_root, flips)
        tests.append(t)
        controls.append(c)
        mantel.append(mantel_variant(name, nc, inputs, stored_mantel))
    tests, controls, mantel = map(pd.concat, (tests, controls, mantel))
    table = criteria(tests, mantel)
    result = verdict(table)

    csv_dir = out_root / "nc1_sensitivity"
    gate_earlier(table, csv_dir / "summary.csv")
    kappa = table.set_index("variant").loc["K-log"]
    tests.to_csv(csv_dir / "tests.csv", index=False)
    table.to_csv(csv_dir / "summary.csv", index=False)
    mantel.to_csv(csv_dir / "mantel.csv", index=False)
    joint = controls[controls["side"] == "all"][["variant", "regime", "pred_mean",
                                                  "random_mean", "set_size", "p_mean_holm"]]
    partial = mantel[mantel["control"].isin(["(none, replication)", "iid_msr", "dataset"])]
    sides = (tests[tests["comparator"] == "Always-CTM"]
             .pivot_table(index=["side", "regime"], columns="variant", values="pred_mean")
             [list(VARIANTS)].assign(always_ctm=tests[tests["comparator"] == "Always-CTM"]
                                     .groupby(["side", "regime"])["baseline_mean"].first()))
    lines = ["# NC1 sensitivity of the headline predictor and the Mantel tests\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/nc1_sensitivity.py` (decision "
             "rule in the docstring and the camera-ready checklist, fixed before the "
             "run). Gates GS, GM, GC, GV passed.\n\n",
             f"**Verdict from C-raw and C-log: {result}.**\n\n",
             f"**K-log, judged on its own: P1 {kappa['P1']}, P2 {kappa['P2']}, "
             f"M1 {kappa['M1']}.**\n\n## Criteria\n\n```\n",
             table.round(3).to_string(index=False), "\n```\n\n",
             "## Joint side: predictor vs size-matched random set\n\n```\n",
             joint.round(3).to_string(index=False), "\n```\n\n",
             "## Mean regret vs Always-CTM per side\n\n```\n",
             sides.round(3).to_string(), "\n```\n\n",
             "## Mantel r: no control, ID performance, source identity\n\n```\n",
             partial.pivot_table(index=["backbone", "study", "control"], columns="variant",
                                 values="r")[list(VARIANTS)].round(3).to_string(),
             "\n```\n\n## Mantel p: same rows\n\n```\n",
             partial.pivot_table(index=["backbone", "study", "control"], columns="variant",
                                 values="p")[list(VARIANTS)].round(4).to_string(),
             "\n```\n"]
    (out_root / "55_nc1_sensitivity.md").write_text("".join(lines))
    print(f"verdict {result}; wrote {out_root / '55_nc1_sensitivity.md'}")


if __name__ == "__main__":
    main()
