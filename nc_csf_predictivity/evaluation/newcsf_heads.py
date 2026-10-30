"""22-CSF predictor: per-CSF heads with Mahalanobis++ and NCI (VGG13 -> ResNet18).

Extends the regime-free predictor of `ablations/calibration_regime_free.py`
from the paper's 20-CSF roster to 22 CSFs (MahaPP and BO-tuned NCI) and
re-evaluates the VGG13 -> ResNet18 transfer.

  1. AUGRC table: the harmonized 20-CSF long table plus MahaPP and NCI rows
     from the BO-era new-CSF aggregate (code/new_csfs_pilot_summary_long.csv).
  2. Labels: per-config top cliques over the 22-CSF roster
     (outputs/newcsf_cliques/perconfig_cliques_bo.parquet, written by
     benchmark_newcsf_cliques.py).
  3. Heads: NC only and NC + source, marginal protocol, VGG13 pool only.
  4. Evaluation on the paper's 56 ResNet18 models: 22-CSF oracle per side,
     the paper's baselines plus Always-MahaPP and Always-NCI, the Wilcoxon
     and mean-difference tests and the size-matched random-set control of
     regimefree_wilcoxon.py, and an anticipation check for the two new CSFs
     (does the predicted probability rank the rows where the CSF is among
     the three best of 22?).

Gates (abort before any 22-CSF number is computed):
  GA  the aggregate's base-arm AUGRC for the 20 legacy CSFs equals the
      harmonized table on every row of the VGG13 and ResNet18 pools.
  GB  the oracle builder reproduces oracle.parquet on the 20-CSF roster.
  GC  the baseline builder reproduces the stored clique_bc baselines.
Label agreement with the paper's 20-CSF cliques is reported, not enforced
(the BO rescore shifts a few borderline ties).

Outputs:
  outputs/newcsf_heads/track1/xarch/<config>_marginal/{preds,coefficients}.parquet
  outputs/newcsf_heads/{tests,aggregate,baseline_aggregate,random_set_control,
                        anticipation}.csv
  outputs/49_newcsf_heads.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "ablations"))
sys.path.insert(0, str(EVAL_DIR))
from baselines import (  # noqa: E402
    ALWAYS_BASELINES,
    aggregate,
    compute_baselines_per_row,
)
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    SIDES,
    add_model_id,
    filter_to_side,
    standardize_per_architecture,
)
from calibration_regime_free import run_config_variant  # noqa: E402
from regimefree_wilcoxon import (  # noqa: E402
    N_FLIPS,
    SEED,
    aggregate_cells,
    arm_rows,
    cell_summary,
    load_parquet,
    random_set_control,
    wilcoxon_cells,
)

SPLIT = "xarch"
NEW_CSFS = ["MahaPP", "NCI"]
CONFIGS = ["none_nr", "source_nr"]
ABLATION_ROOT = "newcsf_heads"
AGGREGATE = EVAL_DIR.parents[1] / "new_csfs_pilot_summary_long.csv"
MODE_TO_EVAL = {
    "ood_nsncs_isun": "isun", "ood_nsncs_lsun_cropped": "lsun cropped",
    "ood_nsncs_lsun_resize": "lsun resize", "ood_nsncs_places365": "places365",
    "ood_nsncs_svhn": "svhn", "ood_nsncs_textures": "textures",
    "ood_nsncs_ti": "tinyimagenet", "ood_sncs_c10": "cifar10",
    "ood_sncs_c100": "cifar100",
}
EXPERIMENT = (r"^(?P<source>[\w-]+?)_paper_sweep/(?P<paradigm>confidnet|devries|dg)"
              r"_bb(?P<arch>vgg13|resnet18)_do(?P<do>[01])_run(?P<run>\d+)"
              r"_rew(?P<rew>[\d.]+)$")
ARCH = {"vgg13": "VGG13", "resnet18": "ResNet18"}
SOURCE_NAMES = {"supercifar": "supercifar100", "tiny-imagenet-200": "tinyimagenet"}
LABEL_CSF_NAMES = {"KPCA RecError": "KPCA RecError global",
                   "PCA RecError": "PCA RecError global"}
KEYS = ["model_id", "eval_dataset", "regime"]
TOL = 1e-6
TOP_K = 3
N_PERM = 10_000


def aggregate_long(csfs: set[str]) -> pd.DataFrame:
    """Base-arm AUGRC rows of the new-CSF aggregate, keyed like the long table."""
    agg = pd.read_csv(AGGREGATE, usecols=["experiment", "arm", "csf", "mode", "AUGRC"])
    agg = agg[(agg["arm"] == "base") & agg["mode"].isin(MODE_TO_EVAL)].copy()
    agg["csf"] = agg["csf"].str.replace("_RecError_global", " RecError global")
    agg = agg[agg["csf"].isin(csfs)]
    parts = agg["experiment"].str.extract(EXPERIMENT)
    keep = parts["arch"].notna()
    agg, parts = agg[keep], parts[keep]
    agg["model_id"] = (parts["arch"].map(ARCH) + "|" + parts["paradigm"] + "|"
                       + parts["source"].replace(SOURCE_NAMES) + "|" + parts["run"]
                       + "|" + parts["do"] + "|"
                       + parts["rew"].astype(float).map(lambda r: f"{r:g}"))
    agg["eval_dataset"] = agg["mode"].map(MODE_TO_EVAL)
    return (agg[["model_id", "eval_dataset", "csf", "AUGRC"]]
            .rename(columns={"AUGRC": "raw_augrc"}))


def gate_aggregate(eval20: pd.DataFrame, agg20: pd.DataFrame) -> None:
    """GA: the aggregate reproduces the harmonized AUGRC on every pool row.

    Raises:
        RuntimeError: On missing rows or any AUGRC deviation above TOL.
    """
    both = eval20.merge(agg20, on=["model_id", "eval_dataset", "csf"], how="left",
                        suffixes=("", "_agg"))
    missing = int(both["raw_augrc_agg"].isna().sum())
    dev = float((both["raw_augrc"] - both["raw_augrc_agg"]).abs().max())
    if missing or dev > TOL or len(both) != len(eval20):
        raise RuntimeError(f"GA failed: {missing} rows missing, max |dAUGRC| "
                           f"{dev:.3g}, rows {len(both)} vs {len(eval20)}")
    print(f"GA: {len(both):,} legacy rows reproduced, max |dAUGRC| {dev:.1e}")


def oracle_table(eval_long: pd.DataFrame) -> pd.DataFrame:
    """Best and worst CSF (and AUGRC) per row and side, in the oracle.parquet layout."""
    out = eval_long[KEYS].drop_duplicates()
    for side in SIDES:
        suffix = "" if side == "all" else f"_{side}"
        ranked = filter_to_side(eval_long, side).sort_values("raw_augrc")
        groups = ranked.groupby(KEYS)[["csf", "raw_augrc"]]
        for which, pick in (("oracle", groups.first()), ("worst", groups.last())):
            out = out.merge(pick.rename(columns={
                "csf": f"{which}_csf{suffix}",
                "raw_augrc": f"{which}_augrc{suffix}"}), left_on=KEYS, right_index=True)
    return out


def gate_oracle(oracle20: pd.DataFrame, stored: pd.DataFrame) -> None:
    """GB: the oracle builder reproduces oracle.parquet on the 20-CSF roster.

    Raises:
        RuntimeError: On a row-set mismatch or any AUGRC deviation above TOL.
    """
    cols = [f"{w}_augrc{s}" for w in ("oracle", "worst") for s in ("", "_head", "_feature")]
    both = stored.merge(oracle20, on=KEYS, how="outer", suffixes=("", "_new"),
                        indicator=True)
    dev = max(float((both[c] - both[f"{c}_new"]).abs().max()) for c in cols)
    if (both["_merge"] != "both").any() or dev > TOL:
        raise RuntimeError(f"GB failed: unmatched {int((both['_merge'] != 'both').sum())}, "
                           f"max |d| {dev:.3g}")
    print(f"GB: {len(both):,} oracle rows reproduced, max |d| {dev:.1e}")


def baseline_rows(eval_test: pd.DataFrame, oracle_test: pd.DataFrame,
                  eval_train: pd.DataFrame, always: list[str]) -> pd.DataFrame:
    """Per-row baseline regret in the layout `wilcoxon_cells` expects."""
    rows = compute_baselines_per_row(eval_test, oracle_test, eval_train, "model_id",
                                     sorted(eval_test["csf"].unique()), always=always)
    return rows.rename(columns={"comparator_name": "comparator"})


def gate_baselines(rows20: pd.DataFrame, stored: pd.DataFrame) -> None:
    """GC: the baseline builder reproduces the stored clique_bc baselines.

    Raises:
        RuntimeError: On a row-set mismatch or any regret deviation above TOL.
    """
    keys = ["model_id", "eval_dataset", "regime", "side", "comparator"]
    both = stored.merge(rows20, on=keys, how="outer", suffixes=("", "_new"),
                        indicator=True)
    dev = float((both["regret_raw"] - both["regret_raw_new"]).abs().max())
    if (both["_merge"] != "both").any() or dev > TOL:
        raise RuntimeError(f"GC failed: unmatched {int((both['_merge'] != 'both').sum())}, "
                           f"max |d| {dev:.3g}")
    print(f"GC: {len(both):,} baseline rows reproduced, max |d| {dev:.1e}")


def labels_22(out_root: Path) -> pd.DataFrame:
    """Per-config 22-CSF top-clique labels, named like the harmonized table."""
    p = load_parquet(out_root / "newcsf_cliques" / "perconfig_cliques_bo.parquet")
    p = p.assign(source=p["source"].replace(SOURCE_NAMES),
                 csf=p["csf"].replace(LABEL_CSF_NAMES),
                 label=p["in_top_clique"].astype(int))
    return p[["paradigm", "source", "dropout", "reward", "regime", "csf", "label"]]


def label_agreement(new: pd.DataFrame, legacy: pd.DataFrame) -> str:
    """Agreement of the 22-CSF labels with the paper's cliques on the legacy CSFs."""
    keys = ["paradigm", "source", "dropout", "reward", "regime", "csf"]
    ood = ["near", "mid", "far"]
    both = (legacy[legacy["regime"].isin(ood)]
            .merge(new, on=keys, suffixes=("_paper", "_new")))
    same = both["label_paper"] == both["label_new"]
    cells = both.assign(same=same).groupby(keys[:-1])["same"].all()
    return (f"{same.mean():.4f} of {len(both):,} (cell, CSF) labels agree; "
            f"{int(cells.sum())} of {len(cells)} cells have identical cliques on "
            f"the 20 legacy CSFs")


def anticipation(preds: pd.DataFrame, eval_test: pd.DataFrame,
                 rng: np.random.Generator) -> pd.DataFrame:
    """Does p_m rank the ResNet18 rows where CSF m is among the TOP_K best of 22?

    AUROC of the model-level predicted probability against the row-level
    target, per regime and pooled; the p-value permutes probabilities across
    models (rows of one model share their probability).
    """
    ranked = eval_test.assign(rank=eval_test.groupby(["model_id", "eval_dataset"])
                              ["raw_augrc"].rank(method="first"))
    out = []
    for csf in NEW_CSFS:
        p_model = (preds[preds["csf"] == csf].groupby("model_id")["p_competitive"]
                   .first())
        rows = ranked[ranked["csf"] == csf].assign(
            target=lambda d: (d["rank"] <= TOP_K).astype(int),
            p=lambda d: d["model_id"].map(p_model))
        for regime, cell in [("pooled", rows), *rows.groupby("regime")]:
            if cell["target"].nunique() < 2:
                continue
            models = cell["model_id"].unique()
            observed = roc_auc_score(cell["target"], cell["p"])
            perm = np.array([roc_auc_score(cell["target"], cell["model_id"].map(
                dict(zip(models, rng.permutation(p_model.reindex(models).to_numpy())))))
                for _ in range(N_PERM)])
            out.append({"csf": csf, "regime": regime, "rows": len(cell),
                        "target_rate": cell["target"].mean(),
                        "shortlist_rate": float((p_model >= 0.5).mean()),
                        "auroc": observed,
                        "p_perm": (np.sum(perm >= observed) + 1) / (N_PERM + 1)})
    return pd.DataFrame(out)


def report(gates: list[str], agreement: str, label_counts: str,
           tests: pd.DataFrame, controls: pd.DataFrame, antic: pd.DataFrame,
           aggregates: pd.DataFrame, reference: pd.DataFrame, out_path: Path) -> None:
    """Write the markdown summary."""
    lines = ["# 22-CSF predictor (Mahalanobis++ and NCI heads), VGG13 -> ResNet18\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/newcsf_heads.py`\n",
             "**Protocol:** regime-free marginal heads (NC only, NC + source) on the "
             "VGG13 pool with per-config 22-CSF top-clique labels; evaluation on the "
             "paper's 56 ResNet18 models with a 22-CSF oracle per side. Baselines: "
             f"Always-X for {', '.join(ALWAYS_BASELINES + NEW_CSFS)}, Oracle-on-train, "
             "Random-CSF (all over 22 CSFs). Tests as in 48_regimefree_wilcoxon.md "
             f"(Wilcoxon and model-clustered sign-flip mean test, {N_FLIPS:,} draws, "
             f"seed {SEED}, Holm within each cell).\n\n",
             "## Gates\n\n" + "".join(f"- {g}\n" for g in gates) + "\n",
             f"## Labels\n\n{agreement}.\n\n{label_counts}\n\n",
             "## Joint-side mean regret, 20 vs 22 CSFs\n\n```\n"
             + reference.to_string(index=False) + "\n```\n\n"]
    for config, cell in tests.groupby("arm", sort=False):
        counts = {c: int(cell[c].sum()) for c in
                  ("reject_p", "reject_p_worse", "reject_p_mean", "reject_p_mean_worse")}
        lines.append(f"## {config}\n\nWilcoxon: better {counts['reject_p']}, worse "
                     f"{counts['reject_p_worse']} of {len(cell)}. Mean test: better "
                     f"{counts['reject_p_mean']}, worse {counts['reject_p_mean_worse']} "
                     f"of {len(cell)}.\n\n```\n" + cell_summary(cell).to_string(index=False)
                     + "\n```\n\nShortlist size and empty rate:\n\n```\n"
                     + aggregates[aggregates["arm"] == config]
                     [["regime", "side", "set_size_mean", "empty_set_share"]]
                     .round(3).to_string(index=False) + "\n```\n\n")
    lines.append("## Size-matched random-set control\n\n```\n"
                 + controls.round(4).to_string(index=False) + "\n```\n\n")
    lines.append(f"## Anticipation of the new CSFs (target: CSF in the top {TOP_K} of 22 "
                 "on the row)\n\n```\n" + antic.round(4).to_string(index=False)
                 + "\n```\n")
    out_path.write_text("".join(lines))


def prepare_pool(out_root: Path) -> tuple[pd.DataFrame, set[str], set[str],
                                          pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the pool, run gates GA/GB/GC, and add the new-CSF rows.

    Returns:
        (long_df, train_ids, test_ids, eval22, oracle22, gate messages).

    Raises:
        RuntimeError: If a gate fails or the new-CSF rows are incomplete.
    """
    long_df = add_model_id(load_parquet(
        out_root / "track1" / "dataset" / "long_harmonized.parquet"))
    split = load_parquet(out_root / "splits" / f"{SPLIT}.parquet")
    train_ids = set(split.loc[split["role"] == "train", "model_id"])
    test_ids = set(split.loc[split["role"] == "test", "model_id"])
    pool = train_ids | test_ids
    eval20 = (long_df[(long_df["regime"] != "test") & long_df["model_id"].isin(pool)]
              [KEYS + ["csf", "augrc"]].rename(columns={"augrc": "raw_augrc"}))

    legacy_csfs = set(eval20["csf"])
    agg = aggregate_long(legacy_csfs | set(NEW_CSFS))
    gates = []
    gate_aggregate(eval20, agg[agg["csf"].isin(legacy_csfs)])
    gates.append("GA passed: aggregate reproduces the 20-CSF AUGRC table exactly")

    oracle_stored = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                              / "oracle.parquet"))
    oracle_stored = oracle_stored[(oracle_stored["regime"] != "test")
                                  & oracle_stored["model_id"].isin(pool)]
    oracle20 = oracle_table(eval20)
    gate_oracle(oracle20, oracle_stored)
    gates.append("GB passed: oracle builder reproduces oracle.parquet")

    stored_baselines = load_parquet(out_root / "clique_bc" / "track1" / SPLIT / "source"
                                    / "baselines" / "per_row.parquet")
    rows20 = baseline_rows(eval20[eval20["model_id"].isin(test_ids)],
                           oracle20[oracle20["model_id"].isin(test_ids)],
                           eval20[eval20["model_id"].isin(train_ids)], ALWAYS_BASELINES)
    gate_baselines(rows20, stored_baselines)
    gates.append("GC passed: baseline builder reproduces the stored clique_bc baselines")

    new_rows = (agg[agg["csf"].isin(NEW_CSFS)]
                .merge(eval20[KEYS].drop_duplicates(), on=["model_id", "eval_dataset"]))
    expected = len(eval20[KEYS].drop_duplicates()) * len(NEW_CSFS)
    if len(new_rows) != expected:
        raise RuntimeError(f"new-CSF coverage {len(new_rows)} of {expected} rows")
    eval22 = pd.concat([eval20, new_rows[KEYS + ["csf", "raw_augrc"]]], ignore_index=True)
    oracle22 = oracle_table(eval22)
    gates.append(f"New-CSF coverage complete: {len(new_rows):,} MahaPP/NCI rows")
    return long_df, train_ids, test_ids, eval22, oracle22, gates


def main() -> None:
    """Gates, 22-CSF heads, ResNet18 evaluation, and report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    out_root = Path(parser.parse_args().out_root)
    long_df, train_ids, test_ids, eval22, oracle22, gates = prepare_pool(out_root)

    labels = labels_22(out_root)
    legacy = load_parquet(out_root / "track1" / "cliques" / "cliques.parquet")
    legacy = legacy.assign(label=legacy["in_top_clique"].astype(int))
    agreement = label_agreement(labels, legacy)
    ood_labels = labels[labels["regime"].isin(["near", "mid", "far"])]
    label_counts = ("Positive (cell, regime) labels for the new CSFs on VGG13: "
                    + ", ".join(f"{c} {int(ood_labels.loc[ood_labels.csf == c, 'label'].sum())}"
                                f" of {int((ood_labels.csf == c).sum())}"
                                for c in NEW_CSFS) + ".")

    std_long = standardize_per_architecture(long_df, NC_PRIMARY)
    eval_test = eval22[eval22["model_id"].isin(test_ids)]
    oracle_test = oracle22[oracle22["model_id"].isin(test_ids)]
    baselines22 = baseline_rows(eval_test, oracle_test,
                                eval22[eval22["model_id"].isin(train_ids)],
                                ALWAYS_BASELINES + NEW_CSFS)
    flips = np.random.default_rng(SEED).choice(np.array([-1.0, 1.0]),
                                               size=(N_FLIPS, len(test_ids)))
    rng = np.random.default_rng(SEED)
    tests, aggs, controls, antic = [], [], [], []
    for config in CONFIGS:
        run_config_variant(std_long, labels, SPLIT, out_root, config, "marginal",
                           ablation_root=ABLATION_ROOT)
        preds = load_parquet(out_root / ABLATION_ROOT / "track1" / SPLIT
                             / f"{config}_marginal" / "preds.parquet")
        arm = f"{config}_marginal"
        rows = arm_rows(preds, eval_test, oracle_test)
        tests.append(wilcoxon_cells(rows, baselines22, flips).assign(arm=arm))
        aggs.append(aggregate_cells(rows).assign(arm=arm))
        controls.append(random_set_control(rows, eval_test, flips).assign(arm=arm))
        antic.append(anticipation(preds, eval_test, rng).assign(arm=arm))
    tests, aggs = pd.concat(tests), pd.concat(aggs)
    controls, antic = pd.concat(controls), pd.concat(antic)

    ref20 = pd.read_csv(out_root / "regimefree_wilcoxon" / f"{SPLIT}_aggregate.csv")
    both = pd.concat([ref20.assign(roster=20), aggs.assign(roster=22)])
    both = both[(both["side"] == "all")
                & both["arm"].isin([f"{c}_marginal" for c in CONFIGS])]
    reference = (both.pivot_table(index=["arm", "roster"], columns="regime",
                                  values="set_regret_imputed_mean")
                 [["near", "mid", "far"]].round(2).reset_index())

    csv_dir = out_root / ABLATION_ROOT
    baseline_agg = aggregate(baselines22.rename(columns={"comparator": "comparator_name"}))
    for name, frame in (("tests", tests), ("aggregate", aggs),
                        ("baseline_aggregate", baseline_agg),
                        ("random_set_control", controls), ("anticipation", antic)):
        frame.to_csv(csv_dir / f"{name}.csv", index=False)
    report(gates, agreement, label_counts, tests, controls, antic, aggs, reference,
           out_root / "49_newcsf_heads.md")
    print(f"wrote {out_root / '49_newcsf_heads.md'}")


if __name__ == "__main__":
    main()
