"""Test the hypothesis that CTM, CTMmean, CTMmeanOC relative performance depends on NC profile.

For each (backbone, source, paradigm, dropout, run, reward, ood_regime) cell, extract AUGRC
for CTM / CTMmean / CTMmeanOC and join with the per-configuration NC metric profile.
Compute pairwise performance deltas:
    - ctm_wins_over_mean   = AUGRC(CTMmean)   - AUGRC(CTM)      (positive -> CTM wins)
    - oc_wins_over_mean    = AUGRC(CTMmean)   - AUGRC(CTMmeanOC)(positive -> OC wins)
and Spearman-correlate each delta against the 8 Papyan NC metrics.

Outputs (under ood_eval_outputs/ctm_variants_hypothesis/):
    correlations_pooled.csv       # per (backbone, delta, nc_metric) Spearman + p
    correlations_by_regime.csv    # same broken down by regime
    paired_deltas.csv             # full joined table
    scatter_<backbone>.pdf/png    # 2 x 4 scatter grid
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "ood_eval_outputs" / "ctm_variants_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
BACKBONES = ["Conv", "ViT"]
MCD = "False"

NC_METRICS = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc", "self_duality",
]

REGIME_MAP = {"0": "test", "1": "near", "2": "mid", "3": "far"}
OOD_REGIMES = ["near", "mid", "far"]  # exclude "test" (ID)
BACKBONE_ARCH = {"Conv": "VGG13", "ViT": "ViT"}
DROPOUT_MAP = {"do0": False, "do1": True}

METHODS_OF_INTEREST = ["CTM", "CTMmean", "CTMmeanOC"]


def _parse_reward(rew: str) -> float | None:
    if pd.isna(rew) or rew in ("rewNone", "None"):
        return None
    m = re.match(r"rew([\d.]+)", str(rew))
    return float(m.group(1)) if m else None


PARADIGM_MAP = {"modelvit": "vit"}  # AUGRC "model" -> NC "study"


def load_augrc_long() -> pd.DataFrame:
    frames = []
    for backbone in BACKBONES:
        for source in SOURCES:
            fpath = REPO / "scores_risk" / f"scores_all_AUGRC_MCD-{MCD}_{backbone}_{source}_fix-config.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            df = df[df["methods"].isin(METHODS_OF_INTEREST)].copy()
            if df.empty:
                continue
            id_cols = ["model", "drop out", "methods", "reward", "run"]
            ood_cols = [c for c in df.columns if c not in id_cols]
            long = df.melt(id_vars=id_cols, value_vars=ood_cols,
                           var_name="dataset", value_name="augrc")
            long["backbone"] = backbone
            long["source"] = source
            frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"drop out": "dropout_tag", "model": "paradigm"})
    out["paradigm"] = out["paradigm"].replace(PARADIGM_MAP)
    out["reward_val"] = out["reward"].apply(_parse_reward)
    out["dropout_bool"] = out["dropout_tag"].map(DROPOUT_MAP)
    return out


def attach_regimes(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for source in SOURCES:
        clip_path = REPO / "clip_scores" / f"clip_distances_{source}.csv"
        if not clip_path.exists():
            continue
        clip = pd.read_csv(clip_path, header=[0, 1])
        clip.columns = clip.columns.droplevel(0)
        clip = clip.rename({"Unnamed: 0_level_1": "dataset",
                            "Unnamed: 5_level_1": "group"}, axis="columns")
        cmap = clip[["dataset", "group"]].copy()
        cmap["group"] = cmap["group"].apply(lambda x: str(int(x)) if pd.notna(x) else None)
        cmap["source"] = source
        pieces.append(cmap)
    regime_map = pd.concat(pieces, ignore_index=True)
    merged = df.merge(regime_map, on=["source", "dataset"], how="left")
    merged["regime"] = merged["group"].map(REGIME_MAP)
    return merged.dropna(subset=["regime"])


def load_nc() -> pd.DataFrame:
    df = pd.read_csv(REPO / "neural_collapse_metrics" / "nc_metrics.csv", index_col=0)
    keep = NC_METRICS + ["dataset", "architecture", "study", "dropout", "run", "reward"]
    df = df[keep].copy()
    df = df.rename(columns={"dataset": "source", "study": "paradigm",
                            "reward": "reward_val", "dropout": "dropout_bool"})
    df["backbone"] = df["architecture"].map({v: k for k, v in BACKBONE_ARCH.items()})
    df["run"] = df["run"].astype(int)
    df["dropout_bool"] = df["dropout_bool"].astype(bool)
    return df


def build_paired_deltas(augrc: pd.DataFrame, nc: pd.DataFrame) -> pd.DataFrame:
    key = ["backbone", "source", "paradigm", "dropout_bool", "run", "reward_val", "regime"]
    pivoted = augrc.pivot_table(
        index=key, columns="methods", values="augrc", aggfunc="first"
    ).reset_index()
    pivoted = pivoted.dropna(subset=METHODS_OF_INTEREST)
    pivoted["ctm_wins_over_mean"] = pivoted["CTMmean"] - pivoted["CTM"]
    pivoted["oc_wins_over_mean"] = pivoted["CTMmean"] - pivoted["CTMmeanOC"]

    nc_key = ["backbone", "source", "paradigm", "dropout_bool", "run", "reward_val"]
    merged = pivoted.merge(nc[nc_key + NC_METRICS], on=nc_key, how="inner")
    return merged


def correlate(df: pd.DataFrame, delta_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols):
        if isinstance(keys, tuple):
            kd = dict(zip(group_cols, keys))
        else:
            kd = {group_cols[0]: keys}
        for nc_metric in NC_METRICS:
            sub = g[[nc_metric, delta_col]].dropna()
            if len(sub) < 10:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(sub[nc_metric], sub[delta_col])
            rows.append({**kd, "delta": delta_col, "nc_metric": nc_metric,
                         "rho": rho, "p": p, "n": len(sub)})
    return pd.DataFrame(rows)


def scatter_panel(df: pd.DataFrame, backbone: str) -> None:
    sub = df[df["backbone"] == backbone].copy()
    if sub.empty:
        return
    facets = [
        ("var_collapse", "var. collapse (NC1)"),
        ("equiangular_uc", "equiang. $\\mu$"),
        ("equiangular_wc", "equiang. W"),
        ("self_duality", "self-duality"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for col_i, (metric, label) in enumerate(facets):
        for row_i, delta in enumerate(["ctm_wins_over_mean", "oc_wins_over_mean"]):
            ax = axes[row_i, col_i]
            for src, color in zip(SOURCES, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
                s = sub[sub["source"] == src]
                ax.scatter(s[metric], s[delta], alpha=0.5, s=18,
                           c=color, label=src, edgecolor="k", linewidth=0.2)
            ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
            rho, p = spearmanr(sub[metric], sub[delta], nan_policy="omit")
            ax.set_title(f"{label}\n$\\rho$={rho:+.2f}, p={p:.1e}", fontsize=9)
            if col_i == 0:
                yl = "CTM wins over CTMmean" if delta == "ctm_wins_over_mean" else "OC wins over CTMmean"
                ax.set_ylabel(f"$\\Delta$AUGRC\n({yl})")
            if row_i == 1:
                ax.set_xlabel(label)
            if col_i == 3 and row_i == 0:
                ax.legend(fontsize=7, loc="best")
            ax.grid(alpha=0.3)
    fig.suptitle(f"CTM variants vs NC profile — {backbone}", fontsize=12)
    plt.tight_layout()
    stem = OUT / f"scatter_{backbone}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    print("Loading AUGRC scores...")
    augrc = load_augrc_long()
    augrc = attach_regimes(augrc)
    print(f"  {len(augrc):,} AUGRC rows across {augrc['backbone'].nunique()} backbones, "
          f"{augrc['source'].nunique()} sources, {augrc['methods'].nunique()} methods")

    print("Loading NC metrics...")
    nc = load_nc()
    print(f"  {len(nc):,} NC configuration rows")

    print("Building paired deltas...")
    paired = build_paired_deltas(augrc, nc)
    print(f"  {len(paired):,} paired cells (after inner join with NC)")
    paired = paired[paired["regime"].isin(OOD_REGIMES)].copy()
    print(f"  {len(paired):,} OOD-only cells (dropped 'test' regime)")
    paired.to_csv(OUT / "paired_deltas.csv", index=False)

    print("Computing pooled correlations...")
    pool_rows = []
    for delta in ["ctm_wins_over_mean", "oc_wins_over_mean"]:
        pool_rows.append(correlate(paired, delta, ["backbone"]))
    pooled = pd.concat(pool_rows, ignore_index=True)
    pooled.to_csv(OUT / "correlations_pooled.csv", index=False)

    print("Computing per-regime correlations...")
    reg_rows = []
    for delta in ["ctm_wins_over_mean", "oc_wins_over_mean"]:
        reg_rows.append(correlate(paired, delta, ["backbone", "regime"]))
    by_regime = pd.concat(reg_rows, ignore_index=True)
    by_regime.to_csv(OUT / "correlations_by_regime.csv", index=False)

    print("Computing per-paradigm correlations (pooled regimes)...")
    par_rows = []
    for delta in ["ctm_wins_over_mean", "oc_wins_over_mean"]:
        par_rows.append(correlate(paired, delta, ["backbone", "paradigm"]))
    by_paradigm = pd.concat(par_rows, ignore_index=True)
    by_paradigm.to_csv(OUT / "correlations_by_paradigm.csv", index=False)

    print("Computing per-(paradigm, regime) correlations...")
    pr_rows = []
    for delta in ["ctm_wins_over_mean", "oc_wins_over_mean"]:
        pr_rows.append(correlate(paired, delta, ["backbone", "paradigm", "regime"]))
    by_par_reg = pd.concat(pr_rows, ignore_index=True)
    by_par_reg.to_csv(OUT / "correlations_by_paradigm_regime.csv", index=False)

    for backbone in BACKBONES:
        print(f"Plotting {backbone}...")
        scatter_panel(paired, backbone)

    print("\n=== Pooled Spearman (per backbone, all regimes) ===")
    wide = pooled.pivot_table(
        index=["backbone", "nc_metric"], columns="delta", values="rho"
    ).round(3)
    print(wide.to_string())

    print("\n=== Pair counts per backbone ===")
    print(paired.groupby("backbone").size().to_string())

    print(f"\nWrote outputs under {OUT}")


if __name__ == "__main__":
    main()
