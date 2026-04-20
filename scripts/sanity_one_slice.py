"""
Sanity check: extract a top clique for one slice — cifar10 × confidnet × do0 × regime=0 (ID).

We bypass load_all_scores because its filename parser assumes the paper's naming
convention without the _fix-config suffix. Here we load the _fix-config CSV directly,
melt to long format, attach the CLIP regime label, and run the
Friedman -> Conover-Holm -> Bron-Kerbosch pipeline for exactly one slice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils_stats import (
    friedman_blocked,
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    greedy_exclusive_layers,
    KNOWN_META_COLS,
    standardize_metric,
)

REPO = Path(__file__).resolve().parents[1]
SCORES_DIR = REPO / "scores_risk"
CLIP_DIR = REPO / "clip_scores"

SOURCE = "cifar10"
MODEL = "confidnet"
DROPOUT = "do0"
REGIME = "0"  # ID bucket
ALPHA = 0.05
METRICS = ["AUGRC", "AURC"]  # RC metric group


def load_fix_config_long(source: str, mcd: bool) -> pd.DataFrame:
    frames = []
    mcd_tag = "MCD-True" if mcd else "MCD-False"
    for metric in METRICS:
        path = SCORES_DIR / f"scores_all_{metric}_{mcd_tag}_Conv_{source}_fix-config.csv"
        df = pd.read_csv(path)
        ds_cols = [c for c in df.columns if c not in KNOWN_META_COLS]
        long = df.melt(
            id_vars=[c for c in df.columns if c not in ds_cols],
            value_vars=ds_cols,
            var_name="dataset",
            value_name="score",
        )
        long["metric"] = metric
        long["reward"] = long["reward"].astype(str).str.lower().map(lambda x: x.split("rew")[1]).astype(float)
        long["MCD"] = "1" if mcd else "0"
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["score_std"] = out.apply(lambda r: standardize_metric(r["metric"], r["score"]), axis=1)
    return out


def attach_regime(df: pd.DataFrame, source: str) -> pd.DataFrame:
    clip_path = CLIP_DIR / f"clip_distances_{source}.csv"
    clip = pd.read_csv(clip_path, header=[0, 1])
    clip.columns = clip.columns.droplevel(0)
    clip = clip.rename(columns={"Unnamed: 0_level_1": "dataset", "Unnamed: 5_level_1": "group"})
    clip["group"] = clip["group"].apply(lambda x: str(int(x)) if pd.notna(x) else x)
    out = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
    return out


def run_one_slice(df: pd.DataFrame, model: str, dropout: str, regime: str, reward: float) -> dict:
    sub = df[
        (df["model"] == model)
        & (df["drop out"] == dropout)
        & (df["group"] == regime)
        & (df["reward"] == reward)
    ].copy()
    if sub.empty:
        return {"error": f"empty slice: model={model}, drop_out={dropout}, group={regime}, reward={reward}"}

    # With (model, reward, drop_out, regime) all fixed, the block structure is
    # (dataset, metric, run).
    sub["block"] = sub[["dataset", "metric", "run"]].astype(str).agg("|".join, axis=1)

    stat, p, pivot = friedman_blocked(sub, entity_col="methods", block_col="block", value_col="score_std")
    info = {
        "slice": dict(source=SOURCE, model=model, drop_out=dropout, regime=regime, reward=reward),
        "n_blocks": pivot.shape[0],
        "n_methods": pivot.shape[1],
        "friedman_stat": float(stat) if stat is not None else None,
        "friedman_p": float(p) if p is not None else None,
    }
    if stat is None or np.isnan(stat):
        info["error"] = "Friedman returned NaN"
        return info

    ph = conover_posthoc_from_pivot(pivot)
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()
    cliques = maximal_cliques_from_pmatrix(ph, ALPHA)
    scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
    layers = greedy_exclusive_layers(scored)

    info["n_maximal_cliques"] = len(cliques)
    info["top_clique"] = sorted(layers[0]["members"]) if layers else []
    info["top_clique_size"] = len(info["top_clique"])
    info["top_clique_best_rank"] = float(layers[0]["best_rank"]) if layers else None
    info["top_clique_mean_rank"] = float(layers[0]["mean_rank"]) if layers else None
    info["layer2_members"] = sorted(layers[1]["members"]) if len(layers) > 1 else []
    info["avg_rank_head"] = avg_ranks.head(10).to_dict()
    return info


def main():
    df = load_fix_config_long(SOURCE, mcd=False)
    df = attach_regime(df, SOURCE)
    print(f"Loaded {len(df)} long-format rows from fix-config CSVs.")
    print(f"  models: {sorted(df['model'].unique())}")
    print(f"  dropouts: {sorted(df['drop out'].unique())}")
    print(f"  regimes: {sorted(df['group'].dropna().unique())}")
    print(f"  rewards: {sorted(df['reward'].unique())}")
    print(f"  datasets: {sorted(df['dataset'].unique())}")

    # Three contrast slices at regime=0 (ID): confidnet@do0, DG@r=2.2@do0, DG@r=10.0@do0.
    # Also show cross-regime (near-OOD, regime=1) for confidnet to exercise multi-dataset blocks.
    slices = [
        (MODEL, "do0", "0", 2.2),
        ("dg", "do0", "0", 2.2),
        ("dg", "do0", "0", 10.0),
        ("confidnet", "do0", "1", 2.2),  # near-OOD regime for cross-check
    ]
    for model, do, reg, rw in slices:
        print(f"\n=== slice: model={model}, do={do}, regime={reg}, reward={rw} ===")
        info = run_one_slice(df, model, do, reg, rw)
        for k, v in info.items():
            if k == "avg_rank_head":
                print(f"{k}:")
                for m, r in list(v.items())[:5]:
                    print(f"    {m}: {r:.3f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()
