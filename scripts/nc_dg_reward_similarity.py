"""
Verify NC profile similarity across DG rewards before committing to an aggregation strategy.

Question: does averaging NC across DG's reward axis obscure relevant information?

Approach:
1. Load nc_metrics.csv, restrict to the 8 NC metrics used in the paper.
2. Z-score each metric across the full VGG-13 grid (all studies, datasets, dropouts, runs, rewards).
3. For each (dataset, dropout, reward) DG cell, take the mean z-vector across its 5 runs.
4. Within-DG-across-rewards variation per (dataset, dropout): mean pairwise L2 distance between
   the per-reward z-vectors, plus per-metric coefficient of variation.
5. Between-paradigm variation: mean pairwise L2 distance between paradigm-level z-vectors
   (confidnet vs. devries vs. dg-averaged) per (dataset, dropout).
6. Compare within-DG vs. between-paradigm. If within-DG is comparable to or larger, averaging hides
   signal and DG rewards should be treated as separate anchors (or a canonical reward selected).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
OUT_DIR = REPO / "ood_eval_outputs" / "nc_dg_reward_similarity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 8 NC metrics used in the paper (from sections/methods/neural_collapse.tex, Table nc_metrics_mapping).
NC_METRICS = [
    "var_collapse",
    "equinorm_uc",
    "equinorm_wc",
    "equiangular_uc",
    "equiangular_wc",
    "max_equiangular_uc",
    "max_equiangular_wc",
    "self_duality",
]


def load_nc() -> pd.DataFrame:
    df = pd.read_csv(NC_CSV)
    keep = ["dataset", "architecture", "study", "dropout", "run", "reward"] + NC_METRICS
    df = df[keep].copy()
    df = df[df["architecture"] == "VGG13"].reset_index(drop=True)
    return df


def zscore_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for m in NC_METRICS:
        mu = out[m].mean()
        sd = out[m].std(ddof=0)
        out[m] = (out[m] - mu) / (sd if sd > 0 else 1.0)
    return out


def cell_mean(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys, as_index=False)[NC_METRICS].mean()


def pairwise_l2(rows: np.ndarray) -> float:
    n = rows.shape[0]
    if n < 2:
        return np.nan
    dists = []
    for i, j in itertools.combinations(range(n), 2):
        dists.append(np.linalg.norm(rows[i] - rows[j]))
    return float(np.mean(dists))


def within_dg_across_rewards(z: pd.DataFrame) -> pd.DataFrame:
    dg = z[z["study"] == "dg"]
    reward_vecs = cell_mean(dg, ["dataset", "dropout", "reward"])
    rows = []
    for (ds, do), sub in reward_vecs.groupby(["dataset", "dropout"]):
        mat = sub[NC_METRICS].to_numpy()
        rewards = sub["reward"].tolist()
        rows.append(
            {
                "dataset": ds,
                "dropout": do,
                "n_rewards": len(rewards),
                "rewards": rewards,
                "mean_pairwise_l2": pairwise_l2(mat),
                "per_metric_std": mat.std(axis=0, ddof=0).tolist(),
            }
        )
    return pd.DataFrame(rows)


def between_paradigm(z: pd.DataFrame) -> pd.DataFrame:
    # Paradigm-level vectors: confidnet and devries use all their runs; DG averages across its reward axis.
    paradigm_vecs = []
    for study in ["confidnet", "devries", "dg"]:
        sub = z[z["study"] == study]
        g = cell_mean(sub, ["dataset", "dropout"])
        g["study"] = study
        paradigm_vecs.append(g)
    pv = pd.concat(paradigm_vecs, ignore_index=True)

    rows = []
    for (ds, do), sub in pv.groupby(["dataset", "dropout"]):
        mat = sub[NC_METRICS].to_numpy()
        rows.append(
            {
                "dataset": ds,
                "dropout": do,
                "n_paradigms": len(sub),
                "mean_pairwise_l2": pairwise_l2(mat),
                "per_metric_std": mat.std(axis=0, ddof=0).tolist(),
            }
        )
    return pd.DataFrame(rows)


def dg_reward_cv(z: pd.DataFrame) -> pd.DataFrame:
    """Per-metric coefficient of variation (absolute, on z-scale) across DG rewards, per cell."""
    dg = z[z["study"] == "dg"]
    reward_vecs = cell_mean(dg, ["dataset", "dropout", "reward"])
    rows = []
    for (ds, do), sub in reward_vecs.groupby(["dataset", "dropout"]):
        mat = sub[NC_METRICS].to_numpy()
        if mat.shape[0] < 2:
            continue
        mu = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=0)
        rec = {"dataset": ds, "dropout": do, "n_rewards": mat.shape[0]}
        for i, name in enumerate(NC_METRICS):
            rec[f"std_{name}"] = float(sd[i])
            rec[f"range_{name}"] = float(mat[:, i].max() - mat[:, i].min())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_nc()
    print(f"Loaded {len(df)} VGG-13 NC rows; studies: {df['study'].value_counts().to_dict()}")

    z = zscore_metrics(df)

    within = within_dg_across_rewards(z)
    between = between_paradigm(z)
    cv = dg_reward_cv(z)

    merged = within[["dataset", "dropout", "n_rewards", "mean_pairwise_l2"]].rename(
        columns={"mean_pairwise_l2": "within_dg_l2"}
    ).merge(
        between[["dataset", "dropout", "mean_pairwise_l2"]].rename(
            columns={"mean_pairwise_l2": "between_paradigm_l2"}
        ),
        on=["dataset", "dropout"],
        how="outer",
    )
    merged["ratio_within_over_between"] = merged["within_dg_l2"] / merged["between_paradigm_l2"]

    print("\n=== Within-DG (across rewards) vs. Between-paradigm L2 on z-scored NC ===")
    print(merged.to_string(index=False))

    print("\n=== Summary: ratio within-DG / between-paradigm ===")
    print(merged["ratio_within_over_between"].describe())

    print("\n=== Per-metric std across DG rewards (z-scale), per cell ===")
    print(cv.to_string(index=False))

    merged.to_csv(OUT_DIR / "within_vs_between_l2.csv", index=False)
    cv.to_csv(OUT_DIR / "dg_reward_per_metric_std.csv", index=False)
    within.to_csv(OUT_DIR / "within_dg_detail.csv", index=False)
    between.to_csv(OUT_DIR / "between_paradigm_detail.csv", index=False)

    print(f"\nSaved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
