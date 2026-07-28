"""Partial Mantel tests: NC geometry vs CSF rankings, controlling for convergence.

Addresses the reviewer request to control the NC-to-CSF-ranking Mantel
association (paper Appendix G, Table 18) for training-convergence proxies.
Stored convergence covariates per model block:

  - iid_msr : IID test-set metric (AUGRC) of the MSR baseline CSF, the
              closest stored proxy for achieved fit quality.
  - iid_mean: IID test-set metric averaged over the retained CSFs.
  - lr      : learning rate (varies only in the ViT stratum, where FD-Shifts
              uses per-configuration swept best values; each CNN stratum is a
              single paradigm with one fixed recipe, common SGD base lr 0.1
              and a paradigm-specific epoch budget of 470/250/300 for
              ConfidNet/DeVries/DG, identical across the four sources, so lr
              and epochs are within-stratum constants).
  - ds      : source-dataset identity (binary same/different distance), which
              subsumes epoch-budget and schedule differences across datasets.

Partial Mantel follows Legendre & Legendre: residualize both condensed
distance vectors on the covariate distances (OLS with intercept, after rank
transformation for the Spearman flavor), correlate residuals, and assess
significance by jointly permuting rows/columns of the residualized NC
distance matrix (9,999 permutations, one-sided, +1 correction).

Run from `code/`:
  ./.venv/bin/python mantel_partial_convergence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR / "archived"))
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from mantel_analysis import (
    PAPYAN_NC_METRICS,
    build_rank_matrix,
    mantel_test,
    method_rank_distance,
    nc_distance_matrix,
)
import mantel_analysis
from nc_regime_analysis import (
    compute_mean_ood_score,
    join_nc_scores,
    load_nc_metrics,
    load_scores,
    ood_columns,
)

N_PERMS = 9999
SEED = 42
DATASETS = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
STRATA = [
    ("Conv", "confidnet"),
    ("Conv", "devries"),
    ("Conv", "dg"),
    ("ViT", "vit"),
]
KEEP_EXCEPTIONS = {
    "KPCA RecError global", "PCA RecError global",
    "MCD-KPCA RecError global", "MCD-PCA RecError global",
}
BLOCK_KEYS = ["dataset", "architecture", "study", "run"]
OUT_DIR = CODE_DIR / "mantel_partial_outputs"


def filter_methods(scores: pd.DataFrame) -> pd.DataFrame:
    """Drop projection variants except the PCA/KPCA global scores (paper roster)."""
    mask = scores["methods"].str.contains("global|class", case=False, na=False)
    mask &= ~scores["methods"].isin(KEEP_EXCEPTIONS)
    return scores[~mask]


def residualize(y: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """OLS residuals of y on covariates (with intercept).

    Args:
        y: Response vector, shape (n,).
        covs: Covariate matrix, shape (n, k); k may be 0.

    Returns:
        Residual vector of the same shape as y.
    """
    X = np.column_stack([np.ones(len(y))] + [covs[:, j] for j in range(covs.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_mantel(
    D1: np.ndarray,
    D2: np.ndarray,
    cov_mats: list[np.ndarray],
    n_perms: int = N_PERMS,
    seed: int = SEED,
) -> dict[str, float]:
    """Spearman partial Mantel of D1 vs D2 controlling for covariate matrices.

    Rank-transforms all condensed vectors, residualizes D1 and D2 on the
    covariates, correlates the residuals (Pearson on ranks = Spearman flavor),
    and permutes the residualized D1 matrix for the null distribution.

    Args:
        D1: Square distance matrix (NC side, permuted under the null).
        D2: Square distance matrix (ranking side).
        cov_mats: Covariate distance matrices to control for (may be empty).
        n_perms: Number of permutations.
        seed: RNG seed.

    Returns:
        Dict with r_obs, p_value, n_perms.
    """
    n = D1.shape[0]
    idx = np.triu_indices(n, k=1)
    d1 = rankdata(D1[idx])
    d2 = rankdata(D2[idx])
    covs = (np.column_stack([rankdata(C[idx]) for C in cov_mats])
            if cov_mats else np.empty((len(d1), 0)))

    e1 = residualize(d1, covs)
    e2 = residualize(d2, covs)
    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
    if denom == 0:
        return {"r_obs": float("nan"), "p_value": float("nan"), "n_perms": n_perms}
    r_obs = float(e1 @ e2 / denom)

    E1 = squareform(e1)
    rng = np.random.default_rng(seed)
    count_ge = 0
    e2_unit = e2 / np.linalg.norm(e2)
    for _ in range(n_perms):
        perm = rng.permutation(n)
        e1_perm = E1[np.ix_(perm, perm)][idx]
        norm1 = np.linalg.norm(e1_perm)
        if norm1 == 0:
            continue
        if float(e1_perm @ e2_unit / norm1) >= r_obs:
            count_ge += 1
    return {"r_obs": r_obs,
            "p_value": (count_ge + 1) / (n_perms + 1),
            "n_perms": n_perms}


def abs_diff_matrix(values: np.ndarray) -> np.ndarray:
    """Pairwise absolute-difference distance matrix of a scalar covariate."""
    v = values.astype(float)
    return np.abs(v[:, None] - v[None, :])


def block_covariates(merged: pd.DataFrame, models_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate convergence covariates to block level, aligned with models_df.

    Args:
        merged: Joined NC + scores frame (post method filtering).
        models_df: Block-level frame returned by build_rank_matrix, whose row
            order defines the distance-matrix object order.

    Returns:
        models_df with iid_msr, iid_mean, lr_mean columns appended.
    """
    msr = (merged[merged["methods"] == "MSR"]
           .groupby(BLOCK_KEYS, as_index=False)["test"].mean()
           .rename(columns={"test": "iid_msr"}))
    iid_mean = (merged.groupby(BLOCK_KEYS, as_index=False)["test"].mean()
                .rename(columns={"test": "iid_mean"}))
    out = models_df.merge(msr, on=BLOCK_KEYS, how="left")
    out = out.merge(iid_mean, on=BLOCK_KEYS, how="left")
    if "lr" in merged.columns:
        lr = (merged.groupby(BLOCK_KEYS, as_index=False)["lr"].mean()
              .rename(columns={"lr": "lr_mean"}))
        out = out.merge(lr, on=BLOCK_KEYS, how="left")
    else:
        out["lr_mean"] = 0.0
    return out


def covariate_sets_for(backbone: str) -> dict[str, list[str]]:
    """Named covariate sets per stratum (lr only varies for ViT)."""
    sets = {
        "iid_msr": ["iid_msr"],
        "iid_mean": ["iid_mean"],
        "dataset": ["ds"],
        "iid_msr+dataset": ["iid_msr", "ds"],
    }
    if backbone == "ViT":
        sets["iid_msr+lr"] = ["iid_msr", "lr"]
        sets["iid_msr+lr+dataset"] = ["iid_msr", "lr", "ds"]
    return sets


def run_stratum(backbone: str, study: str) -> list[dict]:
    """Full + partial Mantel for one (backbone, paradigm) stratum."""
    nc = load_nc_metrics("neural_collapse_metrics/nc_metrics.csv")
    nc["dataset"] = nc["dataset"].replace({"supercifar": "supercifar100"})
    scores = load_scores("scores_risk", "AUGRC", backbone, "False", DATASETS)
    scores = filter_methods(scores)
    scores = scores[scores["study"] == study]
    nc = nc[nc["study"] == study]

    merged = join_nc_scores(nc, scores)
    if merged.empty:
        logger.error(f"{backbone}/{study}: empty join")
        return []
    merged = compute_mean_ood_score(merged, ood_columns(merged))

    models_df, rank_matrix, method_names = build_rank_matrix(
        merged, "mean_ood_score", ascending=True)
    models_df = block_covariates(merged, models_df)
    n = len(models_df)
    logger.info(f"{backbone}/{study}: {n} blocks, {len(method_names)} methods")

    nc_cols = [c for c in PAPYAN_NC_METRICS if c in models_df.columns]
    D_nc = nc_distance_matrix(models_df, nc_cols)
    D_rank = method_rank_distance(rank_matrix)

    cov_lookup = {
        "iid_msr": abs_diff_matrix(models_df["iid_msr"].values),
        "iid_mean": abs_diff_matrix(models_df["iid_mean"].values),
        "lr": abs_diff_matrix(models_df["lr_mean"].values),
        "ds": (models_df["dataset"].values[:, None]
               != models_df["dataset"].values[None, :]).astype(float),
    }

    rows = []
    full = mantel_test(D_nc, D_rank, n_perms=N_PERMS)
    rows.append({
        "backbone": backbone, "study": study, "n_blocks": n,
        "control": "(none, replication)", "r": full["r_obs"],
        "p": full["p_value"],
    })
    baseline_partial = partial_mantel(D_nc, D_rank, [])
    rows.append({
        "backbone": backbone, "study": study, "n_blocks": n,
        "control": "(none, partial-code path)", "r": baseline_partial["r_obs"],
        "p": baseline_partial["p_value"],
    })
    for name, keys in covariate_sets_for(backbone).items():
        res = partial_mantel(D_nc, D_rank, [cov_lookup[k] for k in keys])
        rows.append({
            "backbone": backbone, "study": study, "n_blocks": n,
            "control": name, "r": res["r_obs"], "p": res["p_value"],
        })
        logger.info(f"  control={name:<20s} r={res['r_obs']:+.4f} "
                    f"p={res['p_value']:.4f}")

    corr = models_df[["iid_msr", "iid_mean", "lr_mean"]].corr().round(3)
    logger.info(f"  covariate correlations:\n{corr}")
    return rows


def main() -> None:
    """Run all strata and write the summary CSV and markdown table."""
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    for backbone, study in STRATA:
        all_rows.extend(run_stratum(backbone, study))
    out = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "partial_mantel_summary.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"wrote {csv_path}")

    md_lines = [
        "# Partial Mantel: NC vs CSF rankings, controlling convergence proxies\n",
        f"\nPermutations: {N_PERMS}, seed {SEED}, Spearman flavor, "
        "residual-permutation method (Legendre).\n",
        "\nCovariates: iid_msr / iid_mean = IID test AUGRC (MSR / mean over "
        "retained CSFs); lr = learning rate (varies only for ViT); dataset = "
        "source-identity distance. CNN strata share one training schedule per "
        "dataset by design, so schedule/epoch covariates are constants there.\n\n",
        out.round(4).to_markdown(index=False),
        "\n",
    ]
    md_path = OUT_DIR / "partial_mantel_summary.md"
    md_path.write_text("".join(md_lines))
    logger.info(f"wrote {md_path}")


if __name__ == "__main__":
    main()
