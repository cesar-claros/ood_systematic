"""
Sensitivity and supplementary analyses for the H1 paradigm-axis dose-response.

All tests re-run the H1 pipeline (per-source Mantel + Stouffer across sources) on
perturbed inputs so reviewers can see that the primary finding is not an artifact of
a single modelling choice.

Tests
-----
S1  Global shuffle null     : permute paradigm-entry labels of ΔClique globally;
                              confirms that the Mantel framework under random alignment
                              produces a ρ distribution centered near zero (distinct from
                              the observed ~0.58–0.68).
S2  Pair-type split         : split paradigm-pairs into "cross-paradigm" (includes a non-
                              DG entry) and "DG-internal" (both endpoints DG rewards),
                              then re-run Mantel on each subset. Tests whether H1 is
                              driven by real paradigm differences or only by DG-reward
                              variation.
S3  Per-regime H1           : re-compute ΔClique using only one regime at a time (no mean
                              across regimes) and run Mantel. Confirms the signal exists
                              for all four regimes, not just one.
S4  Stability-threshold     : recompute H1 with mean-Jaccard thresholds {0.50, 0.60, 0.70}
                              to confirm the §6.4 inclusion cut-off is not load-bearing.
S5  Alternative ΔNC norms   : recompute ΔNC with L1 (Manhattan) in addition to L2.

Outputs (ood_eval_outputs/intervention_sensitivity/):
- sensitivity_summary.csv  — one row per test/scope with rho, p, n_pairs, pass flag
- s1_null_distribution.csv — per-permutation ρ values for the global-shuffle null
"""
from __future__ import annotations

import ast
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

REPO = Path(__file__).resolve().parents[1]
CLIQ_DIR = REPO / "ood_eval_outputs" / "intervention_cliques"
DELTA_DIR = REPO / "ood_eval_outputs" / "intervention_deltas"
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
OUT_DIR = REPO / "ood_eval_outputs" / "intervention_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
NC_SOURCE_ALIAS = {"supercifar100": "supercifar"}
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
H1_RHO_THRESHOLD = 0.40
N_PERMUTATIONS = 9_999
N_NULL_DRAWS = 500   # S1 shuffle-null replicates (each is a full Mantel pipeline run)
SEED = 0


# ---------------------------------------------------------------------------
# Data loaders (match intervention_delta_tables.py / intervention_mantel_tests.py)
# ---------------------------------------------------------------------------


def load_clique_slices() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        path = CLIQ_DIR / f"cliques_{source}_base.json"
        records = json.loads(path.read_text())
        frames.append(pd.DataFrame(records))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["top_clique"] = df["top_clique"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    )
    df["paradigm_entry"] = [
        f"dg@{r:g}" if m == "dg" else m
        for m, r in zip(df["model"], df["reward"])
    ]
    return df


def load_nc_z() -> pd.DataFrame:
    df = pd.read_csv(NC_CSV)
    keep = ["dataset", "architecture", "study", "dropout", "run", "reward"] + NC_METRICS
    df = df[keep]
    df = df[df["architecture"] == "VGG13"].reset_index(drop=True)
    for m in NC_METRICS:
        mu = df[m].mean()
        sd = df[m].std(ddof=0)
        df[m] = (df[m] - mu) / (sd if sd > 0 else 1.0)
    df["paradigm_entry"] = [
        f"dg@{r:g}" if s == "dg" else s for s, r in zip(df["study"], df["reward"])
    ]
    df["drop_out"] = np.where(df["dropout"] == True, "do1", "do0")  # noqa: E712
    return df


def _jaccard_distance(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return float("nan")
    return 1.0 - len(sa & sb) / len(sa | sb)


def nc_vec(z: pd.DataFrame, source: str, entry: str, drop_out: str) -> np.ndarray | None:
    nc_source = NC_SOURCE_ALIAS.get(source, source)
    sub = z[(z["dataset"] == nc_source) & (z["paradigm_entry"] == entry) & (z["drop_out"] == drop_out)]
    if sub.empty:
        return None
    return sub[NC_METRICS].mean(axis=0).to_numpy()


# ---------------------------------------------------------------------------
# Flexible pair-level H1 constructor with configurable options
# ---------------------------------------------------------------------------


def compute_delta_clique(
    cliques: pd.DataFrame, source: str, drop_out: str,
    entry_a: str, entry_b: str, stability_threshold: float, regime_filter: list[int] | None,
    aggregator: str = "mean",
) -> tuple[float, int]:
    """Aggregate Jaccard distance across regimes with a configurable stability gate."""
    cell_a = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == drop_out)
        & (cliques["paradigm_entry"] == entry_a)
    ].set_index("regime")
    cell_b = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == drop_out)
        & (cliques["paradigm_entry"] == entry_b)
    ].set_index("regime")
    distances: list[float] = []
    for regime in cell_a.index.intersection(cell_b.index):
        if regime_filter is not None and int(regime) not in regime_filter:
            continue
        a_row = cell_a.loc[regime]
        b_row = cell_b.loc[regime]
        if a_row["mean_jaccard"] < stability_threshold or b_row["mean_jaccard"] < stability_threshold:
            continue
        distances.append(_jaccard_distance(a_row["top_clique"], b_row["top_clique"]))
    if not distances:
        return float("nan"), 0
    if aggregator == "mean":
        return float(np.mean(distances)), len(distances)
    if aggregator == "median":
        return float(np.median(distances)), len(distances)
    if aggregator == "max":
        return float(np.max(distances)), len(distances)
    raise ValueError(aggregator)


def _compute_delta_nc(vec_a: np.ndarray | None, vec_b: np.ndarray | None, norm_kind: str) -> float:
    if vec_a is None or vec_b is None:
        return float("nan")
    diff = vec_a - vec_b
    if norm_kind == "l2":
        return float(np.linalg.norm(diff))
    if norm_kind == "l1":
        return float(np.sum(np.abs(diff)))
    raise ValueError(norm_kind)


def build_h1_table(
    cliques: pd.DataFrame, z: pd.DataFrame, *,
    stability_threshold: float = 0.60,
    regime_filter: list[int] | None = None,
    aggregator: str = "mean",
    nc_norm: str = "l2",
) -> pd.DataFrame:
    rows = []
    for drop_out in ["do0", "do1"]:
        for source in SOURCES:
            entries = sorted(
                cliques[(cliques["source"] == source) & (cliques["drop_out"] == drop_out)][
                    "paradigm_entry"
                ].unique()
            )
            for a, b in combinations(entries, 2):
                d_cliq, n_reg = compute_delta_clique(
                    cliques, source, drop_out, a, b, stability_threshold, regime_filter, aggregator
                )
                d_nc = _compute_delta_nc(
                    nc_vec(z, source, a, drop_out), nc_vec(z, source, b, drop_out), nc_norm
                )
                rows.append(
                    {
                        "drop_out": drop_out,
                        "source": source,
                        "entry_a": a,
                        "entry_b": b,
                        "delta_nc": d_nc,
                        "delta_clique": d_cliq,
                        "n_regimes_used": n_reg,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mantel test (identical construct to the primary analysis)
# ---------------------------------------------------------------------------


def _distance_matrix(vals: dict[tuple[str, str], float], entries: list[str]) -> np.ndarray:
    k = len(entries)
    D = np.full((k, k), np.nan)
    for i, a in enumerate(entries):
        for j, b in enumerate(entries):
            if i == j:
                D[i, j] = 0.0
            else:
                key = (a, b) if (a, b) in vals else (b, a)
                if key in vals:
                    D[i, j] = vals[key]
    return D


def _upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


def mantel_per_source(h1_source: pd.DataFrame, rng: np.random.Generator, n_perm: int) -> dict:
    entries = sorted(set(h1_source["entry_a"]).union(h1_source["entry_b"]))
    nc_map = {(a, b): v for a, b, v in zip(h1_source["entry_a"], h1_source["entry_b"], h1_source["delta_nc"])}
    cq_map = {(a, b): v for a, b, v in zip(h1_source["entry_a"], h1_source["entry_b"], h1_source["delta_clique"])}
    D_nc = _distance_matrix(nc_map, entries)
    D_cq = _distance_matrix(cq_map, entries)
    nc_vec_ = _upper(D_nc)
    cq_vec_ = _upper(D_cq)
    mask = np.isfinite(nc_vec_) & np.isfinite(cq_vec_)
    if mask.sum() < 3:
        return {"rho_obs": float("nan"), "p": float("nan"), "n_pairs": int(mask.sum())}
    rho_obs, _ = spearmanr(nc_vec_[mask], cq_vec_[mask])
    k = len(entries)
    greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(k)
        D_cq_perm = D_cq[np.ix_(perm, perm)]
        cq_perm_vec = _upper(D_cq_perm)
        rho_p, _ = spearmanr(nc_vec_[mask], cq_perm_vec[mask])
        if rho_p >= rho_obs:
            greater += 1
    p_upper = (greater + 1) / (n_perm + 1)
    return {"rho_obs": float(rho_obs), "p": float(p_upper), "n_pairs": int(mask.sum())}


def stouffer_combine(p_values: list[float], weights: list[float]) -> tuple[float, float]:
    p = np.clip(np.asarray(p_values, dtype=float), 1e-15, 1 - 1e-15)
    z = norm.isf(p)
    w = np.asarray(weights, dtype=float)
    Z = float(np.sum(w * z) / np.sqrt(np.sum(w ** 2)))
    return Z, float(norm.sf(Z))


def run_h1_mantel(h1: pd.DataFrame, rng: np.random.Generator, label: str, n_perm: int) -> list[dict]:
    rows = []
    for drop_out in ["do0", "do1"]:
        sub_do = h1[h1["drop_out"] == drop_out]
        per_source = []
        for source in SOURCES:
            hsrc = sub_do[sub_do["source"] == source]
            if hsrc.empty or len(hsrc) < 3:
                continue
            res = mantel_per_source(hsrc.reset_index(drop=True), rng, n_perm)
            rows.append(
                {"label": label, "scope": f"{drop_out}|{source}", "drop_out": drop_out,
                 "source": source, **res, "passes_threshold": None}
            )
            per_source.append(res)
        valid = [r for r in per_source if np.isfinite(r["rho_obs"])]
        if valid:
            p_vals = [r["p"] for r in valid]
            weights = [np.sqrt(r["n_pairs"]) for r in valid]
            Z, p_c = stouffer_combine(p_vals, weights)
            weighted_rho = float(
                np.sum(np.array([r["rho_obs"] for r in valid]) * np.array(weights)) / np.sum(weights)
            )
            rows.append(
                {"label": label, "scope": f"{drop_out}|ALL", "drop_out": drop_out,
                 "source": "ALL", "rho_obs": weighted_rho, "p": p_c,
                 "n_pairs": int(sum(r["n_pairs"] for r in valid)),
                 "passes_threshold": bool(weighted_rho >= H1_RHO_THRESHOLD and p_c <= 0.05)}
            )
    return rows


# ---------------------------------------------------------------------------
# Pair-type classification (S2)
# ---------------------------------------------------------------------------


def _is_dg_entry(entry: str) -> bool:
    return entry.startswith("dg@")


def classify_pair(a: str, b: str) -> str:
    if _is_dg_entry(a) and _is_dg_entry(b):
        return "dg_internal"
    return "cross_paradigm"


# ---------------------------------------------------------------------------
# S1: global shuffle null
# ---------------------------------------------------------------------------


def s1_null_distribution(h1_base: pd.DataFrame, rng: np.random.Generator, n_draws: int) -> pd.DataFrame:
    """Permute paradigm-entry labels globally within (drop_out, source) and record weighted rho."""
    records = []
    for draw in range(n_draws):
        shuffled = h1_base.copy()
        for (do_, src), idx in h1_base.groupby(["drop_out", "source"]).indices.items():
            # Shuffle clique values within the (do, source) block to destroy structure while
            # preserving the within-source ΔClique marginals.
            shuffled.loc[idx, "delta_clique"] = (
                h1_base.loc[idx, "delta_clique"].to_numpy()[rng.permutation(len(idx))]
            )
        weighted = _mantel_combined_rho(shuffled, rng)
        records.append({"draw": draw, **weighted})
    return pd.DataFrame(records)


def _mantel_combined_rho(h1: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Return weighted-rho summary for do0/do1 without running the expensive permutation test."""
    result = {}
    for drop_out in ["do0", "do1"]:
        per_source_rhos = []
        per_source_weights = []
        for source in SOURCES:
            sub = h1[(h1["drop_out"] == drop_out) & (h1["source"] == source)]
            entries = sorted(set(sub["entry_a"]).union(sub["entry_b"]))
            nc_map = {(a, b): v for a, b, v in zip(sub["entry_a"], sub["entry_b"], sub["delta_nc"])}
            cq_map = {(a, b): v for a, b, v in zip(sub["entry_a"], sub["entry_b"], sub["delta_clique"])}
            D_nc = _distance_matrix(nc_map, entries)
            D_cq = _distance_matrix(cq_map, entries)
            nc_v = _upper(D_nc)
            cq_v = _upper(D_cq)
            mask = np.isfinite(nc_v) & np.isfinite(cq_v)
            if mask.sum() < 3:
                continue
            rho, _ = spearmanr(nc_v[mask], cq_v[mask])
            per_source_rhos.append(rho)
            per_source_weights.append(np.sqrt(mask.sum()))
        if per_source_rhos:
            r = np.array(per_source_rhos)
            w = np.array(per_source_weights)
            result[f"rho_{drop_out}"] = float(np.sum(r * w) / np.sum(w))
        else:
            result[f"rho_{drop_out}"] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(SEED)
    cliques = load_clique_slices()
    z = load_nc_z()

    # Baseline H1 (matches primary analysis)
    h1_base = build_h1_table(cliques, z)
    records: list[dict] = []
    print("S0  primary baseline")
    records.extend(run_h1_mantel(h1_base, rng, "baseline", N_PERMUTATIONS))

    # S1 global shuffle null (cheap, just rho distribution)
    print("S1  global shuffle null distribution")
    null_df = s1_null_distribution(h1_base, rng, N_NULL_DRAWS)
    null_df.to_csv(OUT_DIR / "s1_null_distribution.csv", index=False)
    for col in ["rho_do0", "rho_do1"]:
        q = null_df[col].quantile([0.025, 0.5, 0.975])
        print(f"  null {col}: median={q[0.5]:+.3f}  95% CI [{q[0.025]:+.3f}, {q[0.975]:+.3f}]  (N={len(null_df)})")

    # S2 pair-type split
    print("S2  pair-type split")
    h1_base_tagged = h1_base.copy()
    h1_base_tagged["pair_type"] = [classify_pair(a, b) for a, b in zip(h1_base_tagged["entry_a"], h1_base_tagged["entry_b"])]
    for ptype in ["cross_paradigm", "dg_internal"]:
        sub = h1_base_tagged[h1_base_tagged["pair_type"] == ptype].drop(columns=["pair_type"])
        records.extend(run_h1_mantel(sub, rng, f"S2_{ptype}", N_PERMUTATIONS))

    # S3 per-regime
    print("S3  per-regime H1")
    for regime in [0, 1, 2, 3]:
        h1_r = build_h1_table(cliques, z, regime_filter=[regime])
        records.extend(run_h1_mantel(h1_r, rng, f"S3_regime{regime}", N_PERMUTATIONS))

    # S4 stability-threshold sweep
    print("S4  stability-threshold sweep")
    for thr in [0.50, 0.70]:
        h1_thr = build_h1_table(cliques, z, stability_threshold=thr)
        records.extend(run_h1_mantel(h1_thr, rng, f"S4_stab{thr:.2f}", N_PERMUTATIONS))

    # S5 alternative ΔNC norm
    print("S5  alternative ΔNC norm (L1)")
    h1_l1 = build_h1_table(cliques, z, nc_norm="l1")
    records.extend(run_h1_mantel(h1_l1, rng, "S5_nc_l1", N_PERMUTATIONS))

    summary = pd.DataFrame(records)
    summary = summary[["label", "scope", "drop_out", "source", "rho_obs", "p", "n_pairs", "passes_threshold"]]
    out = OUT_DIR / "sensitivity_summary.csv"
    summary.to_csv(out, index=False)

    # Console printout: just the ALL rows per label (most informative)
    alls = summary[summary["source"] == "ALL"].copy()
    print("\n=== ALL-source rows across sensitivity tests ===")
    print(alls.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
