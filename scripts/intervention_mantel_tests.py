"""
Primary Spearman + Mantel permutation tests for the intervention dose-response protocol.

For each paradigm/dropout pair we have two scalars per (source, pair): ||ΔNC|| and ΔClique.
Because pairs within a source form a k×k distance matrix over paradigm-entries, the
defensible null preserves the block structure of each source's matrix. We therefore run:

- **H1 (paradigm axis)**: per-source Mantel test (Spearman correlation between the ΔNC and
  ΔClique distance matrices over that source's paradigm-entries), with permutations of
  paradigm-entry labels on one matrix. Per-source p-values are combined via Stouffer's Z.
  Reported per dropout level (do0/do1) and pooled; H1a/H1b agreement on |Δρ| ≤ 0.10 is
  checked per §2026-04-19 pooling rule.
- **H2 (dropout axis)**: single intervention pair per (source, paradigm-entry), so a block-
  permuted Spearman (shuffle ΔClique within source) gives the exact stratified null.

Supplementary:
- **H3 (directionality)**: fraction of top-quartile-||ΔNC|| pairs with ΔClique > 0.
- **H4 (specificity)**: per-coordinate Spearman ρ between single-axis NC difference and
  ΔClique, reported per dropout level.

Outputs (ood_eval_outputs/intervention_mantel/):
- h1_mantel_results.csv, h2_spearman_results.csv, h3_h4_supplementary.csv
- run_metadata.json (seed, permutation count, thresholds, timestamp)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

REPO = Path(__file__).resolve().parents[1]
DELTA_DIR = REPO / "ood_eval_outputs" / "intervention_deltas"
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
OUT_DIR = REPO / "ood_eval_outputs" / "intervention_mantel"
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

# H1 threshold (ρ ≥ 0.40) and H2 threshold (ρ ≥ 0.30) per the pre-registered protocol.
H1_RHO_THRESHOLD = 0.40
H2_RHO_THRESHOLD = 0.30
POOL_DELTA_RHO_CAP = 0.10  # |rho(do0) - rho(do1)| ≤ 0.10 to justify pooling

N_PERMUTATIONS = 9_999
SEED = 0
TOP_QUARTILE_FRACTION = 0.25
H3_THRESHOLD = 0.60  # ≥ 60% of top-||ΔNC|| pairs must show non-empty ΔClique
H4_RHO_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_list(cell: object) -> list:
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return ast.literal_eval(cell) if cell else []
    return []


def load_h1() -> pd.DataFrame:
    df = pd.read_csv(DELTA_DIR / "h1_paradigm_pairs.csv")
    df["regimes_used"] = df["regimes_used"].apply(_parse_list)
    return df


def load_h2() -> pd.DataFrame:
    df = pd.read_csv(DELTA_DIR / "h2_dropout_pairs.csv")
    df["regimes_used"] = df["regimes_used"].apply(_parse_list)
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


def stouffer_combine(p_values: list[float], weights: list[float] | None = None) -> tuple[float, float]:
    """Combine independent p-values; returns combined Z and two-sided p (clamped into (0, 1))."""
    p = np.asarray(p_values, dtype=float)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    z = norm.isf(p)  # higher Z = smaller p (one-sided, upper tail)
    if weights is None:
        w = np.ones_like(z)
    else:
        w = np.asarray(weights, dtype=float)
    Z = float(np.sum(w * z) / np.sqrt(np.sum(w ** 2)))
    combined_p = float(norm.sf(Z))  # upper-tail combined p
    return Z, combined_p


# ---------------------------------------------------------------------------
# H1: per-source Mantel with paradigm-entry permutations, then Stouffer-combine
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
    nc_vals = {(a, b): v for a, b, v in zip(h1_source["entry_a"], h1_source["entry_b"], h1_source["delta_nc_l2"])}
    cq_vals = {(a, b): v for a, b, v in zip(h1_source["entry_a"], h1_source["entry_b"], h1_source["delta_clique"])}
    D_nc = _distance_matrix(nc_vals, entries)
    D_cq = _distance_matrix(cq_vals, entries)

    nc_vec = _upper(D_nc)
    cq_vec = _upper(D_cq)
    mask = np.isfinite(nc_vec) & np.isfinite(cq_vec)
    if mask.sum() < 3:
        return {"rho_obs": float("nan"), "p": float("nan"), "n_pairs": int(mask.sum()), "n_entries": len(entries)}
    rho_obs, _ = spearmanr(nc_vec[mask], cq_vec[mask])

    # Permute paradigm-entry labels of the clique matrix; keep NC matrix fixed.
    k = len(entries)
    greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(k)
        D_cq_perm = D_cq[np.ix_(perm, perm)]
        cq_vec_perm = _upper(D_cq_perm)
        rho_p, _ = spearmanr(nc_vec[mask], cq_vec_perm[mask])
        if rho_p >= rho_obs:
            greater += 1
    p_upper = (greater + 1) / (n_perm + 1)
    return {
        "rho_obs": float(rho_obs),
        "p": float(p_upper),
        "n_pairs": int(mask.sum()),
        "n_entries": len(entries),
    }


def run_h1(h1: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for drop_out in ["do0", "do1"]:
        sub_do = h1[h1["drop_out"] == drop_out]
        per_source = []
        for source in SOURCES:
            hsrc = sub_do[sub_do["source"] == source]
            if hsrc.empty:
                continue
            res = mantel_per_source(hsrc.reset_index(drop=True), rng, N_PERMUTATIONS)
            rows.append(
                {
                    "hypothesis": "H1",
                    "scope": f"{drop_out}|{source}",
                    "drop_out": drop_out,
                    "source": source,
                    **res,
                }
            )
            per_source.append(res)

        # Stouffer across sources for this dropout
        valid = [r for r in per_source if np.isfinite(r["rho_obs"])]
        if valid:
            p_vals = [r["p"] for r in valid]
            weights = [np.sqrt(r["n_pairs"]) for r in valid]
            Z, p_combined = stouffer_combine(p_vals, weights)
            median_rho = float(np.median([r["rho_obs"] for r in valid]))
            weighted_rho = float(
                np.sum(np.array([r["rho_obs"] for r in valid]) * np.array(weights)) / np.sum(weights)
            )
            rows.append(
                {
                    "hypothesis": "H1",
                    "scope": f"{drop_out}|ALL",
                    "drop_out": drop_out,
                    "source": "ALL",
                    "rho_obs": weighted_rho,
                    "median_rho": median_rho,
                    "stouffer_Z": Z,
                    "p": p_combined,
                    "n_pairs": int(sum(r["n_pairs"] for r in valid)),
                    "n_entries": None,
                    "passes_threshold": bool(weighted_rho >= H1_RHO_THRESHOLD and p_combined <= 0.05),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H2: block-permuted Spearman across (source, paradigm-entry)
# ---------------------------------------------------------------------------


def run_h2(h2: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED + 1)
    df = h2.dropna(subset=["delta_nc_l2", "delta_clique"]).copy()
    rho_obs, _ = spearmanr(df["delta_nc_l2"], df["delta_clique"])

    # Shuffle delta_clique within source → exact stratified null.
    source_groups = df.groupby("source").indices
    n_perm = N_PERMUTATIONS
    greater = 0
    clique = df["delta_clique"].to_numpy().copy()
    nc = df["delta_nc_l2"].to_numpy()
    for _ in range(n_perm):
        perm = clique.copy()
        for idx in source_groups.values():
            perm[idx] = clique[rng.permutation(idx)]
        rho_p, _ = spearmanr(nc, perm)
        if rho_p >= rho_obs:
            greater += 1
    p = (greater + 1) / (n_perm + 1)
    return {
        "rho_obs": float(rho_obs),
        "p": float(p),
        "n": int(len(df)),
        "passes_threshold": bool(rho_obs >= H2_RHO_THRESHOLD and p <= 0.05),
    }


# ---------------------------------------------------------------------------
# H3: directionality — top-quartile ||ΔNC||, fraction with ΔClique > 0
# ---------------------------------------------------------------------------


def run_h3(h1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drop_out in ["do0", "do1", "POOLED"]:
        sub = h1 if drop_out == "POOLED" else h1[h1["drop_out"] == drop_out]
        sub = sub.dropna(subset=["delta_nc_l2", "delta_clique"])
        if sub.empty:
            continue
        q_threshold = sub["delta_nc_l2"].quantile(1.0 - TOP_QUARTILE_FRACTION)
        top = sub[sub["delta_nc_l2"] >= q_threshold]
        nonempty_frac = float((top["delta_clique"] > 0).mean()) if len(top) else float("nan")
        rows.append(
            {
                "hypothesis": "H3",
                "drop_out": drop_out,
                "quartile_threshold": float(q_threshold),
                "n_top_quartile": int(len(top)),
                "fraction_nonempty_clique": nonempty_frac,
                "passes_threshold": bool(nonempty_frac >= H3_THRESHOLD),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H4: specificity — per-NC-coordinate Spearman ρ vs ΔClique
# ---------------------------------------------------------------------------


def _mean_vec_per_cell(z: pd.DataFrame, source: str, entry: str, drop_out: str) -> np.ndarray | None:
    nc_source = NC_SOURCE_ALIAS.get(source, source)
    sub = z[(z["dataset"] == nc_source) & (z["paradigm_entry"] == entry) & (z["drop_out"] == drop_out)]
    if sub.empty:
        return None
    return sub[NC_METRICS].mean(axis=0).to_numpy()


def run_h4(h1: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for drop_out in ["do0", "do1", "POOLED"]:
        sub = h1 if drop_out == "POOLED" else h1[h1["drop_out"] == drop_out]
        sub = sub.dropna(subset=["delta_clique"]).reset_index(drop=True)
        if sub.empty:
            continue

        # Build per-pair per-NC-coordinate absolute differences using the z-scored NC frame.
        per_pair_diffs = np.full((len(sub), len(NC_METRICS)), np.nan)
        for i, row in sub.iterrows():
            vec_a = _mean_vec_per_cell(z, row["source"], row["entry_a"], row["drop_out"])
            vec_b = _mean_vec_per_cell(z, row["source"], row["entry_b"], row["drop_out"])
            if vec_a is None or vec_b is None:
                continue
            per_pair_diffs[i] = np.abs(vec_a - vec_b)

        for k, name in enumerate(NC_METRICS):
            diffs = per_pair_diffs[:, k]
            clique = sub["delta_clique"].to_numpy()
            mask = np.isfinite(diffs) & np.isfinite(clique)
            if mask.sum() < 3:
                continue
            rho_obs, _ = spearmanr(diffs[mask], clique[mask])

            # Stratified permutation within source to match H1/H2 nulls.
            source_groups = sub.loc[mask].groupby("source").indices
            c_valid = clique[mask]
            d_valid = diffs[mask]
            greater = 0
            for _ in range(N_PERMUTATIONS):
                perm = c_valid.copy()
                for idx in source_groups.values():
                    perm[idx] = c_valid[rng.permutation(idx)]
                rho_p, _ = spearmanr(d_valid, perm)
                if rho_p >= rho_obs:
                    greater += 1
            p = (greater + 1) / (N_PERMUTATIONS + 1)
            rows.append(
                {
                    "hypothesis": "H4",
                    "drop_out": drop_out,
                    "nc_coord": name,
                    "rho_obs": float(rho_obs),
                    "p": float(p),
                    "n": int(mask.sum()),
                    "passes_threshold": bool(rho_obs >= H4_RHO_THRESHOLD and p <= 0.05),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["drop_out", "rho_obs"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    h1 = load_h1()
    h2 = load_h2()
    z = load_nc_z()

    print(f"H1 rows = {len(h1)}; H2 rows = {len(h2)}; N permutations = {N_PERMUTATIONS}")

    h1_res = run_h1(h1)
    h2_res = run_h2(h2)
    h3_res = run_h3(h1)
    h4_res = run_h4(h1, z)

    # Pooling check on H1 at the ALL-sources level
    h1_pool = h1_res[h1_res["source"] == "ALL"]
    delta_rho = None
    if len(h1_pool) == 2:
        delta_rho = abs(float(h1_pool.iloc[0]["rho_obs"]) - float(h1_pool.iloc[1]["rho_obs"]))
    pooling_allowed = bool(delta_rho is not None and delta_rho <= POOL_DELTA_RHO_CAP)

    h1_out = OUT_DIR / "h1_mantel_results.csv"
    h2_out = OUT_DIR / "h2_spearman_results.csv"
    h3h4_out = OUT_DIR / "h3_h4_supplementary.csv"
    h1_res.to_csv(h1_out, index=False)
    pd.DataFrame([{"hypothesis": "H2", **h2_res}]).to_csv(h2_out, index=False)
    pd.concat([h3_res, h4_res], ignore_index=True).to_csv(h3h4_out, index=False)

    meta = {
        "sources": SOURCES,
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "h1_threshold_rho": H1_RHO_THRESHOLD,
        "h2_threshold_rho": H2_RHO_THRESHOLD,
        "h3_threshold_frac": H3_THRESHOLD,
        "h4_threshold_rho": H4_RHO_THRESHOLD,
        "pool_delta_rho_cap": POOL_DELTA_RHO_CAP,
        "pool_delta_rho_observed": delta_rho,
        "pooling_allowed": pooling_allowed,
        "h2_result": h2_res,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # --- Console summary ---
    print("\n=== H1 (paradigm-axis Mantel, Stouffer-combined across sources) ===")
    cols = ["scope", "drop_out", "source", "rho_obs", "p", "n_pairs", "passes_threshold"]
    print(h1_res[cols].to_string(index=False))
    if delta_rho is not None:
        print(
            f"\n  |rho(do0) - rho(do1)| = {delta_rho:.3f}  → pooling "
            f"{'ALLOWED (Δ ≤ 0.10)' if pooling_allowed else 'DISALLOWED (Δ > 0.10)'}"
        )

    print("\n=== H2 (dropout-axis stratified-permutation Spearman) ===")
    print(
        f"  rho = {h2_res['rho_obs']:+.3f}, p = {h2_res['p']:.4f}, n = {h2_res['n']}  "
        f"→ {'PASS' if h2_res['passes_threshold'] else 'FAIL'} (threshold ρ ≥ {H2_RHO_THRESHOLD}, p ≤ 0.05)"
    )

    print("\n=== H3 (directionality: top-quartile ΔNC → non-empty ΔClique) ===")
    print(h3_res.to_string(index=False))

    print("\n=== H4 (per-NC-coordinate specificity) ===")
    print(h4_res.to_string(index=False))

    print(f"\nWrote {h1_out}")
    print(f"Wrote {h2_out}")
    print(f"Wrote {h3h4_out}")


if __name__ == "__main__":
    main()
