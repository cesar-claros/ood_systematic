"""
Head-side H4 specificity test.

Replicates the H4 (per-coordinate Spearman on H1 paradigm pairs) structure from
intervention_mantel_tests.py with two modifications:

1. Delta_clique is recomputed using head-side CSFs only (MSR, MLS, Energy, GEN,
   GE, PE, PCE, REN, GradNorm, pNML, Confidence). Each slice's top clique is
   intersected with this set before taking Jaccard distances, then averaged
   across regimes per §6.4 (mean-Jaccard stability >= 0.60 rule preserved).
2. Candidate NC coordinates extended from the 8 Papyan coords with
   bias_collapse and cdnv_score.

Same H1 pair structure (drop_out x source x paradigm-pair) is used.
Stratified permutation within source matches the original H4 null.

Output: ood_eval_outputs/intervention_mantel/h4_headside_per_coord.csv
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
CLIQ_DIR = REPO / "ood_eval_outputs" / "intervention_cliques"
DELTA_DIR = REPO / "ood_eval_outputs" / "intervention_deltas"
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
OUT_DIR = REPO / "ood_eval_outputs" / "intervention_mantel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
NC_SOURCE_ALIAS = {"supercifar100": "supercifar"}
STABILITY_THRESHOLD = 0.60

PAPYAN = [
    "var_collapse",
    "equinorm_uc", "equinorm_wc",
    "equiangular_uc", "equiangular_wc",
    "max_equiangular_uc", "max_equiangular_wc",
    "self_duality",
]
EXTRA = ["bias_collapse", "cdnv_score"]
NC_METRICS = PAPYAN + EXTRA

HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}

N_PERMUTATIONS = 9_999
SEED = 2
H4_RHO_THRESHOLD = 0.40


def load_clique_slices() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        records = json.loads((CLIQ_DIR / f"cliques_{source}_base.json").read_text())
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
    df["stable"] = df["mean_jaccard"] >= STABILITY_THRESHOLD
    return df


def restrict_head_side(cliques: pd.DataFrame) -> pd.DataFrame:
    out = cliques.copy()
    out["top_clique_head"] = out["top_clique"].apply(
        lambda members: [m for m in members if m in HEAD_SIDE_CSFS]
    )
    return out


def load_nc_z() -> pd.DataFrame:
    df = pd.read_csv(NC_CSV)
    keep = ["dataset", "architecture", "study", "dropout", "run", "reward"] + NC_METRICS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"nc_metrics.csv missing columns: {missing}")
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


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return float("nan")
    return 1.0 - len(sa & sb) / len(sa | sb)


def delta_clique_head(
    cliques: pd.DataFrame, source: str, drop_out: str, a: str, b: str
) -> float:
    ca = cliques[(cliques["source"] == source) & (cliques["drop_out"] == drop_out)
                 & (cliques["paradigm_entry"] == a)]
    cb = cliques[(cliques["source"] == source) & (cliques["drop_out"] == drop_out)
                 & (cliques["paradigm_entry"] == b)]
    dists = []
    for regime, ra in ca.set_index("regime").iterrows():
        if regime not in cb["regime"].values:
            continue
        rb = cb.set_index("regime").loc[regime]
        if not ra["stable"] or not rb["stable"]:
            continue
        dists.append(jaccard(ra["top_clique_head"], rb["top_clique_head"]))
    dists = [d for d in dists if np.isfinite(d)]
    return float(np.mean(dists)) if dists else float("nan")


def _mean_vec(z: pd.DataFrame, source: str, entry: str, drop_out: str) -> np.ndarray | None:
    nc_source = NC_SOURCE_ALIAS.get(source, source)
    sub = z[(z["dataset"] == nc_source) & (z["paradigm_entry"] == entry)
            & (z["drop_out"] == drop_out)]
    if sub.empty:
        return None
    return sub[NC_METRICS].mean(axis=0).to_numpy()


def run() -> pd.DataFrame:
    cliques = restrict_head_side(load_clique_slices())
    h1 = pd.read_csv(DELTA_DIR / "h1_paradigm_pairs.csv")
    z = load_nc_z()

    # Recompute delta_clique using head-side-restricted cliques.
    dchead = []
    for _, row in h1.iterrows():
        dchead.append(delta_clique_head(cliques, row["source"], row["drop_out"],
                                        row["entry_a"], row["entry_b"]))
    h1 = h1.copy()
    h1["delta_clique_head"] = dchead

    rng = np.random.default_rng(SEED)
    rows = []
    for drop_out in ["do0", "do1", "POOLED"]:
        sub = h1 if drop_out == "POOLED" else h1[h1["drop_out"] == drop_out]
        sub = sub.dropna(subset=["delta_clique_head"]).reset_index(drop=True)
        if sub.empty:
            continue

        diffs = np.full((len(sub), len(NC_METRICS)), np.nan)
        for i, row in sub.iterrows():
            va = _mean_vec(z, row["source"], row["entry_a"], row["drop_out"])
            vb = _mean_vec(z, row["source"], row["entry_b"], row["drop_out"])
            if va is None or vb is None:
                continue
            diffs[i] = np.abs(va - vb)

        cliq = sub["delta_clique_head"].to_numpy()
        for k, name in enumerate(NC_METRICS):
            d = diffs[:, k]
            mask = np.isfinite(d) & np.isfinite(cliq)
            if mask.sum() < 3:
                continue
            rho, _ = spearmanr(d[mask], cliq[mask])

            # Stratified permutation within source.
            sub_m = sub.loc[mask].reset_index(drop=True)
            groups = sub_m.groupby("source").indices
            c_valid = cliq[mask]
            d_valid = d[mask]
            greater = 0
            for _ in range(N_PERMUTATIONS):
                perm = c_valid.copy()
                for idx in groups.values():
                    perm[idx] = c_valid[rng.permutation(idx)]
                rp, _ = spearmanr(d_valid, perm)
                if rp >= rho:
                    greater += 1
            p = (greater + 1) / (N_PERMUTATIONS + 1)
            rows.append({
                "drop_out": drop_out,
                "nc_coord": name,
                "family": "papyan" if name in PAPYAN else "extra",
                "rho_obs": float(rho),
                "p": float(p),
                "n": int(mask.sum()),
                "passes_threshold": bool(rho >= H4_RHO_THRESHOLD and p <= 0.05),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["drop_out", "rho_obs"], ascending=[True, False]).reset_index(drop=True)
    return df


def main() -> None:
    print(f"N_PERMUTATIONS = {N_PERMUTATIONS}; SEED = {SEED}")
    print(f"Candidate coords ({len(NC_METRICS)}): {NC_METRICS}")
    print(f"Head-side CSFs ({len(HEAD_SIDE_CSFS)}): {sorted(HEAD_SIDE_CSFS)}")

    df = run()
    out = OUT_DIR / "h4_headside_per_coord.csv"
    df.to_csv(out, index=False)

    print("\n=== H4 head-side per-coordinate Spearman ===")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20,
                           "display.max_rows", 200):
        print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
