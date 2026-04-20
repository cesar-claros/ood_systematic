"""
Extract per-slice top cliques for the intervention dose-response analysis.

Loops over every (source, model, reward, drop_out, regime) slice from the `_fix-config`
score CSVs, runs Friedman -> Conover-Holm -> Bron-Kerbosch, and records the top clique
plus a bootstrap-over-runs stability score (mean Jaccard). VGG-13 only — ResNet-18
runs are excluded upstream at the retrieve_scores.py stage via --network bbvgg13.

Outputs (under ood_eval_outputs/intervention_cliques/):
- cliques_{source}.json   : every slice's top clique + stability metadata
- slice_summary.csv       : one row per slice, for the §6.4 20%-abort-rule check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils_stats import (  # noqa: E402
    KNOWN_META_COLS,
    conover_posthoc_from_pivot,
    friedman_blocked,
    greedy_exclusive_layers,
    jaccard,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    standardize_metric,
)

SCORES_DIR = REPO / "scores_risk"
CLIP_DIR = REPO / "clip_scores"
OUT_DIR = REPO / "ood_eval_outputs" / "intervention_cliques"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
METRICS = ["AUGRC", "AURC"]  # RC metric group
ALPHA = 0.05
N_BOOT = 200
STABILITY_THRESHOLD = 0.60
SEED = 0

# Mirrors stats_eval.py --filter-methods: drop 'global'/'class' variants except the
# PCA/KPCA RecError baselines (whose canonical form is the 'global' variant).
FILTER_KEEP_EXCEPTIONS = {
    "KPCA RecError global",
    "PCA RecError global",
    "MCD-KPCA RecError global",
    "MCD-PCA RecError global",
}


def load_fix_config_long(source: str, mcd: bool) -> pd.DataFrame:
    """Load the two RC metric _fix-config CSVs for a source and return a long-format frame."""
    frames = []
    mcd_tag = "MCD-True" if mcd else "MCD-False"
    for metric in METRICS:
        path = SCORES_DIR / f"scores_all_{metric}_{mcd_tag}_Conv_{source}_fix-config.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        ds_cols = [c for c in df.columns if c not in KNOWN_META_COLS]
        long = df.melt(
            id_vars=[c for c in df.columns if c not in ds_cols],
            value_vars=ds_cols,
            var_name="dataset",
            value_name="score",
        )
        long["metric"] = metric
        long["reward"] = (
            long["reward"].astype(str).str.lower().map(lambda x: x.split("rew")[1]).astype(float)
        )
        long["MCD"] = "1" if mcd else "0"
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["score_std"] = out.apply(lambda r: standardize_metric(r["metric"], r["score"]), axis=1)
    return out


def filter_base_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only base CSFs — drop 'global'/'class' variants (except PCA/KPCA RecError global)."""
    mask = df["methods"].str.contains("global|class", case=False, na=False)
    mask &= ~df["methods"].isin(FILTER_KEEP_EXCEPTIONS)
    return df[~mask].copy()


def attach_regime(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Merge the CLIP near/mid/far group label (column 'group' ∈ {'0','1','2','3'}) onto each row."""
    clip_path = CLIP_DIR / f"clip_distances_{source}.csv"
    clip = pd.read_csv(clip_path, header=[0, 1])
    clip.columns = clip.columns.droplevel(0)
    clip = clip.rename(
        columns={"Unnamed: 0_level_1": "dataset", "Unnamed: 5_level_1": "group"}
    )
    clip["group"] = clip["group"].apply(lambda x: str(int(x)) if pd.notna(x) else x)
    return df.merge(clip[["dataset", "group"]], on="dataset", how="left")


def top_clique_from_pivot(pivot: pd.DataFrame) -> tuple[list[str], dict] | None:
    """Run Conover-Holm posthoc + Bron-Kerbosch and return the top greedy layer."""
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    ph = conover_posthoc_from_pivot(pivot)
    avg_ranks = pivot.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    cliques = maximal_cliques_from_pmatrix(ph, ALPHA)
    if not cliques:
        return None
    scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
    layers = greedy_exclusive_layers(scored)
    if not layers:
        return None
    top = layers[0]
    meta = {
        "size": int(top["size"]),
        "best_rank": float(top["best_rank"]),
        "mean_rank": float(top["mean_rank"]),
        "n_maximal_cliques": len(cliques),
    }
    return sorted(top["members"]), meta


def _slice_key(meta: dict) -> str:
    return f"{meta['source']}|{meta['model']}|{meta['drop_out']}|{meta['reward']}|{meta['regime']}"


def bootstrap_stability(sub: pd.DataFrame, baseline_members: list[str], rng: np.random.Generator) -> dict:
    """Resample the 5 runs B times per (dataset, metric); recompute top clique; mean Jaccard vs baseline."""
    runs = sorted(sub["run"].unique().tolist())
    n_runs = len(runs)
    if n_runs < 2:
        return {"n_runs": n_runs, "bootstrap_B": 0, "mean_jaccard": float("nan"), "stable": None}

    jaccards: list[float] = []
    empty_draws = 0
    for _ in range(N_BOOT):
        draw = rng.choice(runs, size=n_runs, replace=True)
        parts = []
        for new_run, original_run in enumerate(draw):
            chunk = sub[sub["run"] == original_run].copy()
            chunk["run"] = f"b{new_run}"
            parts.append(chunk)
        boot = pd.concat(parts, ignore_index=True)
        boot["block"] = boot[["dataset", "metric", "run"]].astype(str).agg("|".join, axis=1)
        stat, _, pivot = friedman_blocked(
            boot, entity_col="methods", block_col="block", value_col="score_std"
        )
        if stat is None or np.isnan(stat):
            empty_draws += 1
            continue
        result = top_clique_from_pivot(pivot)
        if result is None:
            empty_draws += 1
            continue
        members, _ = result
        jaccards.append(jaccard(baseline_members, members))

    if not jaccards:
        return {
            "n_runs": n_runs,
            "bootstrap_B": N_BOOT,
            "bootstrap_valid": 0,
            "mean_jaccard": float("nan"),
            "stable": None,
        }

    mean_j = float(np.mean(jaccards))
    return {
        "n_runs": n_runs,
        "bootstrap_B": N_BOOT,
        "bootstrap_valid": len(jaccards),
        "bootstrap_empty": empty_draws,
        "mean_jaccard": mean_j,
        "std_jaccard": float(np.std(jaccards, ddof=0)),
        "stable": bool(mean_j >= STABILITY_THRESHOLD),
    }


def run_slice(sub: pd.DataFrame, meta: dict, seed: int) -> dict:
    """Process one slice: Friedman -> top clique -> bootstrap stability. Picklable for workers."""
    rng = np.random.default_rng(seed)
    base = dict(meta)
    if sub.empty:
        return {**base, "status": "empty"}

    sub = sub.copy()
    sub["block"] = sub[["dataset", "metric", "run"]].astype(str).agg("|".join, axis=1)
    stat, p, pivot = friedman_blocked(
        sub, entity_col="methods", block_col="block", value_col="score_std"
    )
    info = {
        **base,
        "n_blocks": int(pivot.shape[0]),
        "n_methods": int(pivot.shape[1]),
        "friedman_stat": float(stat) if stat is not None and not np.isnan(stat) else None,
        "friedman_p": float(p) if p is not None and not np.isnan(p) else None,
    }
    if stat is None or np.isnan(stat):
        info["status"] = "friedman_nan"
        return info

    result = top_clique_from_pivot(pivot)
    if result is None:
        info["status"] = "no_clique"
        return info

    members, meta_out = result
    info["status"] = "ok"
    info["top_clique"] = members
    info.update(meta_out)
    info.update(bootstrap_stability(sub, members, rng))
    return info


def _worker(payload: tuple[pd.DataFrame, dict, int]) -> dict:
    sub, meta, seed = payload
    return run_slice(sub, meta, seed)


def build_payloads(df: pd.DataFrame, source: str) -> list[tuple[pd.DataFrame, dict, int]]:
    """One payload per (model, drop_out, reward, regime) slice; deterministic per-slice seeds."""
    payloads: list[tuple[pd.DataFrame, dict, int]] = []
    grouped = df.groupby(["model", "drop out", "reward", "group"], sort=True)
    for idx, ((model, dropout, reward, regime), sub) in enumerate(grouped):
        meta = {
            "source": source,
            "model": model,
            "drop_out": dropout,
            "reward": float(reward),
            "regime": regime,
        }
        slice_seed = SEED + (hash(_slice_key(meta)) & 0x7FFFFFFF)
        payloads.append((sub.reset_index(drop=True), meta, slice_seed))
    return payloads


def process_source(source: str, filter_methods: bool, executor: ProcessPoolExecutor | None) -> tuple[list[dict], dict]:
    print(f"\n=== source={source} ===", flush=True)
    df = load_fix_config_long(source, mcd=False)
    if filter_methods:
        before = len(df)
        df = filter_base_methods(df)
        print(f"  filter-methods: dropped {before - len(df)} rows ({df['methods'].nunique()} CSFs remain)", flush=True)
    df = attach_regime(df, source)
    df = df.dropna(subset=["group"])
    payloads = build_payloads(df, source)
    print(f"  loaded {len(df)} rows; {len(payloads)} slices to evaluate", flush=True)

    if executor is None:
        records = [_worker(p) for p in payloads]
    else:
        records = list(executor.map(_worker, payloads))

    status_counts = pd.Series([r["status"] for r in records]).value_counts().to_dict()
    stable_counts = pd.Series(
        [r.get("stable") for r in records if r.get("status") == "ok"]
    ).value_counts(dropna=False).to_dict()

    summary = {
        "source": source,
        "n_slices": len(records),
        "status_counts": status_counts,
        "stable_counts": {str(k): int(v) for k, v in stable_counts.items()},
    }
    print(f"  status: {status_counts}")
    print(f"  stable: {summary['stable_counts']}")
    return records, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--filter-methods",
        action="store_true",
        help="Keep only base CSFs — drop 'global'/'class' variants except PCA/KPCA RecError global "
        "(mirrors stats_eval.py --filter-methods).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional filename suffix for JSON/CSV outputs (e.g., 'base' when --filter-methods is on).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Process-pool workers for slice-level parallelism. Set to 1 to run serially.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or ("base" if args.filter_methods else None)
    suffix = f"_{tag}" if tag else ""

    all_records: list[dict] = []
    summaries: list[dict] = []

    executor_cm = (
        ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    )
    executor = executor_cm if executor_cm is not None else None
    try:
        print(f"workers: {args.workers}", flush=True)
        for source in SOURCES:
            records, summary = process_source(source, args.filter_methods, executor)
            all_records.extend(records)
            summaries.append(summary)

            out_json = OUT_DIR / f"cliques_{source}{suffix}.json"
            with out_json.open("w") as fh:
                json.dump(records, fh, indent=2, default=str)
            print(f"  wrote {out_json}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary_df = pd.DataFrame(all_records)
    summary_csv = OUT_DIR / f"slice_summary{suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv} ({len(summary_df)} slices)")

    ok = summary_df[summary_df["status"] == "ok"]
    n_ok = len(ok)
    if n_ok:
        n_unstable = int((ok["stable"] == False).sum())  # noqa: E712
        unstable_frac = n_unstable / n_ok
        print(
            f"\n§6.4 abort-rule check: "
            f"{n_unstable}/{n_ok} OK slices unstable (mean Jaccard < {STABILITY_THRESHOLD}) "
            f"= {unstable_frac:.1%}"
        )
        print(f"  20%% threshold {'EXCEEDED — abort trigger' if unstable_frac > 0.20 else 'within bounds'}")

    meta_path = OUT_DIR / f"run_metadata{suffix}.json"
    meta = {
        "sources": SOURCES,
        "metrics": METRICS,
        "alpha": ALPHA,
        "n_boot": N_BOOT,
        "stability_threshold": STABILITY_THRESHOLD,
        "seed": SEED,
        "filter_methods": args.filter_methods,
        "filter_keep_exceptions": sorted(FILTER_KEEP_EXCEPTIONS) if args.filter_methods else [],
        "per_source_summary": summaries,
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
