"""
Build H1 (cross-paradigm) and H2 (cross-dropout) delta tables.

Each row = one intervention pair, with the two quantities the Mantel/Spearman tests consume:

- ||ΔNC||     : L2 distance between the two conditions' mean 8-dim z-scored NC vectors.
- ΔClique     : mean Jaccard distance (1 − J) between the two conditions' top cliques,
                averaged across the 4 regimes.

§6.4 inclusion rule (mean-Jaccard stability ≥ 0.60) is applied per-slice before aggregation:
a paradigm-pair contribution from a single regime is dropped if either endpoint's slice is
unstable. The paradigm-pair point survives as long as at least one regime remains.

Inputs:
- ood_eval_outputs/intervention_cliques/slice_summary_base.csv  (from intervention_cliques.py)
- ood_eval_outputs/intervention_cliques/cliques_{source}_base.json
- neural_collapse_metrics/nc_metrics.csv

Outputs (ood_eval_outputs/intervention_deltas/):
- h1_paradigm_pairs_{dropout}.csv   : per-dropout H1 table (target n=87 each)
- h2_dropout_pairs.csv              : H2 table (target n=28)
- h1_h2_run_metadata.json
"""
from __future__ import annotations

import ast
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CLIQ_DIR = REPO / "ood_eval_outputs" / "intervention_cliques"
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
OUT_DIR = REPO / "ood_eval_outputs" / "intervention_deltas"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = REPO / "configs" / "intervention_config.yaml"


def load_exclusions() -> dict[str, set[str]]:
    if not CONFIG_PATH.exists():
        return {}
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    raw = cfg.get("paradigm_entries_excluded", {}) or {}
    return {src: set(entries) for src, entries in raw.items()}

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
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
STABILITY_THRESHOLD = 0.60

# NC 'dataset' values differ from clique 'source' values in one case.
NC_SOURCE_ALIAS = {"supercifar100": "supercifar"}


def load_clique_slices() -> pd.DataFrame:
    """Concatenate per-source JSONs; attach normalized paradigm-entry labels.

    Applies paradigm_entries_excluded from the config (e.g., supercifar100/dg@2.2).
    """
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
    df["stable"] = df["mean_jaccard"] >= STABILITY_THRESHOLD

    exclusions = load_exclusions()
    if exclusions:
        mask = pd.Series(False, index=df.index)
        for src, entries in exclusions.items():
            mask |= (df["source"] == src) & (df["paradigm_entry"].isin(entries))
        n_excl = int(mask.sum())
        if n_excl:
            df = df[~mask].copy()
            print(
                f"Applied paradigm_entries_excluded: dropped {n_excl} slices "
                f"(sources: {sorted(exclusions.keys())})"
            )
    return df


def load_nc_z() -> pd.DataFrame:
    """Load NC metrics for VGG-13 and z-score each metric across the full grid."""
    df = pd.read_csv(NC_CSV)
    keep = ["dataset", "architecture", "study", "dropout", "run", "reward"] + NC_METRICS
    df = df[keep]
    df = df[df["architecture"] == "VGG13"].reset_index(drop=True)
    z = df.copy()
    for m in NC_METRICS:
        mu = z[m].mean()
        sd = z[m].std(ddof=0)
        z[m] = (z[m] - mu) / (sd if sd > 0 else 1.0)
    z["paradigm_entry"] = [
        f"dg@{r:g}" if s == "dg" else s for s, r in zip(z["study"], z["reward"])
    ]
    z["drop_out"] = np.where(z["dropout"] == True, "do1", "do0")  # noqa: E712
    return z


def nc_vec(z: pd.DataFrame, source: str, paradigm_entry: str, drop_out: str) -> np.ndarray | None:
    """Mean z-vector (8-dim) across runs for one (source, paradigm-entry, dropout) cell."""
    nc_source = NC_SOURCE_ALIAS.get(source, source)
    sub = z[
        (z["dataset"] == nc_source)
        & (z["paradigm_entry"] == paradigm_entry)
        & (z["drop_out"] == drop_out)
    ]
    if sub.empty:
        return None
    return sub[NC_METRICS].mean(axis=0).to_numpy()


def jaccard_distance(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return float("nan")
    return 1.0 - len(sa & sb) / len(sa | sb)


def delta_clique_across_regimes(
    cliques: pd.DataFrame, source: str, drop_out: str,
    entry_a: str, entry_b: str, require_stable: bool = True
) -> tuple[float, int, list[int]]:
    """Mean Jaccard distance between the two entries' cliques, aggregated across regimes."""
    cell_a = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == drop_out)
        & (cliques["paradigm_entry"] == entry_a)
    ]
    cell_b = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == drop_out)
        & (cliques["paradigm_entry"] == entry_b)
    ]
    distances: list[float] = []
    regimes_used: list[int] = []
    for regime, a_row in cell_a.set_index("regime").iterrows():
        if regime not in cell_b["regime"].values:
            continue
        b_row = cell_b.set_index("regime").loc[regime]
        if require_stable and (not a_row["stable"] or not b_row["stable"]):
            continue
        distances.append(jaccard_distance(a_row["top_clique"], b_row["top_clique"]))
        regimes_used.append(int(regime))
    if not distances:
        return float("nan"), 0, []
    return float(np.mean(distances)), len(distances), sorted(regimes_used)


def delta_clique_dropout(
    cliques: pd.DataFrame, source: str, entry: str, require_stable: bool = True
) -> tuple[float, int, list[int]]:
    """Mean Jaccard distance between do0 and do1 cliques, aggregated across regimes."""
    cell_0 = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == "do0")
        & (cliques["paradigm_entry"] == entry)
    ]
    cell_1 = cliques[
        (cliques["source"] == source)
        & (cliques["drop_out"] == "do1")
        & (cliques["paradigm_entry"] == entry)
    ]
    distances: list[float] = []
    regimes_used: list[int] = []
    for regime, a_row in cell_0.set_index("regime").iterrows():
        if regime not in cell_1["regime"].values:
            continue
        b_row = cell_1.set_index("regime").loc[regime]
        if require_stable and (not a_row["stable"] or not b_row["stable"]):
            continue
        distances.append(jaccard_distance(a_row["top_clique"], b_row["top_clique"]))
        regimes_used.append(int(regime))
    if not distances:
        return float("nan"), 0, []
    return float(np.mean(distances)), len(distances), sorted(regimes_used)


def build_h1(cliques: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """For each dropout level, enumerate paradigm-entry pairs within each source."""
    rows = []
    for drop_out in ["do0", "do1"]:
        for source in SOURCES:
            entries = sorted(
                cliques[(cliques["source"] == source) & (cliques["drop_out"] == drop_out)][
                    "paradigm_entry"
                ].unique()
            )
            for a, b in combinations(entries, 2):
                vec_a = nc_vec(z, source, a, drop_out)
                vec_b = nc_vec(z, source, b, drop_out)
                if vec_a is None or vec_b is None:
                    dnc_l2 = float("nan")
                    dnc_l1 = float("nan")
                else:
                    diff = vec_a - vec_b
                    dnc_l2 = float(np.linalg.norm(diff, ord=2))
                    dnc_l1 = float(np.linalg.norm(diff, ord=1))
                d_cliq, n_reg, regimes = delta_clique_across_regimes(
                    cliques, source, drop_out, a, b
                )
                rows.append(
                    {
                        "drop_out": drop_out,
                        "source": source,
                        "entry_a": a,
                        "entry_b": b,
                        "delta_nc_l2": dnc_l2,
                        "delta_nc_l1": dnc_l1,
                        "delta_clique": d_cliq,
                        "n_regimes_used": n_reg,
                        "regimes_used": regimes,
                    }
                )
    return pd.DataFrame(rows)


def build_h2(cliques: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """For each (source, paradigm-entry), the do0 vs do1 intervention pair."""
    rows = []
    for source in SOURCES:
        entries = sorted(
            cliques[cliques["source"] == source]["paradigm_entry"].unique()
        )
        for entry in entries:
            vec_0 = nc_vec(z, source, entry, "do0")
            vec_1 = nc_vec(z, source, entry, "do1")
            if vec_0 is None or vec_1 is None:
                dnc_l2 = float("nan")
                dnc_l1 = float("nan")
            else:
                diff = vec_0 - vec_1
                dnc_l2 = float(np.linalg.norm(diff, ord=2))
                dnc_l1 = float(np.linalg.norm(diff, ord=1))
            d_cliq, n_reg, regimes = delta_clique_dropout(cliques, source, entry)
            rows.append(
                {
                    "source": source,
                    "paradigm_entry": entry,
                    "delta_nc_l2": dnc_l2,
                    "delta_nc_l1": dnc_l1,
                    "delta_clique": d_cliq,
                    "n_regimes_used": n_reg,
                    "regimes_used": regimes,
                }
            )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, name: str, group_keys: list[str] | None = None) -> dict:
    n_total = len(df)
    has_both = df["delta_nc_l2"].notna() & df["delta_clique"].notna()
    n_ok = int(has_both.sum())
    info = {
        "name": name,
        "n_total": n_total,
        "n_complete": n_ok,
        "n_missing_nc": int(df["delta_nc_l2"].isna().sum()),
        "n_missing_clique": int(df["delta_clique"].isna().sum()),
    }
    print(f"\n=== {name} ===")
    print(f"total rows: {n_total}; complete (both deltas finite): {n_ok}")
    if group_keys:
        print(df.groupby(group_keys).size().rename("n").to_string())
    return info


def main() -> None:
    print(f"STABILITY_THRESHOLD = {STABILITY_THRESHOLD}")
    cliques = load_clique_slices()
    z = load_nc_z()

    h1 = build_h1(cliques, z)
    h2 = build_h2(cliques, z)

    h1_info = report(h1, "H1 paradigm-pair table", ["drop_out", "source"])
    h2_info = report(h2, "H2 dropout-pair table", ["source"])

    h1_out = OUT_DIR / "h1_paradigm_pairs.csv"
    h2_out = OUT_DIR / "h2_dropout_pairs.csv"
    h1.to_csv(h1_out, index=False)
    h2.to_csv(h2_out, index=False)

    cfg_hash = (
        hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:12]
        if CONFIG_PATH.exists()
        else None
    )
    meta = {
        "sources": SOURCES,
        "nc_metrics": NC_METRICS,
        "stability_threshold": STABILITY_THRESHOLD,
        "config_hash": cfg_hash,
        "h1_summary": h1_info,
        "h2_summary": h2_info,
    }
    (OUT_DIR / "h1_h2_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {h1_out}")
    print(f"Wrote {h2_out}")


if __name__ == "__main__":
    main()
