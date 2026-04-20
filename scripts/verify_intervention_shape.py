"""Verify that intervention outputs on disk match the frozen spec.

Reads configs/intervention_config.yaml and cross-checks:
  1. Clique JSONs (per source)    -- grid shape, n_methods, CSF set
  2. NC CSV                       -- metrics present, coverage per cell
  3. H1 / H2 delta tables         -- row counts match expected pair counts
  4. Run metadata                 -- config hash stamped

Exits non-zero (and prints a shape mismatch report) on any violation.
Also writes a short JSON report summarizing what was checked.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "intervention_config.yaml"
CLIQ_DIR = REPO / "ood_eval_outputs" / "intervention_cliques"
DELTA_DIR = REPO / "ood_eval_outputs" / "intervention_deltas"
NC_CSV = REPO / "neural_collapse_metrics" / "nc_metrics.csv"
REPORT_PATH = REPO / "ood_eval_outputs" / "intervention_shape_report.json"


def config_hash(cfg_text: str) -> str:
    return hashlib.sha256(cfg_text.encode()).hexdigest()[:12]


def load_cliques(sources: list[str]) -> pd.DataFrame:
    frames = []
    for src in sources:
        recs = json.loads((CLIQ_DIR / f"cliques_{src}_base.json").read_text())
        df = pd.DataFrame(recs)
        df["source"] = src
        frames.append(df)
    all_ = pd.concat(frames, ignore_index=True)
    all_["top_clique"] = all_["top_clique"].apply(
        lambda x: x if isinstance(x, list) else ast.literal_eval(x) if isinstance(x, str) else []
    )
    all_["paradigm_entry"] = [
        f"dg@{r:g}" if m == "dg" else m for m, r in zip(all_["model"], all_["reward"])
    ]
    return all_


def check_cliques(cfg: dict, cliques: pd.DataFrame) -> list[str]:
    errs: list[str] = []
    exp = cfg["expected"]

    # Global slice count (raw cluster output)
    if len(cliques) != exp["total_slices"]:
        errs.append(f"total slices = {len(cliques)}, expected {exp['total_slices']}")

    # Per-source slice counts
    per_src = cliques.groupby("source").size().to_dict()
    for src, want in exp["slices_per_source"].items():
        got = per_src.get(src, 0)
        if got != want:
            errs.append(f"slices[{src}] = {got}, expected {want}")

    # n_methods uniform
    ok = cliques[cliques["status"] == "ok"]
    bad = ok[ok["n_methods"] != exp["n_methods_per_slice"]]
    if not bad.empty:
        errs.append(
            f"{len(bad)} slices with n_methods != {exp['n_methods_per_slice']} "
            f"(values: {sorted(bad['n_methods'].unique().tolist())})"
        )

    # CSF set: union of all top-clique members must be a subset of configured CSFs
    observed_methods: set[str] = set()
    for c in ok["top_clique"]:
        observed_methods.update(c)
    configured = set(cfg["csfs"])
    extra = observed_methods - configured
    if extra:
        errs.append(f"unexpected CSFs in cliques: {sorted(extra)}")

    # Per-cell regime coverage: for each (source, dropout, paradigm_entry), all 4 regimes must exist
    cell_cols = ["source", "drop_out", "paradigm_entry"]
    regimes_expected = set(cfg["regimes"])
    for (src, do, entry), sub in ok.groupby(cell_cols):
        got = set(sub["regime"].astype(str))
        missing = regimes_expected - got
        if missing:
            errs.append(
                f"missing regimes {sorted(missing)} for {src}|{do}|{entry}"
            )

    # Paradigm-entry set per source matches config
    for src, expected_entries in cfg["paradigm_entries"].items():
        got = sorted(cliques[cliques["source"] == src]["paradigm_entry"].unique())
        want = sorted(expected_entries)
        if got != want:
            errs.append(
                f"paradigm_entries[{src}] = {got}, expected {want}"
            )

    return errs


def check_nc(cfg: dict) -> list[str]:
    errs: list[str] = []
    if not NC_CSV.exists():
        return [f"NC CSV missing: {NC_CSV}"]
    nc = pd.read_csv(NC_CSV)
    nc = nc[nc["architecture"] == cfg["backbone"]]

    # 8 NC metrics present
    missing = [m for m in cfg["nc_metrics"] if m not in nc.columns]
    if missing:
        errs.append(f"NC metrics missing in CSV: {missing}")

    # Per-cell run coverage
    alias = cfg["nc_source_alias"]
    n_runs = cfg["expected"]["n_nc_runs_per_cell"]
    for src, entries in cfg["paradigm_entries"].items():
        nc_src = alias.get(src, src)
        for entry in entries:
            study = entry.split("@")[0] if "@" in entry else entry
            reward = float(entry.split("@")[1]) if "@" in entry else None
            for do, do_bool in [("do0", False), ("do1", True)]:
                sub = nc[
                    (nc["dataset"] == nc_src)
                    & (nc["study"] == study)
                    & (nc["dropout"] == do_bool)
                ]
                if reward is not None:
                    sub = sub[sub["reward"] == reward]
                if len(sub) != n_runs:
                    errs.append(
                        f"NC runs for {src}|{do}|{entry} = {len(sub)}, expected {n_runs}"
                    )
    return errs


def check_deltas(cfg: dict) -> list[str]:
    errs: list[str] = []
    exp = cfg["expected"]
    effective = cfg.get("paradigm_entries_effective", cfg["paradigm_entries"])

    h1_path = DELTA_DIR / "h1_paradigm_pairs.csv"
    h2_path = DELTA_DIR / "h2_dropout_pairs.csv"
    if not h1_path.exists():
        errs.append(f"missing {h1_path}")
    else:
        h1 = pd.read_csv(h1_path)
        if len(h1) != exp["h1_total_pairs"]:
            errs.append(f"H1 rows = {len(h1)}, expected {exp['h1_total_pairs']}")
        for do in cfg["dropouts"]:
            n = (h1["drop_out"] == do).sum()
            if n != exp["h1_pairs_per_dropout"]:
                errs.append(
                    f"H1 rows[{do}] = {n}, expected {exp['h1_pairs_per_dropout']}"
                )
        # Per-source pair counts equal C(k,2) over the effective entry list
        for src, entries in effective.items():
            want = len(list(combinations(entries, 2)))
            for do in cfg["dropouts"]:
                n = ((h1["source"] == src) & (h1["drop_out"] == do)).sum()
                if n != want:
                    errs.append(
                        f"H1 rows[{src}|{do}] = {n}, expected {want}"
                    )
        # Excluded paradigm entries should not appear in H1
        for src, excl in (cfg.get("paradigm_entries_excluded") or {}).items():
            leaked = h1[
                (h1["source"] == src)
                & (h1["entry_a"].isin(excl) | h1["entry_b"].isin(excl))
            ]
            if not leaked.empty:
                errs.append(
                    f"H1 contains {len(leaked)} excluded pairs for {src} "
                    f"(entries: {sorted(excl)})"
                )
    if not h2_path.exists():
        errs.append(f"missing {h2_path}")
    else:
        h2 = pd.read_csv(h2_path)
        if len(h2) != exp["h2_pairs"]:
            errs.append(f"H2 rows = {len(h2)}, expected {exp['h2_pairs']}")
        for src, excl in (cfg.get("paradigm_entries_excluded") or {}).items():
            leaked = h2[(h2["source"] == src) & (h2["paradigm_entry"].isin(excl))]
            if not leaked.empty:
                errs.append(
                    f"H2 contains {len(leaked)} excluded rows for {src} "
                    f"(entries: {sorted(excl)})"
                )
    return errs


def main() -> int:
    cfg_text = CONFIG_PATH.read_text()
    cfg = yaml.safe_load(cfg_text)
    cfg_hash = config_hash(cfg_text)

    print(f"config: {CONFIG_PATH.relative_to(REPO)}  (hash={cfg_hash}, version={cfg['version']})")
    print(f"checking against cliques, NC CSV, and delta tables ...\n")

    cliques = load_cliques(cfg["sources"])
    errs_cliques = check_cliques(cfg, cliques)
    errs_nc = check_nc(cfg)
    errs_deltas = check_deltas(cfg)

    sections = [("cliques", errs_cliques), ("nc_metrics", errs_nc), ("deltas", errs_deltas)]
    total_errs = sum(len(e) for _, e in sections)

    for name, errs in sections:
        tag = "OK" if not errs else f"FAIL ({len(errs)})"
        print(f"[{tag}] {name}")
        for e in errs:
            print(f"    - {e}")

    report = {
        "config_hash": cfg_hash,
        "config_version": cfg["version"],
        "counts_observed": {
            "total_slices": int(len(cliques)),
            "per_source": cliques.groupby("source").size().to_dict(),
            "h1_total_pairs": int(len(pd.read_csv(DELTA_DIR / "h1_paradigm_pairs.csv"))) if (DELTA_DIR / "h1_paradigm_pairs.csv").exists() else None,
            "h2_pairs": int(len(pd.read_csv(DELTA_DIR / "h2_dropout_pairs.csv"))) if (DELTA_DIR / "h2_dropout_pairs.csv").exists() else None,
        },
        "errors": {name: errs for name, errs in sections},
        "status": "ok" if total_errs == 0 else "fail",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_PATH.relative_to(REPO)}")

    if total_errs == 0:
        print("\nall shape checks passed.")
        return 0
    print(f"\n{total_errs} shape mismatches — see errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
