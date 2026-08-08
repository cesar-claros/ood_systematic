"""Aggregate X6 Tier-A measurement JSONs into a per-cell CSV and a report.

Reads the per-checkpoint JSONs written by measure_checkpoint.py and produces:
  <out_dir>/tier_a_summary.csv  one row per cell: pool factors, recovery
      margins (global and class level), stability vs its k/D null, class
      -subspace heterogeneity, common-mode and weight dials, Tier-A
      predictions, cross-arm deltas, runtimes;
  <out_dir>/tier_a_report.md    grouped tables (mean +- sd across runs,
      sd with ddof=1), Tier-A prediction tallies (the one-sided no-benefit
      claims to be scored after the freeze), arm-consistency summaries
      (correct-only vs all-sample vs standardized), manifest coverage, and
      flagged cells.

Aggregation only: no outcome table is read, so this is safe to run before
the rule freeze. Analysis stays HPC-side with the JSONs (outputs/ is
gitignored). Dependency-light by design (json + numpy): the campaign
container has no dataframe library requirement.

Usage (from code/, where the JSONs live under x6_spectral/outputs/):
    python x6_spectral/aggregate_tier_a.py \
        [--out_dir=x6_spectral/outputs] [--arm=correct_only]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

SOURCE_NORMALIZATION = {
    "supercifar": "supercifar100",
    "super_cifar100": "supercifar100",
    "tiny-imagenet-200": "tinyimagenet",
}
SOURCE_ORDER = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
VGG_PATTERN = re.compile(
    r"(?P<source>[a-z0-9\-]+)_paper_sweep/confidnet_bbvgg13_"
    r"do(?P<do>[01])_run(?P<run>\d+)_rew")
VIT_PATTERN = re.compile(
    r"vit/(?P<source>[a-z0-9_\-]+)_modelvit_bbvit_lr(?P<lr>[0-9.]+)_"
    r"bs\d+_run(?P<run>\d+)_do(?P<do>[01])_rew")


def parse_model_path(model_path: str) -> dict | None:
    """Pool factors (backbone, source, dropout, run, lr) from the cell name."""
    match = VIT_PATTERN.search(model_path)
    if match:
        source = SOURCE_NORMALIZATION.get(match["source"], match["source"])
        return {"backbone": "ViT", "source": source,
                "dropout": int(match["do"]), "run": int(match["run"]),
                "lr": match["lr"]}
    match = VGG_PATTERN.search(model_path)
    if match:
        source = SOURCE_NORMALIZATION.get(match["source"], match["source"])
        return {"backbone": "VGG13", "source": source,
                "dropout": int(match["do"]), "run": int(match["run"]),
                "lr": ""}
    return None


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def cell_row(record: dict, arm_name: str) -> dict:
    """Flatten one measurement JSON into a summary row for the chosen arm."""
    factors = parse_model_path(record["model_path"]) or {
        "backbone": "?", "source": "?", "dropout": -1, "run": -1, "lr": ""}
    arm = record["arms"][arm_name]
    via = arm["viability"]
    het = arm["class_heterogeneity"]
    census = arm["census"]
    aligns = record.get("spike_alignments", {})
    mean_span = aligns.get("align_mean_span", [])
    row_space = aligns.get("align_w_rowspace", [])
    omegas = census.get("spike_omegas", [])
    tier = record["tier_a"]

    row = {
        "model_path": record["model_path"], **factors,
        "study": record["study"], "dataset": record["dataset"],
        "n_classes": record["n_classes"], "dim": record["dim"],
        "n_train": record["n_train"],
        "eff_n_correct": record["effective_n_correct"],
        "eff_frac": safe_ratio(record["effective_n_correct"],
                               record["n_train"]),
        "train_acc": record["train_acc"],
        "id_val_accuracy": record["id_val_accuracy"],
        "sigma2_bulk": census["sigma2_bulk"], "aspect_y": census["y"],
        "n_spikes": census["n_spikes"],
        "n_nan_inversions": sum(1 for o in omegas
                                if o is None or math.isnan(o)),
        "omega_mean_aligned": via["omega_mean_aligned"],
        "thr_global": via["thr_global"],
        "global_margin": safe_ratio(via["omega_mean_aligned"],
                                    via["thr_global"]),
        "global_viable": bool(via["global_viable"]),
        "omega_class_median": float(np.median(via["omega_per_class"])),
        "thr_class_median": float(np.median(via["thr_per_class"])),
        "class_margin": safe_ratio(float(np.median(via["omega_per_class"])),
                                   float(np.median(via["thr_per_class"]))),
        "frac_classes_viable": via["frac_classes_viable"],
        "class_viable": bool(via["class_viable"]),
        "n_residue_spikes": via["n_residue_spikes"],
        "stability": arm["stability"], "stability_k": arm["stability_k"],
        "stability_null": arm["stability_null"],
        "stability_ratio": safe_ratio(arm["stability"],
                                      arm["stability_null"]),
        "heterogeneity": het["heterogeneity"],
        "het_within": het["within"], "het_within_sd": het["within_sd"],
        "het_evidence": bool(het["heterogeneity"]
                             > 2 * het.get("within_sd", 0.0)),
        "het_classes_used": het["n_classes_used"],
        "common_mode_frac": arm["common_mode"]["energy_fraction"],
        "amplitude_cv": arm["common_mode"]["amplitude_cv"],
        "w_gap": arm["w_gap"], "w_top_align": arm["w_top_align"],
        "n_meanaligned_top40": sum(1 for v in mean_span if v > 0.5),
        "n_rowaligned_top40": sum(1 for v in row_space if v > 0.5),
        "pred_global": tier["global"]["prediction"],
        "pred_class_pred": tier["class pred"]["prediction"],
        "pred_class_avg": tier["class avg"]["prediction"],
        "routing_note": bool(tier["class pred"].get("routing_note")),
        "runtime_forward_sec": record["runtime_sec"]["forward"],
        "runtime_diag_sec": record["runtime_sec"]["diagnostics"],
    }
    for other, tag in (("all", "all"), ("all_standardized", "std")):
        other_arm = record["arms"].get(other)
        if other_arm is None or other == arm_name:
            row[f"d_omega_aligned_{tag}"] = float("nan")
            row[f"d_stability_{tag}"] = float("nan")
            row[f"d_nspikes_{tag}"] = float("nan")
            continue
        row[f"d_omega_aligned_{tag}"] = (
            other_arm["viability"]["omega_mean_aligned"]
            - via["omega_mean_aligned"])
        row[f"d_stability_{tag}"] = other_arm["stability"] - arm["stability"]
        row[f"d_nspikes_{tag}"] = (other_arm["census"]["n_spikes"]
                                   - census["n_spikes"])
    return row


def group_sort_key(key: tuple) -> tuple:
    backbone, source, dropout = key
    src_rank = SOURCE_ORDER.index(source) if source in SOURCE_ORDER else 99
    return (backbone, src_rank, dropout)


def group_rows(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["backbone"], row["source"], row["dropout"])
        groups.setdefault(key, []).append(row)
    return dict(sorted(groups.items(), key=lambda kv: group_sort_key(kv[0])))


def mean_sd(values: list[float], precision: int = 2) -> str:
    vals = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(vals) == 0:
        return "n/a"
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return f"{vals.mean():.{precision}f}+-{sd:.{precision}f}"


def count_true(rows: list[dict], field: str) -> str:
    return f"{sum(bool(r[field]) for r in rows)}/{len(rows)}"


def md_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def group_label(key: tuple, rows: list[dict]) -> str:
    backbone, source, dropout = key
    lrs = sorted({r["lr"] for r in rows if r["lr"]})
    lr_part = f" lr{','.join(lrs)}" if lrs else ""
    return f"{backbone} {source} do{dropout}{lr_part}"


def coverage_section(records: list[dict], out_dir: Path) -> str:
    manifest_path = Path(__file__).resolve().parent / "manifest_dev_pool.txt"
    lines = [f"Cells aggregated: {len(records)}."]
    if manifest_path.exists():
        expected = [ln.strip() for ln in manifest_path.read_text().splitlines()
                    if ln.strip() and not ln.strip().startswith("#")]
        have = {r["model_path"] for r in records}
        missing = [c for c in expected if c not in have]
        extra = sorted(have - set(expected))
        lines.append(f"Manifest cells: {len(expected)}; missing JSONs: "
                     f"{len(missing)}; JSONs outside the manifest: "
                     f"{len(extra)}.")
        if missing:
            shown = ", ".join(f"`{c}`" for c in missing[:8])
            more = f", ... ({len(missing) - 8} more)" if len(missing) > 8 \
                else ""
            lines.append("Missing: " + shown + more)
        if extra:
            lines.append("Outside manifest: "
                         + ", ".join(f"`{c}`" for c in extra))
    else:
        lines.append("Manifest not found next to the script; "
                     "coverage check skipped.")
    return "\n\n".join(lines)


def recovery_section(groups: dict[tuple, list[dict]]) -> str:
    body = []
    for key, rows in groups.items():
        body.append([
            group_label(key, rows), str(len(rows)),
            mean_sd([r["omega_mean_aligned"] for r in rows]),
            f"{rows[0]['thr_global']:.3f}",
            mean_sd([r["global_margin"] for r in rows], 1),
            count_true(rows, "global_viable"),
            mean_sd([r["omega_class_median"] for r in rows]),
            f"{np.median([r['thr_class_median'] for r in rows]):.2f}",
            mean_sd([r["class_margin"] for r in rows], 1),
            count_true(rows, "class_viable"),
            mean_sd([r["frac_classes_viable"] for r in rows]),
        ])
    headers = ["group", "n", "omega_aligned", "thr_glob", "marginx",
               "glob viable", "omega_cls med", "thr_cls", "cls marginx",
               "cls viable", "frac cls viable"]
    return md_table(headers, body)


def structure_section(groups: dict[tuple, list[dict]]) -> str:
    body = []
    for key, rows in groups.items():
        body.append([
            group_label(key, rows),
            mean_sd([r["stability"] for r in rows]),
            f"{np.mean([r['stability_null'] for r in rows]):.3f}",
            mean_sd([r["stability_ratio"] for r in rows], 1),
            mean_sd([float(r["n_spikes"]) for r in rows], 1),
            mean_sd([float(r["n_residue_spikes"]) for r in rows], 1),
            mean_sd([r["heterogeneity"] for r in rows], 3),
            count_true(rows, "het_evidence"),
            mean_sd([r["common_mode_frac"] for r in rows], 3),
            mean_sd([r["w_gap"] for r in rows]),
            mean_sd([r["w_top_align"] for r in rows]),
            mean_sd([r["id_val_accuracy"] for r in rows], 3),
        ])
    headers = ["group", "stability", "null", "stab/null", "n_spikes",
               "n_residue", "heterogeneity", "het evid", "common-mode",
               "w_gap", "w_align", "val acc"]
    return md_table(headers, body)


def predictions_section(groups: dict[tuple, list[dict]]) -> str:
    body = []
    claims: list[str] = []
    for key, rows in groups.items():
        cells = [group_label(key, rows)]
        for field in ("pred_global", "pred_class_pred", "pred_class_avg"):
            tally = sum(r[field] == "no-benefit" for r in rows)
            cells.append(f"{tally}/{len(rows)} no-benefit")
        body.append(cells)
        for r in rows:
            for field, variant in (("pred_global", "global"),
                                   ("pred_class_pred", "class pred")):
                if r[field] == "no-benefit":
                    claims.append(f"`{r['model_path']}` ({variant})")
    table = md_table(["group", "global", "class pred", "class avg"], body)
    claim_text = ("One-sided claims to score after the freeze "
                  "(projection should NOT significantly help):\n- "
                  + "\n- ".join(claims)) if claims else (
                  "No cell produced a global or class-pred no-benefit claim: "
                  "recovery holds across the pool, so every benefit sign is "
                  "Tier-B territory (deployment-batch orientation).")
    return table + "\n\n" + claim_text


def consistency_section(rows: list[dict]) -> str:
    lines = []
    for tag, label in (("all", "all-sample minus correct-only"),
                       ("std", "standardized minus correct-only")):
        for backbone in sorted({r["backbone"] for r in rows}):
            sub = [r for r in rows if r["backbone"] == backbone]
            deltas = {
                "omega_aligned": [r[f"d_omega_aligned_{tag}"] for r in sub],
                "stability": [r[f"d_stability_{tag}"] for r in sub],
                "n_spikes": [r[f"d_nspikes_{tag}"] for r in sub],
            }
            parts = []
            for name, vals in deltas.items():
                clean = [v for v in vals if not math.isnan(v)]
                if clean:
                    parts.append(f"{name}: {mean_sd(clean, 3)} "
                                 f"(max |{max(abs(v) for v in clean):.3f}|)")
            if parts:
                lines.append(f"- {label}, {backbone}: " + "; ".join(parts))
    return "\n".join(lines) if lines else "No comparison arms present."


def flags_section(rows: list[dict], errors: list[str]) -> str:
    flags = []
    for r in rows:
        cell = f"`{r['model_path']}`"
        if not r["global_viable"]:
            flags.append(f"{cell}: global recovery FAILS")
        if not r["class_viable"]:
            flags.append(f"{cell}: class-level recovery fails "
                         f"(frac {r['frac_classes_viable']:.2f})")
        if r["stability_ratio"] < 2:
            flags.append(f"{cell}: stability {r['stability']:.2f} under 2x "
                         f"null {r['stability_null']:.2f}")
        if r["routing_note"]:
            flags.append(f"{cell}: routing warning "
                         f"(val acc {r['id_val_accuracy']:.2f})")
        if r["n_nan_inversions"] > 0:
            flags.append(f"{cell}: {r['n_nan_inversions']} spike inversions "
                         "returned NaN (outliers at/below the edge)")
    flags += [f"unreadable JSON: {e}" for e in errors]
    return "\n".join(f"- {f}" for f in flags) if flags else \
        "None: every cell recovers, is stable, and parsed cleanly."


def headline_section(rows: list[dict]) -> str:
    trusted = [r for r in rows
               if r["global_viable"] and r["stability_ratio"] >= 2]

    def pick(backbone: str, field: str) -> list[float]:
        return [float(r[field]) for r in trusted
                if r["backbone"] == backbone]

    bullets = []
    if len(trusted) < len(rows):
        bullets.append(
            f"Backbone contrasts below use the {len(trusted)}/{len(rows)} "
            "structurally trustworthy cells (global recovery holds and "
            "stability is at least twice its null); the rest are in Flags.")
    if pick("ViT", "heterogeneity") and pick("VGG13", "heterogeneity"):
        n_vit = len(pick("ViT", "heterogeneity"))
        n_vgg = len(pick("VGG13", "heterogeneity"))
        ev_vit = sum(r["het_evidence"] for r in trusted
                     if r["backbone"] == "ViT")
        ev_vgg = sum(r["het_evidence"] for r in trusted
                     if r["backbone"] == "VGG13")
        bullets.append(
            "Class-subspace heterogeneity (CPP precondition): ViT "
            f"{mean_sd(pick('ViT', 'heterogeneity'), 3)} vs VGG13 "
            f"{mean_sd(pick('VGG13', 'heterogeneity'), 3)}; evidence cells "
            f"{ev_vit}/{n_vit} vs {ev_vgg}/{n_vgg}.")
        bullets.append(
            "Residue spikes beyond the class-mean cluster: ViT "
            f"{mean_sd(pick('ViT', 'n_residue_spikes'), 1)} vs VGG13 "
            f"{mean_sd(pick('VGG13', 'n_residue_spikes'), 1)}.")
        bullets.append(
            "Post-ReLU common-mode fraction: VGG13 "
            f"{mean_sd(pick('VGG13', 'common_mode_frac'), 3)} vs ViT "
            f"{mean_sd(pick('ViT', 'common_mode_frac'), 3)}.")
    tiny = [r for r in rows if r["source"] == "tinyimagenet"]
    if tiny:
        bullets.append(
            "Class-level margins are thinnest where y_c is largest: "
            f"tinyimagenet class margin {mean_sd([r['class_margin'] for r in tiny], 1)}"
            f" vs pool {mean_sd([r['class_margin'] for r in rows], 1)}.")
    bullets.append("These are [dev] observations for calibrating and "
                   "freezing the rules; validation claims come only from "
                   "held-out cells.")
    return "\n".join(f"- {b}" for b in bullets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate X6 Tier-A measurement JSONs")
    parser.add_argument("--out_dir", type=str, default="x6_spectral/outputs",
                        help="Directory containing the per-cell JSONs")
    parser.add_argument("--arm", type=str, default="correct_only",
                        choices=("correct_only", "all", "all_standardized"),
                        help="Primary arm for the summary "
                             "(correct_only = implementation-faithful)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    json_paths = sorted(p for p in out_dir.glob("*.json")
                        if not p.name.startswith("tier_a"))
    records, errors = [], []
    for path in json_paths:
        try:
            with open(path) as fh:
                record = json.load(fh)
            if record["arms"].get(args.arm) is None:
                errors.append(f"{path.name}: arm '{args.arm}' absent")
                continue
            records.append(record)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not records:
        sys.exit(f"No usable measurement JSONs under {out_dir}")

    rows = [cell_row(record, args.arm) for record in records]
    rows.sort(key=lambda r: (group_sort_key((r["backbone"], r["source"],
                                             r["dropout"])), r["run"]))
    groups = group_rows(rows)

    csv_path = out_dir / "tier_a_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = "\n\n".join([
        "# X6 Tier-A aggregation (development pool)",
        f"Arm: `{args.arm}` (correct_only = implementation-faithful). "
        "Uncertainty: mean +- sd across independently trained runs within "
        "each (backbone, source, dropout) group, sd with ddof=1. "
        "Aggregation reads measurement JSONs only; no outcome tables.",
        "## 1. Coverage", coverage_section(records, out_dir),
        "## 2. Recovery (P6.1''): global and class-level margins",
        recovery_section(groups),
        "## 3. Stability, structure, and dials", structure_section(groups),
        "## 4. Tier-A predictions", predictions_section(groups),
        "## 5. Arm consistency (mean +- sd of per-cell deltas)",
        consistency_section(rows),
        "## 6. Flags", flags_section(rows, errors),
        "## 7. Headline reads [dev]", headline_section(rows),
        f"Runtime totals: forward "
        f"{sum(r['runtime_forward_sec'] for r in rows)/3600:.1f} h, "
        f"diagnostics {sum(r['runtime_diag_sec'] for r in rows)/3600:.1f} h "
        f"across {len(rows)} cells.",
    ]) + "\n"
    report_path = out_dir / "tier_a_report.md"
    report_path.write_text(report)
    print(f"{len(rows)} cells -> {csv_path} and {report_path}"
          + (f"; {len(errors)} problem file(s), see Flags" if errors else ""))


if __name__ == "__main__":
    main()
