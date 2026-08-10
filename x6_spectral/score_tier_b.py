"""Score Tier-B predictions against cell-level projection deltas (X6).

Joins the stage-2 orientation JSONs with per-cell paired AUGRC deltas
rebuilt from the frozen gate-1 semantics (same generator module, ConfidNet
filter on Conv), at the granularity where orientation actually lives:
(arch, source, dropout-slice, OOD dataset, family, variant). Two prediction
arms are scored, both pre-registered in FREEZE.md:

  rules: per-operator signs from the r2-tierB crossing rules
         (kept / complement / logit); class avg is always -1 (Tier-A);
         class and class pred variants of raw-baselined families reuse the
         operator sign (dev class conditions all hold); RecError class-vs-
         global rows are deferred to stage 2b (per-class artifacts).
  trial: signs of the deployment-batch trial deltas (mls, energy, msr, ncc
         as the labeled Maha proxy), global variant only.

Null baselines: always +1, always -1, and per-(arch, family, variant)
majority. Cells with undetermined orientation are excluded and reported as
coverage. On the dev pool this is CALIBRATION (dev outcome tables are open
by design); held-out scoring uses the same frozen code after the tag.

Usage (from anywhere; needs scores_risk/ and outputs/orientation/):
    python x6_spectral/score_tier_b.py [--out_dir=x6_spectral/outputs]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "x6_spectral"))
os.chdir(CODE_DIR)

import pandas as pd

import projection_filtering_analysis as pfa
from aggregate_tier_a import parse_model_path
from spectra_campaign_harness import FAMILY_OPERATOR, rule_signs

RECERROR_FAMILIES = {"PCA RecError", "KPCA RecError"}
#: Legacy (r4/r5) trial mapping: AUROC deltas of our reconstructed
#: projector, global variant only. Used as fallback for pre-r7 JSONs.
TRIAL_FAMILY = {"mls": "MLS", "energy": "Energy", "msr": "MSR",
                "maha": "Maha"}
#: r7 deployed-trial mapping: batch-AUGRC deltas computed with the
#: deployed PF params (exact components, tuned k, per-class estimators),
#: variant-aware. Preferred whenever trial_deployed_mean is present.
TRIAL_FAMILY7 = {
    "mls": ("MLS", "global"), "energy": ("Energy", "global"),
    "msr": ("MSR", "global"), "maha": ("Maha", "global"),
    "gradnorm": ("GradNorm", "global"),
    "maha_cp": ("Maha", "class pred"),
    "gradnorm_cp": ("GradNorm", "class pred"),
    "recerr_cp": ("PCA RecError", "class pred"),
    "recerr_class": ("PCA RecError", "class"),
}
TRIAL7_LOOKUP = {fv: key for key, fv in TRIAL_FAMILY7.items()}
MATERIAL_DELTA = 1.0

#: Per-pool loading configuration, mirroring make_projection_targets.py:
#: (scores_dir, file backbone tag, filename suffix, paradigm filter, arch).
#: The resnet18 pool implements the pinned held-out protocol: fix-config
#: exports, model == confidnet, within-dropout-slice pairing (which the
#: merge yields naturally since fix-config retains both slices), single run
#: per cell, trial arm primary per decision r3.
POOLS: dict[str, dict] = {
    "dev": {
        "sets": [("scores_risk", "Conv", "", "confidnet", "VGG13"),
                 ("scores_risk", "ViT", "", None, "ViT")],
        "scoring_csv": "tier_b_dev_scoring.csv",
        "report_md": "tier_b_dev_report.md",
        "note": ("Dev pool: calibration only (outcome tables open by "
                 "design); held-out scoring reuses this frozen code."),
    },
    "resnet18": {
        "sets": [("scores_risk_resnet18", "Conv", "_fix-config",
                  "confidnet", "ResNet18")],
        "scoring_csv": "tier_b_heldout_resnet18_scoring.csv",
        "report_md": "tier_b_heldout_resnet18_report.md",
        "note": ("HELD-OUT pool under the frozen r3 protocol: fix-config "
                 "exports, within-dropout-slice pairing, single run per "
                 "cell (no run averaging, no checkpoint-level "
                 "uncertainty). Trial arm is primary; rule arms are the "
                 "pre-registered negative control."),
    },
}


def load_pool_scores(scores_dir: str, backbone_tag: str,
                     suffix: str) -> pd.DataFrame:
    frames = []
    for src in pfa.SOURCES:
        path = Path(scores_dir) \
            / f"scores_AUGRC_MCD-False_{backbone_tag}_{src}{suffix}.csv"
        if not path.exists():
            print(f"  warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df["source"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_cells(pool: dict) -> list[dict]:
    """Per-cell paired deltas under the frozen gate-1 semantics."""
    cells = []
    for scores_dir, tag, suffix, paradigm, arch in pool["sets"]:
        all_df = load_pool_scores(scores_dir, tag, suffix)
        if paradigm is not None:
            all_df = all_df[all_df["model"] == paradigm]
        for base_method, variants in pfa.METHODS_OF_INTEREST.items():
            family = base_method[:-len(" global")] \
                if base_method.endswith(" global") else base_method
            for variant_method in variants:
                variant = variant_method[len(family) + 1:]
                for src in pfa.SOURCES:
                    src_df = all_df[all_df["source"] == src]
                    base_rows = src_df[src_df["methods"] == base_method]
                    var_rows = src_df[src_df["methods"] == variant_method]
                    if base_rows.empty or var_rows.empty:
                        continue
                    merged = base_rows.merge(var_rows, on=pfa.MERGE_KEYS,
                                             suffixes=("_base", "_var"))
                    for _, row in merged.iterrows():
                        for ood in pfa.OOD_DATASETS[src]:
                            cb, cv = f"{ood}_base", f"{ood}_var"
                            if cb not in merged.columns:
                                continue
                            delta = row[cb] - row[cv]
                            if np.isnan(delta):
                                continue
                            cells.append({
                                "arch": arch,
                                "family": family, "variant": variant,
                                "source": src, "ood": ood,
                                "dropout": int(row["drop out"] == "do1"),
                                "delta": float(delta),
                            })
    return cells


def record_signs(rec: dict, entry: dict) -> tuple[int, int, int, int, float,
                                                  bool]:
    """Per-record majority signs, recomputed with the CURRENT rule version
    from the stored per-draw scalars (a_hat, lam_hat, logit ratio and
    visibility) whenever they are present, so rule patches such as
    r2-tierB.2 apply uniformly without re-forwarding; falls back to the
    measurement-time summary signs otherwise. Also recomputes the kept sign
    at the secondary mean-span rank q = C-1 from the stored per-draw
    tier_b_meanspan a_hat (the q90 rank saturates on real features: dev
    calibration found a_hat ~0.9 everywhere, degenerating both rules into
    constant +1 predictors)."""
    if "draws" in entry and "dim" in rec and "q_used" in rec:
        votes: dict[str, list[int]] = {"kept": [], "complement": [],
                                       "logit": [], "kept_ms": []}
        a_ms_vals: list[float] = []
        q_ms = int(rec.get("n_classes", 2)) - 1
        for draw in entry["draws"]:
            ori = draw["orientation"]
            stored = draw.get("tier_b", {})
            signs = rule_signs(ori["a_hat"], ori["lam_hat"], rec["dim"],
                               rec["q_used"],
                               logit_ratio=stored.get("logit_response_ratio"),
                               logit_visibility=stored.get(
                                   "logit_visibility"))
            if not signs["undetermined"]:
                for op in ("kept", "complement", "logit"):
                    votes[op].append(signs[op])
            ms = draw.get("tier_b_meanspan")
            if ms is not None and "a_hat" in ms:
                a_ms_vals.append(ms["a_hat"])
                signs_ms = rule_signs(ms["a_hat"], ori["lam_hat"],
                                      rec["dim"], q_ms)
                if not signs_ms["undetermined"]:
                    votes["kept_ms"].append(signs_ms["kept"])
        maj = {op: int(np.sign(sum(v))) if v else 0
               for op, v in votes.items()}
        a_ms = float(np.mean(a_ms_vals)) if a_ms_vals else float("nan")
        return (maj["kept"], maj["complement"], maj["logit"],
                maj["kept_ms"], a_ms, True)
    s = entry["summary"]
    return (s["sign_kept"], s["sign_complement"], s["sign_logit"], 0,
            float("nan"), False)


def load_orientation_groups(ori_dir: Path) -> tuple[dict, int, int]:
    """Aggregate orientation signs per (arch, source, dropout, ood)."""
    groups: dict[tuple, dict[str, list]] = {}
    n_recomputed, n_fallback = 0, 0
    for path in sorted(ori_dir.glob("*.json")):
        with open(path) as fh:
            rec = json.load(fh)
        factors = parse_model_path(rec["model_path"])
        if factors is None:
            continue
        for ood, entry in rec["datasets"].items():
            if "summary" not in entry:
                continue
            s = entry["summary"]
            kept, comp, logit, kept_ms, a_ms, recomputed = \
                record_signs(rec, entry)
            n_recomputed += recomputed
            n_fallback += not recomputed
            key = (factors["backbone"], factors["source"],
                   factors["dropout"], ood)
            g = groups.setdefault(key, {"kept": [], "complement": [],
                                        "logit": [], "kept_ms": [],
                                        "a_hat": [], "a_hat_ms": [],
                                        "trial": [], "trial7": []})
            g["kept"].append(kept)
            g["complement"].append(comp)
            g["logit"].append(logit)
            g["kept_ms"].append(kept_ms)
            g["a_hat"].append(s["a_hat_mean"])
            g["a_hat_ms"].append(a_ms)
            g["trial"].append(s["trial_mean"])
            g["trial7"].append(s.get("trial_deployed_mean", {}))
    out = {}
    for key, g in groups.items():
        trial_means = {}
        trial_aux = {}
        for name in TRIAL_FAMILY:
            vals = [t.get(f"{name}_delta") for t in g["trial"]
                    if t.get(f"{name}_delta") is not None]
            trial_means[name] = float(np.mean(vals)) if vals else None
            raws = [t.get(f"{name}_raw") for t in g["trial"]
                    if t.get(f"{name}_raw") is not None]
            globs = [t.get(f"{name}_global") for t in g["trial"]
                     if t.get(f"{name}_global") is not None]
            trial_aux[name] = {
                "raw": float(np.mean(raws)) if raws else None,
                "global": float(np.mean(globs)) if globs else None,
            }
        trial7 = {}
        for name in TRIAL_FAMILY7:
            entry = {}
            for suffix in ("augrc_delta", "augrc_raw", "augrc_var",
                           "auroc_raw", "auroc_var"):
                vals = [t.get(f"{name}_{suffix}") for t in g["trial7"]
                        if t.get(f"{name}_{suffix}") is not None]
                entry[suffix] = float(np.mean(vals)) if vals else None
            if entry["augrc_delta"] is not None:
                trial7[name] = entry
        out[key] = {
            "sign": {op: int(np.sign(sum(g[op]))) for op in
                     ("kept", "complement", "logit", "kept_ms")},
            "a_hat_ms": float(np.nanmean(g["a_hat_ms"]))
            if g["a_hat_ms"] else float("nan"),
            "agreement": {op: float(np.mean([v == np.sign(sum(g[op]))
                                             for v in g[op] if v != 0]))
                          if any(v != 0 for v in g[op]) else 0.0
                          for op in ("kept", "complement", "logit")},
            "a_hat": float(np.mean(g["a_hat"])),
            "n_runs": len(g["kept"]),
            "trial": trial_means,
            "trial_aux": trial_aux,
            "trial7": trial7,
        }
    return out, n_recomputed, n_fallback


def predict(cell: dict, group: dict) -> tuple[int | None, str]:
    """Rule-arm predicted sign for one cell, or None with a reason."""
    family, variant = cell["family"], cell["variant"]
    operator = FAMILY_OPERATOR.get(family)
    if operator is None:
        return None, "family not in operator map"
    if family in RECERROR_FAMILIES:
        return None, "deferred to stage 2b (variant vs global baseline)"
    if variant == "class avg":
        return -1, "tier-a"
    sign = group["sign"][operator]
    if sign == 0:
        return None, "orientation undetermined"
    return sign, "rule"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Tier-B predictions")
    parser.add_argument("--out_dir", type=str, default="x6_spectral/outputs")
    parser.add_argument("--pool", type=str, default="dev",
                        choices=sorted(POOLS))
    args = parser.parse_args()
    pool = POOLS[args.pool]
    out_dir = Path(args.out_dir)
    ori_dir = out_dir / "orientation"
    groups, n_recomputed, n_fallback = load_orientation_groups(ori_dir)
    if not groups:
        sys.exit(f"No orientation JSONs under {ori_dir}")
    cells = build_cells(pool)

    joined = []
    for cell in cells:
        key = (cell["arch"] if cell["arch"] != "VGG13" else "VGG13",
               cell["source"], cell["dropout"], cell["ood"])
        group = groups.get(key)
        if group is None:
            continue
        sign_true = int(np.sign(cell["delta"]))
        pred, reason = predict(cell, group)
        trial_pred = None
        trial_source = None
        trial_raw_auc, trial_global_auc, trial_delta_mean = None, None, None
        trial_base_augrc, trial_var_augrc = None, None
        key7 = TRIAL7_LOOKUP.get((cell["family"], cell["variant"]))
        entry7 = group.get("trial7", {}).get(key7) if key7 else None
        if entry7 is not None:
            trial_pred = int(np.sign(entry7["augrc_delta"]))
            trial_delta_mean = entry7["augrc_delta"]
            trial_base_augrc = entry7["augrc_raw"]
            trial_var_augrc = entry7["augrc_var"]
            trial_raw_auc = entry7["auroc_raw"]
            trial_global_auc = entry7["auroc_var"]
            trial_source = "deployed"
        else:
            for short, fam in TRIAL_FAMILY.items():
                if fam == cell["family"] and cell["variant"] == "global":
                    tv = group["trial"].get(short)
                    trial_pred = int(np.sign(tv)) if tv is not None else None
                    trial_delta_mean = tv
                    aux = group.get("trial_aux", {}).get(short, {})
                    trial_raw_auc = aux.get("raw")
                    trial_global_auc = aux.get("global")
                    if trial_pred is not None:
                        trial_source = "legacy"
        ms_sign = group["sign"].get("kept_ms", 0)
        ms_pred = None
        if (pred is not None and reason == "rule"
                and FAMILY_OPERATOR.get(cell["family"]) == "kept"
                and ms_sign != 0):
            ms_pred = ms_sign
        joined.append({**cell, "sign_true": sign_true, "rule_pred": pred,
                       "rule_basis": reason, "trial_pred": trial_pred,
                       "trial_source": trial_source,
                       "trial_raw_auc": trial_raw_auc,
                       "trial_global_auc": trial_global_auc,
                       "trial_delta_mean": trial_delta_mean,
                       "trial_base_augrc": trial_base_augrc,
                       "trial_var_augrc": trial_var_augrc,
                       "rule_ms_pred": ms_pred,
                       "a_hat": group["a_hat"],
                       "a_hat_ms": group.get("a_hat_ms", float("nan")),
                       "operator": FAMILY_OPERATOR.get(cell["family"], "?"),
                       "material": abs(cell["delta"]) > MATERIAL_DELTA})
    if not joined:
        sys.exit("Orientation groups and cells did not join; check labels")

    csv_path = out_dir / pool["scoring_csv"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)

    def accuracy(rows: list[dict], field: str) -> str:
        scored = [r for r in rows if r[field] is not None
                  and r["sign_true"] != 0]
        if not scored:
            return "n/a"
        acc = np.mean([r[field] == r["sign_true"] for r in scored])
        return f"{acc:.3f} (n={len(scored)})"

    def null_line(rows: list[dict]) -> str:
        scored = [r for r in rows if r["sign_true"] != 0]
        if not scored:
            return "n/a"
        pos = np.mean([r["sign_true"] > 0 for r in scored])
        return (f"always+1 {pos:.3f}, always-1 {1 - pos:.3f}, "
                f"majority {max(pos, 1 - pos):.3f}")

    lines = [f"# X6 Tier-B scoring ({args.pool} pool)", "",
             f"Cells joined: {len(joined)}; orientation groups: "
             f"{len(groups)}. {pool['note']} "
             f"Signs recomputed with the current rule version for "
             f"{n_recomputed} records ({n_fallback} fallback to stored "
             "summary signs).",
             ""]
    for scope_name, rows in (
            ("all cells", joined),
            ("material cells (|delta| > 1)",
             [r for r in joined if r["material"]])):
        lines.append(f"## {scope_name}")
        lines.append(f"- rule arm: {accuracy(rows, 'rule_pred')}; "
                     f"trial arm: {accuracy(rows, 'trial_pred')}")
        dep = [r for r in rows if r["trial_source"] == "deployed"]
        leg = [r for r in rows if r["trial_source"] == "legacy"]
        lines.append(f"- trial arm by source: deployed "
                     f"{accuracy(dep, 'trial_pred')}; legacy "
                     f"{accuracy(leg, 'trial_pred')}")
        lines.append(f"- nulls: {null_line(rows)}")
        for op in ("kept", "complement", "logit"):
            sub = [r for r in rows if r["operator"] == op
                   and r["rule_basis"] == "rule"]
            lines.append(f"- {op}: rule {accuracy(sub, 'rule_pred')}")
        kept_rows = [r for r in rows if r["operator"] == "kept"
                     and r["rule_basis"] == "rule"]
        lines.append(f"- kept at mean-span rank (q = C-1): "
                     f"{accuracy(kept_rows, 'rule_ms_pred')}")
        deferred = sum(1 for r in rows
                       if r["rule_basis"].startswith("deferred"))
        undet = sum(1 for r in rows
                    if r["rule_basis"] == "orientation undetermined")
        lines.append(f"- deferred (stage 2b): {deferred}; "
                     f"undetermined orientation: {undet}")
        combos = sorted({(r["family"], r["variant"]) for r in rows
                         if r["trial_pred"] is not None})
        for fam, var in combos:
            sub = [r for r in rows if r["family"] == fam
                   and r["variant"] == var and r["trial_pred"] is not None]

            def mean_of(field: str) -> float | None:
                vals = [r[field] for r in sub if r[field] is not None]
                return float(np.mean(vals)) if vals else None

            parts = []
            base_a, var_a = mean_of("trial_base_augrc"), \
                mean_of("trial_var_augrc")
            if base_a is not None:
                parts.append(f"AUGRC base {base_a:.4f} var {var_a:.4f}")
            raw_r, var_r = mean_of("trial_raw_auc"), \
                mean_of("trial_global_auc")
            if raw_r is not None:
                ceiling = " CEILING" if raw_r > 0.98 else ""
                parts.append(f"AUROC base {raw_r:.3f} var {var_r:.3f}"
                             f"{ceiling}")
            n_dep = sum(r["trial_source"] == "deployed" for r in sub)
            lines.append(f"- trial magnitudes {fam} / {var}: "
                         + "; ".join(parts)
                         + f" (deployed {n_dep}/{len(sub)})")
        lines.append("")
    report_path = out_dir / pool["report_md"]
    report_path.write_text("\n".join(lines))
    print(f"{len(joined)} cells -> {csv_path} and {report_path}")


if __name__ == "__main__":
    main()
