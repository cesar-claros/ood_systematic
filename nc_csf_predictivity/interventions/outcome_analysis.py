"""Registered outcome analysis for Pilot 1 (manifest sections 5-6, Addendum A).

Consumes the per-model stats CSVs written by csf_eval
(`<exp>/analysis/stats_RW0_RF0_ASHNone_<mode>.csv`) and the committed
stage-2b sign table, and evaluates the registered endpoints on the
primary scale L = 1 - AUROC_f (AUGRC/1000 as the secondary scale):

  E1/E2/E4  paired same-seed gap deltas per (arm, OOD set), compared
            cell-wise against the COMMITTED signs on committed-material
            cells; endpoint-level one-sided t on the per-seed pooled,
            committed-direction-aligned delta.
  E3        TOST equivalence for the null scores (Maha, CTM_mean,
            PCA_RecError, Residual); margin per (score, set) = 2 x SD of
            the baseline seeds' L, computed from baseline rows only.
  E5 / X-f  Brown-Forsythe seed-variance inflation of head-side scores
            on A1- (registered) and A2 (exploratory).
  X-a       fDBD - CTM gap deltas per arm (exploratory).

Holm correction within the axis across the confirmatory family
{E1, E2, E4, E5}. Outputs outcome_report.md + .json.

Usage (from code/, after the scoring sweep):
    python nc_csf_predictivity/interventions/outcome_analysis.py \
        --stats_root $EXPERIMENT_ROOT_DIR/cifar100_intervention \
        [--stage2b nc_csf_predictivity/interventions/stage2b_predictions.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PAIRS = {"E1": ("MLS", "Maha"), "E2": ("CTM", "CTM_mean"),
         "E4": ("Energy", "MLS")}
NULL_SCORES = ("Maha", "CTM_mean", "PCA_RecError", "Residual")
HEAD_SIDE = ("MSR", "MLS", "Energy", "CTM", "fDBD")
ARM_LAMS = {"A1-": "-0.1", "A1+": "0.3", "A1++": "1.0", "A2": "hard"}
MODE_TO_SET = {
    "ood_sncs_c10": "ood_sncs", "ood_nsncs_svhn": "ood_nsncs_svhn",
    "ood_nsncs_ti": "ood_nsncs_ti",
    "ood_nsncs_lsun_cropped": "ood_nsncs_lsun_cropped",
    "ood_nsncs_lsun_resize": "ood_nsncs_lsun_resize",
    "ood_nsncs_isun": "ood_nsncs_isun",
    "ood_nsncs_textures": "ood_nsncs_textures",
    "ood_nsncs_places365": "ood_nsncs_places365",
}
STATS_TEMPLATE = "stats_RW0_RF0_ASHNone_{mode}.csv"
# The stage2b table maps score names identically except head-CTM, which is
# the plain 'CTM' variant in the stats CSVs.
STAGE2B_SCORE = {"CTM": "CTM_head"}
# The registered "PCA_RE" null (manifest section 5) is the paper-roster
# canonical PCA reconstruction error. That score is ProjectionFiltering's
# own reconstruction error and only exists as the '_global' variant (there
# is no plain PCA_RecError by construction), so the stats-CSV row name is
# canonicalized here. Fixed before any E3 outcome was examined.
CANONICAL_METHODS = {"PCA_RecError_global": "PCA_RecError"}


def load_long(stats_root: Path) -> pd.DataFrame:
    """Long table (name, lam, run, set_name, method) -> AUROC_f, AUGRC."""
    rows = []
    for exp_dir in sorted(p for p in stats_root.iterdir() if p.is_dir()):
        name = exp_dir.name
        lam = name.rsplit("_lam", 1)[-1]
        run = int(name.split("_run")[1].split("_")[0])
        for mode, set_name in MODE_TO_SET.items():
            path = exp_dir / "analysis" / STATS_TEMPLATE.format(mode=mode)
            if not path.exists():
                raise FileNotFoundError(f"missing stats CSV: {path}")
            df = pd.read_csv(path, index_col=0)
            for method, row in df.iterrows():
                method = CANONICAL_METHODS.get(method, method)
                rows.append({"name": name, "lam": lam, "run": run,
                             "set_name": set_name, "method": method,
                             "auroc_f": float(row["AUROC_f"]),
                             "augrc": float(row["AUGRC"]) / 1000.0})
    return pd.DataFrame(rows)


def _loss(table: pd.DataFrame, lam: str, run: int, set_name: str,
          method: str, scale: str) -> float:
    sel = table[(table.lam == lam) & (table.run == run)
                & (table.set_name == set_name) & (table.method == method)]
    if len(sel) != 1:
        raise ValueError(f"expected 1 row for {lam}/{run}/{set_name}/"
                         f"{method}, got {len(sel)}")
    value = float(sel.iloc[0]["auroc_f" if scale == "auroc_f" else "augrc"])
    return (1.0 - value) if scale == "auroc_f" else value


def paired_gap_deltas(table: pd.DataFrame, lam: str, set_name: str,
                      pair: tuple[str, str], runs: list[int],
                      scale: str) -> np.ndarray:
    """Per-seed (gap_arm - gap_baseline) for one (arm, set, pair)."""
    deltas = []
    for run in runs:
        g_arm = (_loss(table, lam, run, set_name, pair[0], scale)
                 - _loss(table, lam, run, set_name, pair[1], scale))
        g_base = (_loss(table, "0.0", run, set_name, pair[0], scale)
                  - _loss(table, "0.0", run, set_name, pair[1], scale))
        deltas.append(g_arm - g_base)
    return np.array(deltas)


def analyze_pairs(table: pd.DataFrame, committed: dict, runs: list[int],
                  scale: str) -> dict:
    """E1/E2/E4 cell tables, agreement counts, endpoint-level tests."""
    out: dict = {}
    for endpoint, pair in PAIRS.items():
        arms: dict = {}
        aligned_by_seed = {r: [] for r in runs}
        for label, lam in ARM_LAMS.items():
            committed_cells = committed[endpoint][label]["cells"]
            cells = {}
            agree = total = 0
            for set_name, ccell in committed_cells.items():
                deltas = paired_gap_deltas(table, lam, set_name, pair,
                                           runs, scale)
                observed_sign = int(np.sign(deltas.mean()))
                cell = {"mean_delta": float(deltas.mean()),
                        "sd_delta": float(deltas.std(ddof=1)),
                        "observed_sign": observed_sign,
                        "committed_sign": ccell["sign"],
                        "committed_material": ccell["material"]}
                if ccell["material"]:
                    total += 1
                    cell["agree"] = observed_sign == ccell["sign"]
                    agree += int(cell["agree"])
                    for run, d in zip(runs, deltas):
                        aligned_by_seed[run].append(d * ccell["sign"])
                cells[set_name] = cell
            arms[label] = {"cells": cells, "agree": agree,
                           "material": total,
                           "agreement": agree / total if total else None}
        pooled = np.array([np.mean(aligned_by_seed[r]) for r in runs])
        t_stat, p_two = sps.ttest_1samp(pooled, 0.0)
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        out[endpoint] = {
            "arms": arms,
            "pooled_aligned_mean": float(pooled.mean()),
            "pooled_aligned_sd": float(pooled.std(ddof=1)),
            "t": float(t_stat), "p_one_sided": float(p_one),
            "agreement_overall": (
                sum(a["agree"] for a in arms.values())
                / max(sum(a["material"] for a in arms.values()), 1)),
        }
    return out


def analyze_nulls(table: pd.DataFrame, runs: list[int],
                  scale: str) -> dict:
    """E3: TOST equivalence per (null score, arm); margins from baseline."""
    out: dict = {}
    set_names = sorted(table.set_name.unique())
    present = set(table.method.unique())
    for score in NULL_SCORES:
        if score not in present:
            # Registered null absent from the sweep entirely (e.g.
            # PCA_RecError before the supplementary --projections global
            # pass). Report it as missing instead of crashing; a partial
            # presence still raises in _loss (broken sweep).
            out[score] = {"status": "missing"}
            continue
        margins = {
            s: 2.0 * float(np.std(
                [_loss(table, "0.0", r, s, score, scale) for r in runs],
                ddof=1))
            for s in set_names}
        out[score] = {}
        for label, lam in ARM_LAMS.items():
            per_set = {}
            equivalent = 0
            for s in set_names:
                deltas = np.array([
                    _loss(table, lam, r, s, score, scale)
                    - _loss(table, "0.0", r, s, score, scale)
                    for r in runs])
                margin = margins[s]
                se = deltas.std(ddof=1) / np.sqrt(len(deltas))
                if se == 0:
                    tost_p = 0.0 if abs(deltas.mean()) < margin else 1.0
                else:
                    t_low = (deltas.mean() + margin) / se
                    t_high = (deltas.mean() - margin) / se
                    df = len(deltas) - 1
                    tost_p = max(1.0 - sps.t.cdf(t_low, df),
                                 sps.t.cdf(t_high, df))
                eq = tost_p < 0.05
                equivalent += int(eq)
                per_set[s] = {"mean_delta": float(deltas.mean()),
                              "margin": margin, "tost_p": float(tost_p),
                              "equivalent": bool(eq),
                              "ratio_to_margin": float(
                                  abs(deltas.mean()) / margin)
                              if margin > 0 else float("inf")}
            out[score][label] = {"per_set": per_set,
                                 "n_equivalent": equivalent,
                                 "n_sets": len(set_names)}
    return out


def analyze_variance(table: pd.DataFrame, lam: str, runs: list[int],
                     scale: str) -> dict:
    """Brown-Forsythe seed-variance comparison vs baseline, head-side."""
    set_names = sorted(table.set_name.unique())
    per_score: dict = {}
    ratios = []
    all_dev_arm, all_dev_base = [], []
    for score in HEAD_SIDE:
        devs_arm, devs_base = [], []
        for s in set_names:
            arm_vals = np.array([_loss(table, lam, r, s, score, scale)
                                 for r in runs])
            base_vals = np.array([_loss(table, "0.0", r, s, score, scale)
                                  for r in runs])
            devs_arm += list(np.abs(arm_vals - np.median(arm_vals)))
            devs_base += list(np.abs(base_vals - np.median(base_vals)))
            if base_vals.std(ddof=1) > 0:
                ratios.append(arm_vals.std(ddof=1) / base_vals.std(ddof=1))
        t_stat, p = sps.ttest_ind(devs_arm, devs_base, equal_var=False)
        per_score[score] = {"bf_t": float(t_stat), "bf_p_two": float(p),
                            "mean_abs_dev_arm": float(np.mean(devs_arm)),
                            "mean_abs_dev_base": float(np.mean(devs_base))}
        all_dev_arm += devs_arm
        all_dev_base += devs_base
    t_stat, p_two = sps.ttest_ind(all_dev_arm, all_dev_base,
                                  equal_var=False)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
    return {"per_score": per_score,
            "pooled_p_one_sided": float(p_one),
            "median_sd_ratio": float(np.median(ratios))}


def analyze_fdbd_ctm(table: pd.DataFrame, runs: list[int],
                     scale: str) -> dict:
    """X-a: paired fDBD-vs-CTM gap deltas per arm (exploratory)."""
    out = {}
    for label, lam in ARM_LAMS.items():
        per_set = {s: float(paired_gap_deltas(
            table, lam, s, ("fDBD", "CTM"), runs, scale).mean())
            for s in sorted(table.set_name.unique())}
        negatives = sum(v < 0 for v in per_set.values())
        out[label] = {"per_set": per_set,
                      "n_fdbd_gains": negatives,
                      "n_sets": len(per_set)}
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    m = len(items)
    for i, (key, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adjusted[key] = min(running, 1.0)
    return adjusted


def render(result: dict, scale: str) -> str:
    lines = [f"# Pilot 1 Registered Outcome Report (scale: {scale})", ""]
    lines.append("| endpoint | agreement on committed-material cells "
                 "| pooled aligned delta (sd) | p (one-sided) | p Holm |")
    lines.append("|---|---|---|---|---|")
    for e in ("E1", "E2", "E4"):
        r = result["pairs"][e]
        lines.append(
            f"| {e} {PAIRS[e][0]} vs {PAIRS[e][1]} "
            f"| {r['agreement_overall']:.3f} "
            f"| {r['pooled_aligned_mean']:+.4f} ({r['pooled_aligned_sd']:.4f}) "
            f"| {r['p_one_sided']:.4f} | {result['holm'][e]:.4f} |")
    e5 = result["E5"]
    lines.append(f"| E5 A1- variance inflation | median sd ratio "
                 f"{e5['median_sd_ratio']:.2f} | - "
                 f"| {e5['pooled_p_one_sided']:.4f} "
                 f"| {result['holm']['E5']:.4f} |")
    lines.append("")
    for e in ("E1", "E2", "E4"):
        lines.append(f"## {e}: {PAIRS[e][0]} vs {PAIRS[e][1]}")
        lines.append("")
        lines.append("| arm | agree/material | per-set observed delta "
                     "(committed sign) |")
        lines.append("|---|---|---|")
        for label, arm in result["pairs"][e]["arms"].items():
            per_set = "; ".join(
                f"{s.replace('ood_nsncs_', '').replace('ood_', '')} "
                f"{c['mean_delta']:+.4f}"
                f"({'+' if c['committed_sign'] > 0 else '-'}"
                f"{'*' if c['committed_material'] else ''})"
                for s, c in arm["cells"].items())
            lines.append(f"| {label} | {arm['agree']}/{arm['material']} "
                         f"| {per_set} |")
        lines.append("")
    lines.append("## E3 nulls (TOST; margin = 2 x baseline seed SD)")
    lines.append("")
    lines.append("| score | " + " | ".join(ARM_LAMS) + " |")
    lines.append("|---|" + "---|" * len(ARM_LAMS))
    missing_nulls = []
    for score, arms in result["E3"].items():
        if arms.get("status") == "missing":
            missing_nulls.append(score)
            cells = " | ".join(["MISSING"] * len(ARM_LAMS))
        else:
            cells = " | ".join(
                f"{arms[label]['n_equivalent']}/{arms[label]['n_sets']} eq"
                for label in ARM_LAMS)
        lines.append(f"| {score} | {cells} |")
    if missing_nulls:
        lines.append("")
        lines.append(
            f"**MISSING registered null(s): {', '.join(missing_nulls)}** — "
            f"not present in the sweep stats. PCA_RecError only exists as "
            f"the global-projection variant; run "
            f"`pilot1/run_pca_re_pilot1.sh`, then re-run this analysis.")
    lines.append("")
    lines.append("## Exploratory")
    lines.append("")
    xf = result["X_f"]
    lines.append(f"- X-f A2 variance inflation: median sd ratio "
                 f"{xf['median_sd_ratio']:.2f}, pooled one-sided p "
                 f"{xf['pooled_p_one_sided']:.4f}")
    for label, rec in result["X_a"].items():
        lines.append(f"- X-a fDBD-CTM ({label}): fDBD gains in "
                     f"{rec['n_fdbd_gains']}/{rec['n_sets']} sets")
    lines.append("")
    return "\n".join(lines)


def run_analysis(table: pd.DataFrame, committed: dict,
                 scale: str = "auroc_f") -> dict:
    runs = sorted(table.run.unique())
    result = {"pairs": analyze_pairs(table, committed, runs, scale),
              "E3": analyze_nulls(table, runs, scale),
              "E5": analyze_variance(table, ARM_LAMS["A1-"], runs, scale),
              "X_f": analyze_variance(table, ARM_LAMS["A2"], runs, scale),
              "X_a": analyze_fdbd_ctm(table, runs, scale)}
    pvals = {e: result["pairs"][e]["p_one_sided"] for e in PAIRS}
    pvals["E5"] = result["E5"]["pooled_p_one_sided"]
    result["holm"] = holm(pvals)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 1 outcome analysis")
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument(
        "--stage2b", type=str,
        default="nc_csf_predictivity/interventions/stage2b_predictions.json")
    parser.add_argument(
        "--out", type=str,
        default="nc_csf_predictivity/interventions/outcome_report.md")
    args = parser.parse_args()

    table = load_long(Path(args.stats_root))
    committed = json.loads(Path(args.stage2b).read_text())
    outputs = {}
    for scale in ("auroc_f", "augrc"):
        outputs[scale] = run_analysis(table, committed, scale)
    primary = outputs["auroc_f"]
    text = render(primary, "1 - AUROC_f (primary)")
    text += "\n\n---\n\n" + render(outputs["augrc"], "AUGRC (secondary)")
    Path(args.out).write_text(text)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(outputs, indent=1, default=float))
    for e in ("E1", "E2", "E4"):
        r = primary["pairs"][e]
        print(f"{e}: agree {r['agreement_overall']:.3f} "
              f"p={r['p_one_sided']:.4f} holm={primary['holm'][e]:.4f}")
    print(f"E5: p={primary['E5']['pooled_p_one_sided']:.4f} "
          f"holm={primary['holm']['E5']:.4f}; wrote {args.out}")
    missing = [s for s, arms in primary["E3"].items()
               if arms.get("status") == "missing"]
    if missing:
        print(f"WARNING: E3 registered null(s) missing from the sweep: "
              f"{', '.join(missing)}. Run pilot1/run_pca_re_pilot1.sh and "
              f"re-run this analysis for the complete E3 verdict.")


if __name__ == "__main__":
    main()
