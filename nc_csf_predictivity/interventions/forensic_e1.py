"""E1 forensic reanalysis (evaluation doc sections 5.1-5.4, order-of-work
steps 1, 2 and 4 of X1_X3_pilot_campaign_evaluation_and_next_steps.md).

POST HOC: nothing here can rescue or overturn the registered Pilot 1 /
Pilot 2 verdicts. The outputs are design data for the continue gate and
the fresh E1-only confirmation.

Sections of the report:

  A (5.1)  A1-only E1 reanalysis. Per-seed aligned effects over the
           committed-material cells, separately for A1-, A1+, A1++, the
           A1 pool, and A2 (reported separately, never pooled with A1).
           For each: the four seed-level values, mean, 95% t-CI, one-sided
           t-test (sensitivity), and the exact one-sided sign test (with
           four seeds its floor is 1/16 = 0.0625).
  B (5.2)  Detector-gap decomposition. Per (arm, OOD set): the paired
           change in MLS loss, in Mahalanobis loss, and the resulting gap
           change, with a per-arm attribution share telling whether the
           Mahalanobis (feature-covariance) or MLS (head) component
           carries the response.
  C (5.3)  Paired-response transport, E1 only. Observed response
           R = gap(arm) - gap(same-seed baseline) against the frozen
           plug-in's predicted response. Raw direction and magnitude per
           arm, an A1-fitted calibrated model evaluated on A2 against
           no-change / response-cell-mean / delta-nuisance comparators,
           and a within-A1 leave-one-dose-out preview (on-support).
  D (5.4)  Amplitude attribution. Component-wise predicted vs observed
           paired losses (MLS and Mahalanobis separately) for the A1 pool
           and for A2, locating the amplitude failure.

Usage (from code/, where the stats root and stage2b/geometry dirs live):
    python nc_csf_predictivity/interventions/forensic_e1.py \
        --stats_root $EXPERIMENT_ROOT_DIR/cifar100_intervention
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nc_csf_predictivity.interventions.outcome_analysis import (
    _loss,
    load_long,
)
from nc_csf_predictivity.interventions.pilot2_transport import (
    Q_FIELDS,
    fit_linear,
    load_geometry,
    load_stage2b,
    pc1_scores,
)

PAIR = ("MLS", "Maha")
ARM_LAMS = {"A1-": "-0.1", "A1+": "0.3", "A1++": "1.0", "A2": "hard"}
A1_LABELS = ("A1-", "A1+", "A1++")
SCALE = "auroc_f"  # forensic sections use the primary scale only


# ---------------------------------------------------------------------------
# Shared observed / predicted quantities (E1 only).
# ---------------------------------------------------------------------------

def obs_gap(table: pd.DataFrame, lam: str, run: int, s: str) -> float:
    return (_loss(table, lam, run, s, PAIR[0], SCALE)
            - _loss(table, lam, run, s, PAIR[1], SCALE))


def obs_response(table: pd.DataFrame, lam: str, run: int, s: str) -> float:
    return obs_gap(table, lam, run, s) - obs_gap(table, "0.0", run, s)


def pred_gap(stage2b: dict, lam: str, run: int, s: str) -> float:
    p = stage2b[(lam, run)]["sets"][s]["preds"]["emp"]
    return (1.0 - p[PAIR[0]]) - (1.0 - p[PAIR[1]])


def pred_response(stage2b: dict, lam: str, run: int, s: str) -> float:
    return pred_gap(stage2b, lam, run, s) - pred_gap(stage2b, "0.0", run, s)


def sign_test_p(values: np.ndarray) -> float:
    """Exact one-sided sign test P(#positive >= observed | p=1/2)."""
    n = len(values)
    k = int((values > 0).sum())
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


# ---------------------------------------------------------------------------
# Section A (5.1): A1-only reanalysis.
# ---------------------------------------------------------------------------

def a1_only_analysis(table: pd.DataFrame, committed_e1: dict,
                     runs: list[int]) -> dict:
    groups: dict[str, list[str]] = {lab: [lab] for lab in ARM_LAMS}
    groups["A1_pooled"] = list(A1_LABELS)
    out: dict = {}
    for gname, labels in groups.items():
        seed_vals = []
        for r in runs:
            vals = []
            for lab in labels:
                lam = ARM_LAMS[lab]
                for s, cell in committed_e1[lab]["cells"].items():
                    if not cell["material"]:
                        continue
                    vals.append(obs_response(table, lam, r, s)
                                * cell["sign"])
            seed_vals.append(float(np.mean(vals)) if vals else float("nan"))
        arr = np.array(seed_vals)
        n = len(arr)
        mean, sd = float(arr.mean()), float(arr.std(ddof=1))
        se = sd / math.sqrt(n)
        tcrit = float(sps.t.ppf(0.975, n - 1))
        t_stat, p_two = sps.ttest_1samp(arr, 0.0)
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        agree = material = 0
        for lab in labels:
            lam = ARM_LAMS[lab]
            for s, cell in committed_e1[lab]["cells"].items():
                if not cell["material"]:
                    continue
                material += 1
                mean_d = float(np.mean(
                    [obs_response(table, lam, r, s) for r in runs]))
                agree += int(np.sign(mean_d) == cell["sign"])
        out[gname] = {
            "seed_effects": [float(v) for v in arr],
            "mean": mean, "sd": sd,
            "ci95": [mean - tcrit * se, mean + tcrit * se],
            "t_p_one_sided": float(p_one),
            "sign_test_p": sign_test_p(arr),
            "n_seeds_positive": int((arr > 0).sum()),
            "cell_agreement": f"{agree}/{material}",
        }
    return out


# ---------------------------------------------------------------------------
# Section B (5.2): decomposition into MLS and Mahalanobis responses.
# ---------------------------------------------------------------------------

def decompose(table: pd.DataFrame, runs: list[int],
              set_names: list[str]) -> dict:
    out: dict = {}
    for label, lam in ARM_LAMS.items():
        per_set = {}
        abs_mls, abs_maha = [], []
        for s in set_names:
            d_mls = np.array([
                _loss(table, lam, r, s, "MLS", SCALE)
                - _loss(table, "0.0", r, s, "MLS", SCALE) for r in runs])
            d_maha = np.array([
                _loss(table, lam, r, s, "Maha", SCALE)
                - _loss(table, "0.0", r, s, "Maha", SCALE) for r in runs])
            per_set[s] = {
                "d_mls_mean": float(d_mls.mean()),
                "d_mls_sd": float(d_mls.std(ddof=1)),
                "d_maha_mean": float(d_maha.mean()),
                "d_maha_sd": float(d_maha.std(ddof=1)),
                "d_gap_mean": float((d_mls - d_maha).mean()),
            }
            abs_mls += list(np.abs(d_mls))
            abs_maha += list(np.abs(d_maha))
        share = float(np.mean(abs_maha)
                      / (np.mean(abs_mls) + np.mean(abs_maha)))
        out[label] = {
            "per_set": per_set,
            "mean_abs_d_mls": float(np.mean(abs_mls)),
            "mean_abs_d_maha": float(np.mean(abs_maha)),
            "maha_share": share,
        }
    return out


# ---------------------------------------------------------------------------
# Section C (5.3): paired-response transport (post hoc, design-only).
# ---------------------------------------------------------------------------

def response_cells(table: pd.DataFrame, stage2b: dict, runs: list[int],
                   set_names: list[str], labels) -> pd.DataFrame:
    rows = []
    for label in labels:
        lam = ARM_LAMS[label]
        for r in runs:
            for s in set_names:
                rows.append({"label": label, "lam": lam, "run": r,
                             "set_name": s,
                             "r_obs": obs_response(table, lam, r, s),
                             "r_hat": pred_response(stage2b, lam, r, s)})
    return pd.DataFrame(rows)


def response_transport(table: pd.DataFrame, stage2b: dict,
                       geometry: dict | None, runs: list[int],
                       set_names: list[str]) -> dict:
    all_cells = response_cells(table, stage2b, runs, set_names,
                               list(ARM_LAMS))
    out: dict = {"per_arm_raw": {}}
    for label in ARM_LAMS:
        sub = all_cells[all_cells.label == label]
        agree = float(np.mean(np.sign(sub.r_hat) == np.sign(sub.r_obs)))
        out["per_arm_raw"][label] = {
            "sign_agreement": agree,
            "mae_raw": float(np.abs(sub.r_hat - sub.r_obs).mean()),
            "mae_no_change": float(np.abs(sub.r_obs).mean()),
        }

    a1 = all_cells[all_cells.label.isin(A1_LABELS)]
    a2 = all_cells[all_cells.label == "A2"]
    alpha, beta = fit_linear(a1.r_hat.values, a1.r_obs.values)
    cal = alpha + beta * a2.r_hat.values
    comps = {
        "calibrated_plugin": float(np.abs(cal - a2.r_obs.values).mean()),
        "raw_plugin": float(np.abs(a2.r_hat - a2.r_obs).mean()),
        "no_change": float(np.abs(a2.r_obs).mean()),
    }
    set_mean = a1.groupby("set_name")["r_obs"].mean()
    cellmean_pred = a2.set_name.map(set_mean).values
    comps["response_cell_mean"] = float(
        np.abs(cellmean_pred - a2.r_obs.values).mean())
    if geometry is not None:
        models = sorted(geometry)
        train_models = [m for m in models if m[0] != "hard"]
        full = np.array([[geometry[m][f] for f in Q_FIELDS]
                         for m in models])
        train = np.array([[geometry[m][f] for f in Q_FIELDS]
                          for m in train_models])
        z = dict(zip(models, pc1_scores(train, full)))
        dz = {(lam, r): z[(lam, r)] - z[("0.0", r)]
              for lam, r in models if lam != "0.0"}
        x_a1 = np.array([dz[(l, r)] for l, r in zip(a1.lam, a1.run)])
        x_a2 = np.array([dz[(l, r)] for l, r in zip(a2.lam, a2.run)])
        qa, qb = fit_linear(x_a1, a1.r_obs.values)
        comps["delta_nuisance"] = float(
            np.abs(qa + qb * x_a2 - a2.r_obs.values).mean())
    out["a1_to_a2"] = {"fit": {"alpha": alpha, "beta": beta},
                       "mae": comps,
                       "sign_agreement_calibrated": float(
                           np.mean(np.sign(cal) == np.sign(a2.r_obs)))}

    loo = {}
    for held in A1_LABELS:
        tr = a1[a1.label != held]
        te = a1[a1.label == held]
        la, lb = fit_linear(tr.r_hat.values, tr.r_obs.values)
        pred = la + lb * te.r_hat.values
        loo[held] = {
            "mae_calibrated": float(np.abs(pred - te.r_obs.values).mean()),
            "mae_no_change": float(np.abs(te.r_obs).mean()),
            "sign_agreement": float(
                np.mean(np.sign(pred) == np.sign(te.r_obs))),
        }
    out["within_a1_loo_dose"] = loo
    return out


# ---------------------------------------------------------------------------
# Section D (5.4, artifact-computable parts): amplitude attribution.
# ---------------------------------------------------------------------------

def amplitude_attribution(table: pd.DataFrame, stage2b: dict,
                          runs: list[int], set_names: list[str]) -> dict:
    out: dict = {}
    for gname, labels in (("A1", A1_LABELS), ("A2", ("A2",))):
        rec: dict = {}
        for score in PAIR:
            obs, pred = [], []
            for label in labels:
                lam = ARM_LAMS[label]
                for r in runs:
                    for s in set_names:
                        obs.append(
                            _loss(table, lam, r, s, score, SCALE)
                            - _loss(table, "0.0", r, s, score, SCALE))
                        p_arm = stage2b[(lam, r)]["sets"][s]["preds"]["emp"]
                        p_base = stage2b[("0.0", r)]["sets"][s]["preds"]["emp"]
                        pred.append((1.0 - p_arm[score])
                                    - (1.0 - p_base[score]))
            obs_a, pred_a = np.array(obs), np.array(pred)
            rec[score] = {
                "mean_obs": float(obs_a.mean()),
                "mean_pred": float(pred_a.mean()),
                "mae_pred": float(np.abs(pred_a - obs_a).mean()),
            }
        total = rec["MLS"]["mae_pred"] + rec["Maha"]["mae_pred"]
        rec["maha_error_share"] = (rec["Maha"]["mae_pred"] / total
                                   if total > 0 else float("nan"))
        out[gname] = rec
    return out


# ---------------------------------------------------------------------------
# Rendering + CLI.
# ---------------------------------------------------------------------------

def render(result: dict) -> str:
    lines = ["# E1 Forensic Reanalysis (post hoc, design data only)", ""]
    lines.append("Scale: L = 1 - AUROC_f. Nothing here amends the "
                 "registered Pilot 1 / Pilot 2 verdicts (evaluation doc "
                 "section 5.3).")
    lines.append("")
    lines.append("## A. A1-only E1 (5.1)")
    lines.append("")
    lines.append("| group | seed effects | mean [95% CI] | t p (one-sided, "
                 "sensitivity) | sign test p | cells agree |")
    lines.append("|---|---|---|---|---|---|")
    for g, r in result["a1_only"].items():
        seeds = ", ".join(f"{v:+.4f}" for v in r["seed_effects"])
        lines.append(
            f"| {g} | {seeds} | {r['mean']:+.4f} "
            f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
            f"| {r['t_p_one_sided']:.4f} | {r['sign_test_p']:.4f} "
            f"| {r['cell_agreement']} |")
    lines.append("")
    lines.append("A2 is listed for reference only and is never pooled "
                 "with A1. The sign-test floor at four seeds is 0.0625.")
    lines.append("")
    lines.append("## B. MLS / Mahalanobis decomposition (5.2)")
    lines.append("")
    lines.append("| arm | mean |dL_MLS| | mean |dL_Maha| | Maha share "
                 "| reading |")
    lines.append("|---|---|---|---|---|")
    for label, r in result["decompose"].items():
        reading = ("feature/covariance channel dominates"
                   if r["maha_share"] > 0.5 else "head channel dominates")
        lines.append(f"| {label} | {r['mean_abs_d_mls']:.4f} "
                     f"| {r['mean_abs_d_maha']:.4f} "
                     f"| {r['maha_share']:.2f} | {reading} |")
    lines.append("")
    lines.append("## C. Paired-response transport (5.3, POST HOC)")
    lines.append("")
    lines.append("| arm | sign agreement (raw) | MAE raw | MAE no-change |")
    lines.append("|---|---|---|---|")
    for label, r in result["response_transport"]["per_arm_raw"].items():
        lines.append(f"| {label} | {r['sign_agreement']:.3f} "
                     f"| {r['mae_raw']:.4f} | {r['mae_no_change']:.4f} |")
    a12 = result["response_transport"]["a1_to_a2"]
    lines.append("")
    mae_line = "; ".join(f"{k} {v:.4f}" for k, v in a12["mae"].items())
    lines.append(f"- A1-fitted -> A2 (alpha {a12['fit']['alpha']:+.4f}, "
                 f"beta {a12['fit']['beta']:+.3f}): {mae_line}; calibrated "
                 f"sign agreement {a12['sign_agreement_calibrated']:.3f}")
    lines.append("- Within-A1 leave-one-dose-out (on-support preview):")
    for held, r in result["response_transport"]["within_a1_loo_dose"].items():
        lines.append(f"  - hold out {held}: MAE {r['mae_calibrated']:.4f} "
                     f"vs no-change {r['mae_no_change']:.4f}, sign "
                     f"{r['sign_agreement']:.3f}")
    lines.append("")
    lines.append("## D. Amplitude attribution (5.4, artifact-computable "
                 "parts)")
    lines.append("")
    lines.append("| group | component | mean obs dL | mean pred dL "
                 "| MAE pred |")
    lines.append("|---|---|---|---|---|")
    for g, rec in result["amplitude"].items():
        for score in PAIR:
            r = rec[score]
            lines.append(f"| {g} | {score} | {r['mean_obs']:+.4f} "
                         f"| {r['mean_pred']:+.4f} | {r['mae_pred']:.4f} |")
    for g, rec in result["amplitude"].items():
        lines.append(f"- {g}: Mahalanobis share of prediction error "
                     f"{rec['maha_error_share']:.2f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    base = "nc_csf_predictivity/interventions"
    parser = argparse.ArgumentParser(description="E1 forensic reanalysis")
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--stage2b_dir", type=str, default=f"{base}/stage2b")
    parser.add_argument("--geometry_dir", type=str,
                        default=f"{base}/geometry")
    parser.add_argument("--stage2b_predictions", type=str,
                        default=f"{base}/stage2b_predictions.json")
    parser.add_argument("--out", type=str,
                        default=f"{base}/forensic_e1_report.md")
    args = parser.parse_args()

    table = load_long(Path(args.stats_root))
    stage2b = load_stage2b(Path(args.stage2b_dir))
    committed_e1 = json.loads(
        Path(args.stage2b_predictions).read_text())["E1"]
    try:
        geometry = load_geometry(Path(args.geometry_dir))
    except FileNotFoundError:
        geometry = None
        print("WARNING: geometry dir not found; delta-nuisance comparator "
              "skipped.")
    runs = sorted(table.run.unique())
    set_names = sorted(stage2b[("0.0", runs[0])]["sets"])

    result = {
        "a1_only": a1_only_analysis(table, committed_e1, runs),
        "decompose": decompose(table, runs, set_names),
        "response_transport": response_transport(table, stage2b, geometry,
                                                 runs, set_names),
        "amplitude": amplitude_attribution(table, stage2b, runs, set_names),
    }
    Path(args.out).write_text(render(result))
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(result, indent=1, default=float))
    a1 = result["a1_only"]["A1_pooled"]
    print(f"A1-only pooled: mean {a1['mean']:+.4f}, t p {a1['t_p_one_sided']:.4f}, "
          f"sign p {a1['sign_test_p']:.4f}, cells {a1['cell_agreement']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
