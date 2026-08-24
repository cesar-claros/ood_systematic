"""Campaign figures for the phase-diagram paper integration.

Revised per documentation/X1_X3_phase_diagram_integration_recommendations.md
(section 3 figure audit + section 4.3 compact main-text figure):

  fig1     paired intervention-response decomposition (appendix): seed
           points + mean markers, COMMON vertical scale, Delta-E1 sign
           convention stated, registered/post-hoc labeling.
  fig2     fixed-head geometry (appendix): retained; sentence-case title;
           the caption note that the three diagnostics share a log axis
           for visibility only.
  fig3     B-axis trajectories (main text via figmain): frozen spectral
           acceptance band, material-side shading, dose labels, open
           markers = adaptive round 1, filled = prospective extension,
           anomaly identified, multivariate-gate caveat.
  fig4     sensitivity of Mahalanobis response prediction to the
           covariance population (appendix): structural panel only;
           the fitted cap moves to fig4_supp with a post-hoc label.
  figmain  compact two-panel main-text figure: (A) E1 gap response by
           arm with A2 visually separated; (B) the revised B-axis panel.
  chronology  evidence table; iterated Pilot 0 gates reclassified as
           pre-intervention operator development.

Style: viridis-derived colorblind-safe palette, PDF + PNG, missing-data
figures skip with a message.

Usage (from code/):
    python nc_csf_predictivity/interventions/campaign_figures.py \
        [--figures fig1 fig2 fig3 fig4 figmain chronology] \
        [--stats_root nc_csf_predictivity/interventions/stats_local]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = Path(__file__).resolve().parent
RESPONSE_ARMS = [("-0.1", "A1-"), ("0.3", "A1+"), ("1.0", "A1++"),
                 ("hard", "A2")]
ARMS = [("-0.1", "A1-"), ("0.0", "base"), ("0.3", "A1+"),
        ("1.0", "A1++"), ("hard", "A2")]
EXT_DOSES = {"varreg": {"0.003", "0.01", "0.03"},
             "ctrreg": {"0.00003", "0.0001", "0.0003"}}
VIRIDIS = matplotlib.colormaps["viridis"]
C_TRAIN, C_VAL, C_VALC = VIRIDIS(0.15), VIRIDIS(0.55), VIRIDIS(0.85)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / name}.pdf (+.png)")


def _load_dir(path: Path, pattern: str = "*.json") -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(path.glob(pattern))
            if not p.name.endswith("FAILED.json")]


def _seed_deltas(stats_root: str):
    from nc_csf_predictivity.interventions.outcome_analysis import (
        _loss,
        load_long,
    )
    table = load_long(Path(stats_root))
    runs = sorted(table.run.unique())
    sets = sorted(table.set_name.unique())

    def delta(lam: str, run: int, method: str) -> float:
        return float(np.mean([
            _loss(table, lam, run, s, method, "auroc_f")
            - _loss(table, "0.0", run, s, method, "auroc_f")
            for s in sets]))

    return runs, delta


def _plot_gap_panel(ax, runs, delta, annotate: bool = True) -> None:
    """E1 gap response by arm; A2 visually separated (audit 4.3 panel A)."""
    xpos = {label: i + (0.6 if lam == "hard" else 0.0)
            for i, (lam, label) in enumerate(RESPONSE_ARMS)}
    ax.axvline(xpos["A1++"] + 0.8, color="gray", linewidth=0.6,
               linestyle="--", alpha=0.6)
    for lam, label in RESPONSE_ARMS:
        vals = [delta(lam, r, "MLS") - delta(lam, r, "Maha") for r in runs]
        color = VIRIDIS(0.85 if lam == "hard" else 0.30)
        x = xpos[label]
        ax.scatter([x] * len(vals), vals, s=22, color=color,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [np.mean(vals)] * 2, color=color,
                linewidth=2.0, zorder=2)
    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xticks([xpos[label] for _, label in RESPONSE_ARMS])
    ax.set_xticklabels([label for _, label in RESPONSE_ARMS])
    ax.set_ylabel(r"$\Delta$E1 gap")
    if annotate:
        ax.text(0.02, 0.03,
                "registered pooled E1: agreement 0.842, Holm p = 0.0103 "
                "(A2-dominated);\nper-arm view post hoc. "
                r"$\Delta$E1 = $\Delta L_{MLS}-\Delta L_{Maha}$, "
                r"$L=1-$AUROC$_f$; negative = MLS gains.",
                transform=ax.transAxes, fontsize=6.4, va="bottom")


def fig1(out_dir: Path, stats_root: str | None) -> None:
    if not stats_root or not Path(stats_root).is_dir():
        print("fig1 SKIPPED: --stats_root required.")
        return
    runs, delta = _seed_deltas(stats_root)
    panels = [("MLS", r"$\Delta L_{MLS}$"),
              ("Maha", r"$\Delta L_{Maha}$"),
              ("gap", r"$\Delta$E1 gap = $\Delta L_{MLS}-\Delta L_{Maha}$")]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), sharey=True)
    for ax, (method, title) in zip(axes, panels):
        for i, (lam, label) in enumerate(RESPONSE_ARMS):
            if method == "gap":
                vals = [delta(lam, r, "MLS") - delta(lam, r, "Maha")
                        for r in runs]
            else:
                vals = [delta(lam, r, method) for r in runs]
            color = VIRIDIS(0.85 if lam == "hard" else 0.30)
            ax.scatter([i] * len(vals), vals, s=18, color=color,
                       edgecolor="black", linewidth=0.4, zorder=3)
            ax.plot([i - 0.2, i + 0.2], [np.mean(vals)] * 2, color=color,
                    linewidth=2.0, zorder=2)
        ax.axhline(0.0, color="gray", linewidth=0.7)
        ax.set_xticks(range(len(RESPONSE_ARMS)))
        ax.set_xticklabels([label for _, label in RESPONSE_ARMS])
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel(r"paired response ($L=1-$AUROC$_f$)")
    fig.suptitle("Paired intervention-response decomposition (COMMON "
                 "vertical scale; dots = seeds, bar = mean; negative "
                 r"$\Delta$E1 = MLS gains; E1 registered, "
                 "decomposition post hoc)", fontsize=8.5, y=1.06)
    _save(fig, out_dir, "fig1_response_decomposition")


def fig2(out_dir: Path) -> None:
    recs = [r for r in _load_dir(BASE / "nullspace")
            if r.get("kind") in ("etfreg", "etfhard")]
    if not recs:
        print("fig2 SKIPPED: no Pilot 1 nullspace JSONs found.")
        return
    by_lam: dict[str, list[dict]] = {}
    for r in recs:
        by_lam.setdefault(r["lam"], []).append(r)
    metrics = [("self_duality_full", "self-duality (full space)", None),
               ("self_duality_proj", "self-duality (row-span projected)",
                "rank_99"),
               ("eta_perp", "nullspace energy fraction", "rank_99")]
    colors = [VIRIDIS(0.2), VIRIDIS(0.55), VIRIDIS(0.85)]
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for j, (key, label, sens) in enumerate(metrics):
        for i, (lam, _) in enumerate(ARMS):
            vals = []
            for r in by_lam.get(lam, []):
                if sens and "sensitivity" in r:
                    vals.append(r["sensitivity"][sens][
                        key if key != "self_duality_proj"
                        else "self_duality_proj"])
                else:
                    vals.append(r[key])
            x = i + (j - 1) * width
            ax.bar(x, np.mean(vals), width=width * 0.92, color=colors[j],
                   alpha=0.5, label=label if i == 0 else None, zorder=1)
            ax.scatter([x] * len(vals), vals, s=9, color=colors[j],
                       edgecolor="black", linewidth=0.3, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab in ARMS])
    ax.set_ylabel("value (log scale)")
    ax.set_title("A fixed ETF head leaves both in-span misalignment and "
                 "nullspace energy\n(exact ETF rank 99; dots = seeds; "
                 "shared axis for visibility, diagnostics are not "
                 "commensurate)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, out_dir, "fig2_fixed_head_geometry")


def _plot_baxis_panel(ax, ref: list[dict], bpts: list[dict],
                      spans: dict, compact: bool = False) -> None:
    band = spans["eig_max_over_mean"]
    ax.axhspan(band[0], band[1], color=VIRIDIS(0.5), alpha=0.10,
               zorder=0)
    base_vc = np.mean([r["var_collapse"] for r in ref
                       if r["lam"] == "0.0"])
    material_edge = base_vc - 0.0068
    ax.axvspan(0.035, material_edge, color="gray", alpha=0.08, zorder=0)
    ax.axvline(material_edge, color="gray", linestyle=":", linewidth=0.9)
    arm_shades = {"-0.1": 0.10, "0.0": 0.25, "0.3": 0.40, "1.0": 0.55}
    for r in ref:
        ax.scatter(r["var_collapse"], r["eig_max_over_mean"], s=38,
                   marker="s", color=VIRIDIS(arm_shades[r["lam"]]),
                   edgecolor="black", linewidth=0.5, zorder=4)
    ax.scatter([], [], s=38, marker="s", color=VIRIDIS(0.35),
               edgecolor="black", linewidth=0.5,
               label="baseline + A1 reference")
    styles = {"varreg": (VIRIDIS(0.70), "o", "-"),
              "ctrreg": (VIRIDIS(0.92), "^", "--")}
    for kind, (color, marker, ls) in styles.items():
        pts = [r for r in bpts if r["kind"] == kind
               and r["var_collapse"] < 0.5]
        lams = sorted({r["lam"] for r in pts}, key=float)
        med = [(np.median([p["var_collapse"] for p in pts
                           if p["lam"] == lam]),
                np.median([p["eig_max_over_mean"] for p in pts
                           if p["lam"] == lam])) for lam in lams]
        ax.plot([m[0] for m in med], [m[1] for m in med], ls,
                color=color, linewidth=1.0, zorder=2)
        for p in pts:
            prospective = p["lam"] in EXT_DOSES[kind]
            ax.scatter(p["var_collapse"], p["eig_max_over_mean"], s=22,
                       marker=marker,
                       facecolor=color if prospective else "none",
                       edgecolor=color if not prospective else "black",
                       linewidth=0.7 if not prospective else 0.4,
                       zorder=3)
        ax.scatter([], [], s=22, marker=marker, facecolor=color,
                   edgecolor="black", linewidth=0.4,
                   label=f"{kind} (filled = prospective ext.)")
        # Dose-direction labels at the gentle and strong ends.
        for lam in (lams[-1], lams[0]):
            m = med[lams.index(lam)]
            ax.annotate(f"$\\lambda$={lam}", xy=m,
                        xytext=(m[0] - 0.0015, m[1] - 0.55),
                        fontsize=5.8, color=color, ha="center")
    ax.annotate("materiality (A1++ displacement)",
                xy=(material_edge - 0.0006, 14.3), fontsize=6.4,
                color="gray", ha="right", rotation=90, va="bottom")
    n_clip = sum(1 for r in bpts if r["var_collapse"] >= 0.5)
    if n_clip:
        ax.annotate("varreg $\\lambda$=0.003 run1 anti-collapse "
                    "(vc 1.15) off-axis", xy=(0.985, 0.975),
                    xycoords="axes fraction", ha="right", va="top",
                    fontsize=6.2, color="gray")
    ax.set_xlabel("var_collapse")
    ax.set_ylabel("eig_max / mean eig (within-class)")
    if not compact:
        ax.set_title("B-axis paths leave the frozen spectral band before "
                     "material contraction\n(2-D view of the decisive "
                     "violation; the support gate is multivariate)",
                     fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.4, loc="lower right")


def _baxis_data():
    ref = [r for r in _load_dir(BASE / "geometry", "*__last.json")
           if r.get("kind") == "etfreg"
           and r.get("lam") in ("0.0", "-0.1", "0.3", "1.0")]
    bpts = _load_dir(BASE / "geometry_bpilot", "*__last.json")
    spans = json.loads(
        (BASE / "b_dose_report.json").read_text())["reference"]["spans"]
    return ref, bpts, spans


def fig3(out_dir: Path) -> None:
    try:
        ref, bpts, spans = _baxis_data()
    except FileNotFoundError as err:
        print(f"fig3 SKIPPED: {err}")
        return
    if not ref or not bpts:
        print("fig3 SKIPPED: missing reference or B-pilot records.")
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    _plot_baxis_panel(ax, ref, bpts, spans)
    _save(fig, out_dir, "fig3_baxis_trajectories")


def fig4(out_dir: Path) -> None:
    recs = _load_dir(BASE / "maha_repair")
    if not recs:
        print("fig4 SKIPPED: no maha_repair JSONs found.")
        return
    by_key = {(r["lam"], r["run"]): r for r in recs}
    cells = []
    for (lam, run), r in by_key.items():
        if lam == "0.0":
            continue
        base = by_key.get(("0.0", run))
        if base is None:
            continue
        for set_name, s in r["sets"].items():
            b = base["sets"][set_name]
            cells.append({
                "is_a2": lam == "hard",
                "obs": b["emp_auroc"] - s["emp_auroc"],
                "d_old": b["pred_old"] - s["pred_old"],
                "d_old_val": b["pred_old_val"] - s["pred_old_val"],
                "d_old_valc": b["pred_old_valc"] - s["pred_old_valc"],
                "d_min_val": b["pred_min_val"] - s["pred_min_val"],
            })

    # Paper panel: structural operators only (audit 3.4).
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ops = [("d_old", "train-fit", C_TRAIN, "o"),
           ("d_old_val", "all-validation", C_VAL, "s"),
           ("d_old_valc", "correct-filtered val", C_VALC, "^")]
    for key, label, color, marker in ops:
        a1 = [c for c in cells if not c["is_a2"]]
        a2 = [c for c in cells if c["is_a2"]]
        ax.scatter([c["obs"] for c in a1], [c[key] for c in a1], s=9,
                   marker=marker, color=color, alpha=0.45, linewidth=0)
        ax.scatter([c["obs"] for c in a2], [c[key] for c in a2], s=22,
                   marker=marker, color=color, edgecolor="black",
                   linewidth=0.3, label=label)
    lim = (-0.06, 0.42)
    ax.plot(lim, lim, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("observed paired Maha response")
    ax.set_ylabel("structural prediction")
    ax.set_title("Sensitivity of Mahalanobis response prediction to the "
                 "covariance population\n(A2 bold, A1 faint; structural "
                 "operators, no fitted parameters)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    _save(fig, out_dir, "fig4_maha_population")

    # Supplementary: the fitted cap, explicitly post hoc.
    report = json.loads((BASE / "maha_repair_report.json").read_text())
    sel = report["calibration"]["selected"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.array([c[sel["input"]] for c in cells])
    y = np.array([c["obs"] for c in cells])
    a2_mask = np.array([c["is_a2"] for c in cells])
    ax.scatter(x[~a2_mask], y[~a2_mask], s=9, color=VIRIDIS(0.35),
               alpha=0.45, linewidth=0, label="A1 cells")
    ax.scatter(x[a2_mask], y[a2_mask], s=20, color=VIRIDIS(0.8),
               edgecolor="black", linewidth=0.3, label="A2 cells")
    grid = np.linspace(min(x.min(), -0.02), x.max() * 1.05, 200)
    cap = abs(sel["params"][0])
    ax.plot(grid, cap * np.tanh(grid / cap), color="black",
            linewidth=1.1, label=f"cap A tanh(R/A), A={cap:.3f}")
    ax.plot(grid, grid, color="gray", linewidth=0.8, linestyle="--",
            label="identity")
    ax.set_xlabel(f"raw {sel['input']} response (predicted)")
    ax.set_ylabel("observed paired Maha response")
    ax.set_title("Fitted calibration: post-hoc candidate selected using "
                 "A2;\nno independent validation", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out_dir, "fig4_supp_calibration")


def figmain(out_dir: Path, stats_root: str | None) -> None:
    if not stats_root or not Path(stats_root).is_dir():
        print("figmain SKIPPED: --stats_root required for panel A.")
        return
    try:
        ref, bpts, spans = _baxis_data()
    except FileNotFoundError as err:
        print(f"figmain SKIPPED: {err}")
        return
    runs, delta = _seed_deltas(stats_root)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    _plot_gap_panel(ax_a, runs, delta)
    ax_a.set_title("A. E1 gap response by arm", fontsize=9, loc="left")
    _plot_baxis_panel(ax_b, ref, bpts, spans, compact=True)
    ax_b.set_title("B. B-axis paths vs the frozen spectral band",
                   fontsize=9, loc="left")
    _save(fig, out_dir, "figmain_intervention")


CHRONOLOGY = [
    ("2026-08-14", "Plan v2, novelty audit, Pilot 0 harness",
     "original registered"),
    ("2026-08-15/16", ("Pilot 0/0b operator gates (v1->v3, revised after "
     "inspecting Pilot 0 behavior); w_perp proposition"),
     "pre-intervention operator development"),
    ("2026-08-18", ("Manifest frozen (env, endpoints, failure rule); "
     "smoke gate 4/4"), "original registered"),
    ("2026-08-19", ("Manipulation report; A2 relabel + M1'; stage-2b sign "
     "commitment"), "outcome-blind amendment"),
    ("2026-08-19/20", ("Pilot 1 unblinding (E1 pass, E2/E4 refuted); "
     "Pilot 2 transport FAIL; reverse EXPAND"), "registered outcome"),
    ("2026-08-20", ("Audit #1; forensics (A1-only, decomposition, "
     "nullspace, A0); continue gate"), "post-hoc forensic"),
    ("2026-08-20/21", "B-axis dose search round 1; GB2 amendment",
     "adaptive geometry search"),
    ("2026-08-21/24", ("Decade-down extension w/ committed spectrum "
     "prediction + hard stop -> outcome C"),
     "prospective sequential extension"),
    ("2026-08-21/24", ("Maha operator repair rounds 1-2 + R2 refit "
     "(candidate only)"), "exploratory model development"),
    ("2026-08-24", ("Closure audit #2; R1-R3 robustness; claim outline "
     "frozen"), "post-hoc forensic"),
]


def chronology(out_dir: Path) -> None:
    lines = ["# Evidence chronology (campaign closure)", "",
             "| date | event | evidence class |", "|---|---|---|"]
    for date, event, cls in CHRONOLOGY:
        lines.append(f"| {date} | {event} | {cls} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence_chronology.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'evidence_chronology.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign figures")
    parser.add_argument("--figures", nargs="+",
                        default=["fig1", "fig2", "fig3", "fig4",
                                 "figmain", "chronology"])
    parser.add_argument("--stats_root", type=str, default=str(
        BASE / "stats_local"))
    parser.add_argument("--out_dir", type=str,
                        default=str(BASE / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if "fig1" in args.figures:
        fig1(out_dir, args.stats_root)
    if "fig2" in args.figures:
        fig2(out_dir)
    if "fig3" in args.figures:
        fig3(out_dir)
    if "fig4" in args.figures:
        fig4(out_dir)
    if "figmain" in args.figures:
        figmain(out_dir, args.stats_root)
    if "chronology" in args.figures:
        chronology(out_dir)


if __name__ == "__main__":
    main()
