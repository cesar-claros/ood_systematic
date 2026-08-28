"""Hero phase diagram for the companion paper (required-experiments step 7:
generated ONLY from audited regions; constraints in sections/theorems.tex).

Design note (2026-08-24): winner-among-all maps are DEGENERATE under the
exact Gaussian model (Mahalanobis with true parameters is a near-oracle),
so the hero shows the certified PAIRWISE gap surfaces, which are the
paper's actual theorem layer:

  A  Analytic Energy-vs-CTM gap surface over (gamma*a, SNR s): the SAME
     pair as the empirical crossing. The material boundary (|gap| = 0.01,
     the audited tolerance) moves left and deepens as s grows: stronger
     collapse hands CTM the advantage earlier in the norm-confound axis,
     the theory analog of panel C. Dashed anchors: dictionary-mapped SNR
     (s = (C-1)/sqrt(C var_collapse)) of the empirical var-collapse
     tertiles of the 280 VGG-13 checkpoints.
  B  Analytic MLS-vs-Mahalanobis gap surface over (gamma*a, theta_w) at
     s = 24 (the intervention campaign's E1 pair): the head-score parity
     region shrinks as self-duality degrades while Mahalanobis is
     invariant (self-duality separation theorem).
  C  Empirical Energy-CTM AUGRC gap versus continuous severity by
     var-collapse tertile (280 checkpoints; PAVA curves, simultaneous
     cluster-bootstrap bands, crossings marked; ordering certified by the
     crossing robustness audit).

Both surfaces use only CALIBRATED formulas (MC audit p95 err 0.001-0.006).
MSR is excluded (tied-logit failure; abstract exception) and fDBD's
analytic proxy duplicates head-CTM, so neither is drawn analytically.
Base config: theta_w = 6 deg (A), s = 24 (B), a = 0.9 unique alignment,
rho = 1, Std(eta) = 0.1 (quenched draw seed 0), logit-scale target 10,
C = 100, D = 512, isotropic covariance.

Usage (from code/):  python hero_phase_diagram.py [--quick]
Output: paper/ICLR_2027/figures/hero_phase_diagram.pdf (+ .png preview).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from crossing_robustness_audit import (
    attach_d,
    build_cells,
    curve,
    load_severity_rows,
    make_data,
    severity_map,
    tertiles,
)
from mc_phase_audit import BASE, build_config_model
from pilot0.theory import (
    HeadContext,
    NoiseModel,
    predicted_aurocs,
    predicted_ctm_mean_auroc,
    predicted_maha_auroc,
)

PAPER_FIG = CODE.parent / "paper/ICLR_2027/figures"
VIRIDIS = matplotlib.colormaps["viridis"]
TOL = 0.01
C_MODEL, D_MODEL = 100, 512
TERTILE_S = {"strong": 120.8, "middle": 46.3, "weak": 23.6}
# Visual corrections 2026-08-26 per audit #7 section 4.4: support overlay is
# a hexbin density raster (not 2,240 dots), no figure-wide title, no
# overlapping support annotation, larger panel/axis fonts, in-panel legend
# for panel C's source markers, saturation annotation 82% -> 80% (full-pool
# E4 value).
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150,
                     "xtick.labelsize": 9, "ytick.labelsize": 9})


def analytic_pixel(model: dict) -> dict[str, float]:
    """Analytic AUROCs for the four calibrated detectors (iso covariance:
    precision is diagonal, no per-pixel pinv)."""
    ctx = HeadContext.from_head(model["w"], model["b"])
    dim = model["means"].shape[1]
    sigma = model["sigma"]
    rho = float(np.sqrt(model["cov_ood"][0, 0] / model["cov_id"][0, 0]))
    noise_id = NoiseModel.isotropic(sigma, ctx, dim)
    noise_ood = NoiseModel.isotropic(rho * sigma, ctx, dim)
    head = predicted_aurocs(model["means"], model["class_freq"], noise_id,
                            model["m_ood"], noise_ood, ctx)
    out = {"MLS": head["MLS"], "Energy": head["Energy"]}
    out["CTM"] = predicted_ctm_mean_auroc(
        model["means"], model["class_freq"], model["cov_id"],
        model["m_ood"], model["cov_ood"])
    precision = np.eye(dim) / sigma ** 2
    out["Maha"] = predicted_maha_auroc(
        model["means"], precision, model["cov_id"], model["m_ood"],
        model["cov_ood"])
    return {k: float(v) for k, v in out.items()}


def gap_map(pair: tuple[str, str], ga_grid: np.ndarray,
            y_grid: np.ndarray, y_axis: str) -> np.ndarray:
    """Analytic gap AUROC_a - AUROC_b over the plane (calibrated pair)."""
    gap = np.zeros((len(y_grid), len(ga_grid)))
    for i, y in enumerate(y_grid):
        cfg = dict(BASE)
        if y_axis == "s":
            cfg["s"] = float(y)
        else:
            cfg["theta_deg"] = float(y)
        for j, ga in enumerate(ga_grid):
            cfg["ga"] = float(ga)
            cfg_full = dict(cfg, C=C_MODEL, D=D_MODEL,
                            family="hero", cluster=None, draw=0)
            model = build_config_model(C_MODEL, D_MODEL, cfg_full, seed=0)
            au = analytic_pixel(model)
            gap[i, j] = au[pair[0]] - au[pair[1]]
    return gap


AUDIT_JSON = CODE / "nc_csf_predictivity/outputs/track1/hero_boundary_audit_report.json"


def overlay_audit(ax, panel: str) -> None:
    """R4 (audit #5): overlay the MC-audited boundary points on a panel.
    Filled circle = well-conditioned + verified (drawn sharp); open square =
    shallow-slope, read the contour as a band there."""
    if not AUDIT_JSON.exists():
        return
    import json as _json

    pts = _json.loads(AUDIT_JSON.read_text())["panels"][panel]["boundary"]
    for r in pts:
        if r["display_sharp"]:
            ax.plot(r["ga"], r["y"], marker="o", ms=2.4, mfc="white",
                    mec="black", mew=0.5, ls="none", zorder=5)
        else:
            ax.plot(r["ga"], r["y"], marker="s", ms=2.4, mfc="none",
                    mec="white", mew=0.7, ls="none", zorder=5)


def draw_gap(ax, ga_grid, y_grid, gap, ylabel, yscale=None):
    depth = np.clip(-gap, 0.0, 1.0)
    pcm = ax.pcolormesh(ga_grid, y_grid, depth, cmap="viridis_r",
                        vmin=0, vmax=1, shading="nearest",
                        rasterized=True)
    cs = ax.contour(ga_grid, y_grid, gap, levels=[-0.5, -0.1, -TOL],
                    colors="black", linewidths=[0.5, 0.5, 1.3])
    ax.clabel(cs, fontsize=6.5,
              fmt={-0.5: "-0.5", -0.1: "-0.1", -TOL: "boundary"})
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\gamma a$ (OOD norm ratio $\times$ alignment)")
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    return pcm


CELL_CACHE = CODE / ("nc_csf_predictivity/outputs/track1/"
                     "theory_cell_predictions.parquet")

# Held-out validation numbers (heldout_theory_report.md + stage2_closure E3).
HELDOUT = {
    "modes": ["checkpoint\nheld-out", "source\nheld-out"],
    "arms": ["frozen theory", "severity only", "geometry model"],
    "values": [[0.099, 0.684, 0.774], [0.099, 0.607, 0.703]],
    "majority": 0.600,
    "per_source_geometry": {"cifar10": 0.182, "cifar100": 0.878,
                            "supercifar100": 0.803, "tinyimagenet": 0.993},
    "per_source_severity": {"cifar10": 0.259, "cifar100": 0.989,
                            "supercifar100": 0.682, "tinyimagenet": 0.618},
}
# Post-hoc joint confounding audit (joint_confound_audit_report.md): macro
# balanced accuracy; the paired increment crosses zero.
POSTHOC = {"metadata": 0.947, "metadata_geometry": 0.954,
           "increment": "+0.007 [-0.001, 0.017]"}


def overlay_support(ax) -> None:
    """Audit #6 section 8 / audit #7 section 4.4 panel A: measured benchmark
    support on the analytic surface as a hexbin density raster (one count per
    checkpoint-shift cell at its measured (gamma*a, dictionary-s)
    coordinates); the support sits on the saturated side and does not
    traverse the analytic boundary. The overlay is explained in the caption,
    not by an in-panel annotation."""
    if not CELL_CACHE.exists():
        return
    import pandas as pd
    fr = pd.read_parquet(CELL_CACHE).dropna(subset=["ga", "s_dict"])
    ga = fr.ga.clip(0.2, 2.2)
    s = fr.s_dict.clip(8, 130)
    hb = ax.hexbin(ga, s, gridsize=(32, 22), yscale="log", cmap="Greys",
                   mincnt=1, alpha=0.75, linewidths=0.0, zorder=4,
                   extent=(0.2, 2.2, np.log10(8), np.log10(130)))
    # shift the gray ramp so single-count hexes render mid-gray, not white
    cmax = float(hb.get_array().max())
    hb.set_clim(-0.8 * cmax, cmax)


def panel_heldout(ax) -> None:
    """Panel C per audit #8 section 7: two labeled comparisons. Solid bars =
    the pre-specified Stage-2 test (material-cell sign accuracy); hatched
    bars = the post-hoc joint confounding audit (macro balanced accuracy)
    with the paired increment and its interval. Per-source markers keep the
    heterogeneity visible."""
    x = np.arange(2)
    width = 0.26
    colors = [VIRIDIS(0.85), VIRIDIS(0.55), VIRIDIS(0.2)]
    for k, arm in enumerate(HELDOUT["arms"]):
        vals = [HELDOUT["values"][m][k] for m in range(2)]
        ax.bar(x + (k - 1) * width, vals, width, color=colors[k], label=arm)
    # post-hoc audit group (hatched, macro balanced accuracy)
    xp = 2.15
    ax.bar(xp - width / 2, POSTHOC["metadata"], width * 0.9,
           color="white", edgecolor=VIRIDIS(0.55), hatch="///",
           linewidth=1.0)
    ax.bar(xp + width / 2, POSTHOC["metadata_geometry"], width * 0.9,
           color="white", edgecolor=VIRIDIS(0.2), hatch="///",
           linewidth=1.0)
    yb = max(POSTHOC["metadata"], POSTHOC["metadata_geometry"]) + 0.02
    ax.plot([xp - width / 2, xp - width / 2, xp + width / 2,
             xp + width / 2], [yb, yb + 0.012, yb + 0.012, yb],
            color="black", linewidth=0.7)
    ax.annotate(POSTHOC["increment"], xy=(xp, yb + 0.02), ha="center",
                fontsize=6.4)
    marks = {"cifar10": "o", "cifar100": "s", "supercifar100": "D",
             "tinyimagenet": "^"}
    from matplotlib.lines import Line2D
    handles = []
    for src, m in marks.items():
        ax.plot(1 + (1 - 1) * width, HELDOUT["per_source_severity"][src],
                marker=m, ms=3.8, mfc="white", mec="black", mew=0.6,
                ls="none", zorder=5)
        ax.plot(1 + (2 - 1) * width, HELDOUT["per_source_geometry"][src],
                marker=m, ms=3.8, mfc="white", mec="black", mew=0.6,
                ls="none", zorder=5)
        handles.append(Line2D([], [], marker=m, ms=3.8, mfc="white",
                              mec="black", mew=0.6, ls="none", label=src))
    ax.plot([-0.45, 1.45], [HELDOUT["majority"]] * 2, color="black",
            linewidth=0.7, linestyle=":", alpha=0.8)
    ax.annotate("train-fold majority", xy=(-0.42, HELDOUT["majority"] + 0.012),
                fontsize=6.8)
    ax.annotate("saturated:\n80% zero margin", xy=(-0.40, 0.13),
                fontsize=6.8)
    ax.set_xticks([0, 1, 2.15])
    ax.set_xticklabels(["checkpoint\nheld-out\n(sign acc.)",
                        "source\nheld-out\n(sign acc.)",
                        "post-hoc audit\n(macro bal.\nacc.)"], fontsize=7.5)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("held-out accuracy (material cells)")
    leg_arms = ax.legend(fontsize=6.2, loc="upper left", frameon=False,
                         bbox_to_anchor=(0.0, 1.0))
    ax.add_artist(leg_arms)
    ax.legend(handles=handles, fontsize=6.4, loc="center",
              bbox_to_anchor=(0.60, 0.38), frameon=True, framealpha=0.85,
              edgecolor="none", handletextpad=0.3, borderpad=0.3,
              title="held-out source", title_fontsize=6.4)


def panel_c(ax, b_boot: int = 500) -> None:
    import pandas as pd
    df = pd.read_parquet(
        CODE / "nc_csf_predictivity/outputs/track1/dataset/"
               "long_harmonized.parquet")
    rows = load_severity_rows()
    cells = attach_d(build_cells(df), severity_map(
        rows, ("kid", "fd", "text_align", "img_centroid")))
    strata = tertiles(cells)
    data, active, fine = make_data(cells)
    rng = np.random.default_rng(0)
    shades = {"strong": 0.15, "middle": 0.5, "weak": 0.85}
    for name in ("strong", "middle", "weak"):
        sub = [c for c in active if c in strata[name]]
        g0 = curve("pava", data, sub, fine)
        color = VIRIDIS(shades[name])
        devs = np.empty(b_boot)
        for i in range(b_boot):
            boot = list(rng.choice(sub, len(sub), replace=True))
            devs[i] = np.nanmax(np.abs(curve("pava", data, boot, fine)
                                       - g0))
        q = np.quantile(devs, 0.95)
        ax.fill_between(fine, g0 - q, g0 + q, color=color, alpha=0.12,
                        linewidth=0)
        ax.plot(fine, g0, color=color, linewidth=1.6,
                label=f"{name} collapse")
        s = np.sign(g0)
        for i in range(len(s) - 1):
            if s[i] < 0 <= s[i + 1]:
                x0 = fine[i] + (fine[i + 1] - fine[i]) * (
                    g0[i] / (g0[i] - g0[i + 1]))
                ax.plot([x0], [0.0], marker="o", color=color,
                        markersize=5, markeredgecolor="black",
                        markeredgewidth=0.5, zorder=5)
                break
    ax.axhline(0.0, color="gray", linewidth=0.7)
    ax.set_xlabel("continuous OOD severity $d$ (CLIP composite)")
    ax.set_ylabel(r"AUGRC$_{Energy}$ $-$ AUGRC$_{CTM}$")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hero phase diagram")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n_ga, n_y = (16, 10) if args.quick else (41, 26)

    ga_grid = np.linspace(0.2, 2.2, n_ga)
    s_grid = np.geomspace(8, 130, n_y)
    theta_grid = np.linspace(0, 60, n_y)

    print("panel A surface (Energy vs CTM over gamma*a x s) ...")
    gap_a = gap_map(("Energy", "CTM"), ga_grid, s_grid, "s")
    print("panel B surface (MLS vs Maha over gamma*a x theta_w) ...")
    gap_b = gap_map(("MLS", "Maha"), ga_grid, theta_grid, "theta")

    # Redesigned 2026-08-25 per audit #6 section 8: A = analytic surface with
    # the measured benchmark support overlaid (theory-versus-reality made
    # visible); B = the empirical geometry-ordered pattern; C = the held-out
    # verdict with per-source heterogeneity. The self-duality surface moves
    # to its own figure (appendix).
    fig = plt.figure(figsize=(11.6, 3.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 0.05, 1.2, 1.0],
                          wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    pcm = draw_gap(ax_a, ga_grid, s_grid, gap_a, "SNR $s$", yscale="log")
    overlay_audit(ax_a, "A")
    overlay_support(ax_a)
    for name, s_val in TERTILE_S.items():
        ax_a.axhline(s_val, color="white", linewidth=0.8,
                     linestyle="--", alpha=0.9)
        ax_a.annotate(f"{name} tertile", xy=(1.62, s_val * 1.06),
                      fontsize=7, ha="left", color="white")
    ax_a.set_title("A. Analytic surface + measured support",
                   fontsize=10, loc="left")

    print("panel B (empirical strata) ...")
    panel_c(ax_b)
    ax_b.set_title("B. 280 checkpoints, by collapse tertile",
                   fontsize=10, loc="left")

    print("panel C (held-out verdict) ...")
    panel_heldout(ax_c)
    ax_c.set_title("C. Held-out validation", fontsize=10, loc="left")

    cbar = fig.colorbar(pcm, cax=ax_cb)
    cbar.set_label("CTM advantage (AUROC)", fontsize=8)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG / "hero_phase_diagram.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIG / "hero_phase_diagram.png", bbox_inches="tight")
    plt.close(fig)

    fig2, ax_sd = plt.subplots(figsize=(3.6, 3.0))
    pcm2 = draw_gap(ax_sd, ga_grid, theta_grid, gap_b,
                    r"self-duality angle $\theta_w$ (deg)")
    overlay_audit(ax_sd, "B")
    ax_sd.set_title("MLS vs Mahalanobis over $(\\gamma a, \\theta_w)$, "
                    "$s{=}24$", fontsize=8.5, loc="left")
    cb2 = fig2.colorbar(pcm2, ax=ax_sd, fraction=0.046)
    cb2.ax.set_title("advantage\n(AUROC)", fontsize=6.4, pad=5)
    fig2.savefig(PAPER_FIG / "selfdual_panel.pdf", bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {PAPER_FIG / 'hero_phase_diagram.pdf'} (+.png) and "
          f"selfdual_panel.pdf")


if __name__ == "__main__":
    main()
