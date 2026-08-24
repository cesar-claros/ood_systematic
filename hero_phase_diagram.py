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
Output: Paper/ICLR_2027/figures/hero_phase_diagram.pdf (+ .png preview).
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

PAPER_FIG = CODE.parent / "Paper/ICLR_2027/figures"
VIRIDIS = matplotlib.colormaps["viridis"]
TOL = 0.01
C_MODEL, D_MODEL = 100, 512
TERTILE_S = {"strong": 120.8, "middle": 46.3, "weak": 23.6}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


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


def draw_gap(ax, ga_grid, y_grid, gap, ylabel, yscale=None):
    depth = np.clip(-gap, 0.0, 1.0)
    pcm = ax.pcolormesh(ga_grid, y_grid, depth, cmap="viridis_r",
                        vmin=0, vmax=1, shading="nearest",
                        rasterized=True)
    cs = ax.contour(ga_grid, y_grid, gap, levels=[-0.5, -0.1, -TOL],
                    colors="black", linewidths=[0.5, 0.5, 1.3])
    ax.clabel(cs, fontsize=5.5,
              fmt={-0.5: "-0.5", -0.1: "-0.1", -TOL: "boundary"})
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"$\gamma a$ (OOD norm ratio $\times$ alignment)")
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale)
    return pcm


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
    ax.legend(frameon=False, fontsize=7, loc="lower right")


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

    fig = plt.figure(figsize=(11.6, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.0, 0.05, 1.3],
                          wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 3])

    pcm = draw_gap(ax_a, ga_grid, s_grid, gap_a, "SNR $s$", yscale="log")
    for name, s_val in TERTILE_S.items():
        ax_a.axhline(s_val, color="white", linewidth=0.8,
                     linestyle="--", alpha=0.9)
        ax_a.annotate(f"{name} tertile", xy=(0.24, s_val * 1.06),
                      fontsize=6.0, ha="left", color="white")
    ax_a.set_title("A. Energy vs CTM over $(\\gamma a, s)$",
                   fontsize=8.5, loc="left")

    draw_gap(ax_b, ga_grid, theta_grid, gap_b,
             r"self-duality angle $\theta_w$ (deg)")
    ax_b.set_title("B. MLS vs Mahalanobis over "
                   "$(\\gamma a, \\theta_w)$, $s{=}24$",
                   fontsize=8.5, loc="left")

    print("panel C (empirical) ...")
    panel_c(ax_c)
    ax_c.set_title("C. 280 checkpoints, by collapse tertile",
                   fontsize=8.5, loc="left")

    cbar = fig.colorbar(pcm, cax=ax_cb)
    cbar.ax.set_title("advantage\n(AUROC)", fontsize=6.8, pad=6)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    fig.suptitle("Detector selection as a phase diagram: certified "
                 "pairwise boundaries (A, B; calibrated formulas, "
                 "audited tolerance 0.01) and the empirical transition "
                 "they organize (C)", fontsize=9, y=1.05)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG / "hero_phase_diagram.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIG / "hero_phase_diagram.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PAPER_FIG / 'hero_phase_diagram.pdf'} (+.png)")


if __name__ == "__main__":
    main()
