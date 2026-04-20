"""Scatter plot of ||ΔNC||_2 vs ΔClique for H1 (cross-paradigm) and H2 (cross-dropout)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
DELTAS = REPO / "ood_eval_outputs" / "intervention_deltas"
OUT = REPO / "ood_eval_outputs" / "intervention_mantel"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
COLORS = dict(zip(SOURCES, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))


X_COL = "delta_nc_l2"
X_LABEL = r"$\|\Delta \mathrm{NC}\|_2$  (log scale)"


def scatter(ax, df, title, marker="o", s=40):
    rho, p = spearmanr(df[X_COL], df["delta_clique"])
    for src in SOURCES:
        sub = df[df["source"] == src]
        if sub.empty:
            continue
        ax.scatter(
            sub[X_COL], sub["delta_clique"],
            c=COLORS[src], label=src, s=s, alpha=0.75,
            edgecolor="k", linewidth=0.3, marker=marker,
        )
    ax.set_xscale("log")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(r"$\Delta$Clique  (mean Jaccard dist.)")
    ax.set_title(f"{title}\n" + r"$\rho$" + f"={rho:+.3f}, p={p:.2e}, n={len(df)}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")


def main() -> None:
    h1 = pd.read_csv(DELTAS / "h1_paradigm_pairs.csv").dropna(
        subset=[X_COL, "delta_clique"]
    )
    h2 = pd.read_csv(DELTAS / "h2_dropout_pairs.csv").dropna(
        subset=[X_COL, "delta_clique"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    scatter(axes[0], h1[h1["drop_out"] == "do0"], "H1: cross-paradigm (do0)")
    scatter(axes[1], h1[h1["drop_out"] == "do1"], "H1: cross-paradigm (do1)")
    scatter(axes[2], h2, "H2: cross-dropout (do0 vs do1)", marker="D", s=60)

    plt.tight_layout()
    out_stem = OUT / "delta_nc_vs_delta_clique"
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {out_stem}.pdf and .png")


if __name__ == "__main__":
    main()
