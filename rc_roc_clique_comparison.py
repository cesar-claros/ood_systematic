"""Compare top cliques under RC (AUGRC+AURC) vs ROC (AUROC_f+FPR@95TPR).

Answers the reviewer question of whether the top cliques identified under the
risk-coverage metrics persist under threshold-oriented detection metrics.
Reads the clique JSONs exported by `stats_eval.py --backbone ... --model ...
--metric-group {RC,ROC} --filter-methods` and reports, per (paradigm, source,
regime): Jaccard overlap, winner containment both ways, clique sizes, and the
feature-side share of members (the family-transition signal).

Run from `code/`:
  ./.venv/bin/python rc_roc_clique_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PANELS = [
    ("Conv", "confidnet"),
    ("Conv", "devries"),
    ("Conv", "dg"),
    ("ViT", "modelvit"),
]
REGIMES = ["near", "mid", "far"]
FEATURE_SIDE = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}


COMPARISONS = {
    "ROC (AUROC_f+FPR@95)": ("ood_eval_outputs_roc", "ROC"),
    "FPR@95 only": ("ood_eval_outputs_fpr", "ROC"),
}


def load_cliques(dirname: str, token: str, backbone: str, model: str) -> dict:
    """Clique JSON for one (output dir, filename token, backbone, paradigm)."""
    path = (CODE_DIR / dirname
            / f"top_cliques_{backbone}_False_{token}_{model}_cliques.json")
    with open(path) as fh:
        return json.load(fh)


def feature_share(members: list[str]) -> float:
    """Fraction of clique members on the feature side."""
    return (sum(m in FEATURE_SIDE for m in members) / len(members)
            if members else float("nan"))


def main() -> None:
    """Build the per-cell comparison table and regime-level aggregates."""
    rows = []
    for cmp_name, (dirname, token) in COMPARISONS.items():
        for backbone, model in PANELS:
            rc = load_cliques("ood_eval_outputs_rc", "RC", backbone, model)
            roc = load_cliques(dirname, token, backbone, model)
            for source in [k for k in rc if k != "_ranks"]:
                for regime in REGIMES:
                    a = rc.get(source, {}).get(regime, [])
                    b = roc.get(source, {}).get(regime, [])
                    if not a or not b:
                        continue
                    sa, sb = set(a), set(b)
                    rows.append({
                        "comparison": cmp_name,
                        "panel": f"{backbone}/{model}",
                        "source": source,
                        "regime": regime,
                        "jaccard": len(sa & sb) / len(sa | sb),
                        "rc_winner_in_roc": a[0] in sb,
                        "roc_winner_in_rc": b[0] in sa,
                        "size_rc": len(a),
                        "size_roc": len(b),
                        "featshare_rc": feature_share(a),
                        "featshare_roc": feature_share(b),
                        "rc_clique": ", ".join(a),
                        "roc_clique": ", ".join(b),
                    })
    df = pd.DataFrame(rows)

    agg = (df.groupby(["comparison", "regime"], sort=False)
           .agg(cells=("jaccard", "size"),
                mean_jaccard=("jaccard", "mean"),
                rc_winner_in_roc=("rc_winner_in_roc", "mean"),
                roc_winner_in_rc=("roc_winner_in_rc", "mean"),
                mean_size_rc=("size_rc", "mean"),
                mean_size_roc=("size_roc", "mean"),
                featshare_rc=("featshare_rc", "mean"),
                featshare_roc=("featshare_roc", "mean"))
           .round(3))
    by_panel = (df.groupby(["comparison", "panel", "regime"], sort=False)
                .agg(mean_jaccard=("jaccard", "mean"),
                     rc_winner_in_roc=("rc_winner_in_roc", "mean"),
                     featshare_rc=("featshare_rc", "mean"),
                     featshare_roc=("featshare_roc", "mean"))
                .round(3))

    print("=== Aggregate by regime ===")
    print(agg.to_string())
    print("\n=== By panel x regime ===")
    print(by_panel.to_string())

    out = CODE_DIR / "mantel_partial_outputs" / "rc_roc_clique_comparison.md"
    out.write_text(
        "# RC vs ROC top-clique agreement\n\n"
        "RC = AUGRC+AURC blocks; ROC = AUROC_f+FPR@95TPR blocks; both through "
        "the identical Friedman-Conover clique pipeline (filter-methods, "
        "alpha=0.05).\n\n## Aggregate by regime\n\n"
        + agg.to_markdown() + "\n\n## By panel x regime\n\n"
        + by_panel.to_markdown() + "\n\n## Per-cell detail\n\n"
        + df.drop(columns=["rc_clique", "roc_clique"]).round(3).to_markdown(index=False)
        + "\n\n## Cliques (RC vs ROC)\n\n"
        + df[["comparison", "panel", "source", "regime", "rc_clique",
              "roc_clique"]].to_markdown(index=False) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
