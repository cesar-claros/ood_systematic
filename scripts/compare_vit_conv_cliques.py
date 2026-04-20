"""Compare pooled top cliques between Conv (VGG-13) and ViT backbones.

Reads the two stats_eval.py outputs and reports per-cell Jaccard overlap / distance,
plus aggregate stats. Answers the descriptive question: do the two architectures
agree on which CSFs form the top rank-equivalent clique in each (source, regime) cell?
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
VIT_JSON = REPO / "ood_eval_outputs" / "vit_cliques" / "top_cliques_ViT_False_RC_cliques.json"
CONV_JSON = REPO / "ood_eval_outputs" / "conv_cliques_pooled" / "top_cliques_Conv_False_RC_cliques.json"
OUT_DIR = REPO / "ood_eval_outputs" / "vit_cliques"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
REGIMES = ["test", "near", "mid", "far", "all"]


def jaccard(a: list[str], b: list[str]) -> tuple[float, float]:
    sa, sb = set(a), set(b)
    if not (sa | sb):
        return float("nan"), float("nan")
    sim = len(sa & sb) / len(sa | sb)
    return sim, 1.0 - sim


def main() -> None:
    vit = json.loads(VIT_JSON.read_text())
    conv = json.loads(CONV_JSON.read_text())

    rows = []
    for src in SOURCES:
        for reg in REGIMES:
            vit_members = vit.get(src, {}).get(reg, [])
            conv_members = conv.get(src, {}).get(reg, [])
            sim, dist = jaccard(vit_members, conv_members)
            rows.append(
                {
                    "source": src,
                    "regime": reg,
                    "conv_clique": conv_members,
                    "vit_clique": vit_members,
                    "intersection": sorted(set(vit_members) & set(conv_members)),
                    "conv_only": sorted(set(conv_members) - set(vit_members)),
                    "vit_only": sorted(set(vit_members) - set(conv_members)),
                    "n_conv": len(conv_members),
                    "n_vit": len(vit_members),
                    "jaccard_sim": sim,
                    "jaccard_dist": dist,
                }
            )
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "conv_vs_vit_clique_overlap.csv"
    df.to_csv(out_csv, index=False)

    print("=== Per-cell Jaccard overlap ===")
    print(
        df[["source", "regime", "n_conv", "n_vit", "jaccard_sim", "jaccard_dist"]].to_string(index=False)
    )
    print()
    print("=== Aggregate ===")
    print(f"mean Jaccard similarity: {df['jaccard_sim'].mean():.3f}")
    print(f"median Jaccard similarity: {df['jaccard_sim'].median():.3f}")
    print(f"cells with any overlap   : {(df['jaccard_sim'] > 0).sum()} / {len(df)}")
    print(f"cells with perfect match : {(df['jaccard_sim'] == 1).sum()} / {len(df)}")
    print()
    print("=== Per-regime aggregate ===")
    agg = df.groupby("regime")["jaccard_sim"].agg(["mean", "median", "min", "max"])
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
