"""
Paired AUGRC comparison of base methods vs projection-filtering variants.

For each base method and each of its projection variants, computes paired
AUGRC differences across all source × OOD dataset × paradigm × run
combinations, and tests significance with a Wilcoxon signed-rank test.

Usage:
    python projection_filtering_analysis.py --backbone Conv
    python projection_filtering_analysis.py --backbone ViT
    python projection_filtering_analysis.py  # both backbones
"""

import argparse
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

SCORES_DIR = Path("scores_risk")

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]

OOD_DATASETS = {
    "cifar10": ["cifar100", "tinyimagenet", "lsun resize", "isun", "places365", "lsun cropped", "textures", "svhn"],
    "cifar100": ["cifar10", "tinyimagenet", "lsun resize", "isun", "places365", "lsun cropped", "textures", "svhn"],
    "supercifar100": ["cifar10", "tinyimagenet", "lsun resize", "isun", "places365", "lsun cropped", "textures", "svhn"],
    "tinyimagenet": ["cifar100", "cifar10", "lsun resize", "isun", "places365", "lsun cropped", "textures", "svhn"],
}

METHODS_OF_INTEREST = {
    "CTM": ["CTM global", "CTM class", "CTM class pred", "CTM class avg"],
    "CTMmean": ["CTMmean global", "CTMmean class", "CTMmean class pred", "CTMmean class avg"],
    "fDBD": ["fDBD global", "fDBD class pred", "fDBD class avg"],
    "GradNorm": ["GradNorm global", "GradNorm class pred", "GradNorm class avg"],
    "KPCA RecError global": ["KPCA RecError class", "KPCA RecError class pred", "KPCA RecError class avg"],
    "PCA RecError global": ["PCA RecError class", "PCA RecError class pred", "PCA RecError class avg"],
    "Energy": ["Energy global", "Energy class", "Energy class pred", "Energy class avg"],
    "MLS": ["MLS global", "MLS class", "MLS class pred", "MLS class avg"],
    "MSR": ["MSR global", "MSR class", "MSR class pred", "MSR class avg"],
    "GEN": ["GEN global", "GEN class", "GEN class pred", "GEN class avg"],
    "NNGuide": ["NNGuide global", "NNGuide class pred", "NNGuide class avg"],
    "Maha": ["Maha global", "Maha class pred", "Maha class avg"],
    "pNML": ["pNML global", "pNML class pred", "pNML class avg"],
    "GE": ["GE global", "GE class", "GE class pred", "GE class avg"],
    "PCE": ["PCE global", "PCE class", "PCE class pred", "PCE class avg"],
    "PE": ["PE global", "PE class", "PE class pred", "PE class avg"],
    "REN": ["REN global", "REN class", "REN class pred", "REN class avg"],
}

MERGE_KEYS = ["model", "drop out", "reward"]


def load_scores(backbone: str) -> pd.DataFrame:
    frames = []
    for src in SOURCES:
        path = SCORES_DIR / f"scores_AUGRC_MCD-False_{backbone}_{src}.csv"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df["source"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def paired_comparison(all_df: pd.DataFrame, base_method: str, variant: str) -> dict | None:
    """Compute paired AUGRC differences (base - variant) across all configurations.
    Positive difference means variant is better (lower AUGRC)."""
    diffs = []
    for src in SOURCES:
        ood_cols = OOD_DATASETS[src]
        src_df = all_df[all_df["source"] == src]

        base_rows = src_df[src_df["methods"] == base_method]
        var_rows = src_df[src_df["methods"] == variant]

        if base_rows.empty or var_rows.empty:
            continue

        merged = base_rows.merge(var_rows, on=MERGE_KEYS, suffixes=("_base", "_var"))

        for col in ood_cols:
            col_base = f"{col}_base"
            col_var = f"{col}_var"
            if col_base in merged.columns and col_var in merged.columns:
                d = merged[col_base].values - merged[col_var].values
                diffs.extend(d[~np.isnan(d)])

    diffs = np.array(diffs)
    if len(diffs) < 5:
        return None

    n_better = int(np.sum(diffs > 0))
    n_worse = int(np.sum(diffs < 0))
    n_tied = int(np.sum(diffs == 0))

    try:
        stat, pval = wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        pval = 1.0

    return {
        "base": base_method,
        "variant": variant,
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "std_diff": float(np.std(diffs)),
        "n_better": n_better,
        "n_worse": n_worse,
        "n_tied": n_tied,
        "n_total": len(diffs),
        "p_value": float(pval),
        "significant": pval < 0.05,
    }


def run_analysis(backbone: str) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"  {backbone}")
    print(f"{'='*80}")

    all_df = load_scores(backbone)
    available_methods = set(all_df["methods"].unique())

    results = []
    for base_method, variants in METHODS_OF_INTEREST.items():
        if base_method not in available_methods:
            continue

        for variant in variants:
            if variant not in available_methods:
                continue

            result = paired_comparison(all_df, base_method, variant)
            if result is None:
                continue

            result["backbone"] = backbone
            results.append(result)

            sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
            direction = "variant better" if result["mean_diff"] > 0 else "base better"
            print(
                f"  {base_method:25s} vs {variant:30s}  "
                f"mean={result['mean_diff']:+.3f}  median={result['median_diff']:+.3f}  "
                f"win/loss={result['n_better']}/{result['n_worse']} of {result['n_total']}  "
                f"p={result['p_value']:.4f} {sig}  ({direction})"
            )

    return pd.DataFrame(results)


def find_best_variants(results_df: pd.DataFrame) -> pd.DataFrame:
    """For each base method, find the best variant (largest significant mean improvement)."""
    sig = results_df[results_df["significant"] & (results_df["mean_diff"] > 0)].copy()
    if sig.empty:
        return pd.DataFrame()
    best = sig.loc[sig.groupby(["backbone", "base"])["mean_diff"].idxmax()]
    return best[["backbone", "base", "variant", "mean_diff", "median_diff", "n_better", "n_worse", "n_total", "p_value"]].sort_values(
        ["backbone", "mean_diff"], ascending=[True, False]
    )


def main():
    parser = argparse.ArgumentParser(description="Paired AUGRC analysis: base vs projection variants")
    parser.add_argument("--backbone", type=str, choices=["Conv", "ViT"], default=None,
                        help="Backbone to analyze (default: both)")
    parser.add_argument("--output-dir", type=str, default="projection_analysis_outputs",
                        help="Output directory for CSV results")
    args = parser.parse_args()

    backbones = [args.backbone] if args.backbone else ["Conv", "ViT"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    all_results = []
    for backbone in backbones:
        results_df = run_analysis(backbone)
        all_results.append(results_df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_dir / "projection_paired_all.csv", index=False)
    print(f"\nFull results saved to {output_dir / 'projection_paired_all.csv'}")

    best = find_best_variants(combined)
    if not best.empty:
        best.to_csv(output_dir / "projection_paired_best.csv", index=False)
        print(f"Best variants saved to {output_dir / 'projection_paired_best.csv'}")

        print(f"\n{'='*80}")
        print("  SUMMARY: Methods with significant improvement from projection filtering")
        print(f"{'='*80}")
        for _, row in best.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
            print(
                f"  {row['backbone']:5s}  {row['base']:25s} → {row['variant']:30s}  "
                f"ΔAUGRC={row['mean_diff']:+.3f}  win/loss={int(row['n_better'])}/{int(row['n_worse'])}  "
                f"p={row['p_value']:.4f} {sig}"
            )
    else:
        print("\nNo significant improvements found.")


if __name__ == "__main__":
    main()
