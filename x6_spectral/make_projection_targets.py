"""Build the development-tier projection targets for the X6 campaign (gate 1).

Reuses the paper's own generator (projection_filtering_analysis.py: same
score files, OOD-column lists, pairing, sign convention, and Wilcoxon test)
so the Delta semantics are identical by construction, and applies the
campaign pool restriction: Conv rows filtered to the ConfidNet paradigm, ViT
rows unchanged (single paradigm). The pinned semantics live in
x6_spectral/FREEZE.md; this script is the executable form of gate 1.

Writes x6_spectral/projection_targets_dev.csv with columns
  arch, base_csf, variant, delta_augrc, significant_improvement,
  p_value, median_diff, n_total
where delta_augrc = mean of paired (base - variant) AUGRC x1000 differences
(positive = variant better) and significant_improvement requires p < 0.05
and a positive mean, exactly as in the paper's analysis.

Usage (from anywhere):
    python x6_spectral/make_projection_targets.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)

import projection_filtering_analysis as pfa

ARCH_NAME = {"Conv": "VGG13", "ViT": "ViT"}
PARADIGM_FILTER = {"Conv": "confidnet", "ViT": None}


def family_and_variant(base_method: str, variant_method: str
                       ) -> tuple[str, str]:
    """Map generator method strings onto (base_csf, variant) labels."""
    family = base_method[:-len(" global")] if base_method.endswith(" global") \
        else base_method
    assert variant_method.startswith(family + " "), (base_method,
                                                     variant_method)
    return family, variant_method[len(family) + 1:]


def main() -> None:
    out_path = CODE_DIR / "x6_spectral" / "projection_targets_dev.csv"
    rows = []
    for backbone in ("Conv", "ViT"):
        all_df = pfa.load_scores(backbone)
        paradigm = PARADIGM_FILTER[backbone]
        if paradigm is not None:
            all_df = all_df[all_df["model"] == paradigm]
        print(f"{backbone}: {len(all_df)} rows"
              + (f" (paradigm {paradigm})" if paradigm else ""))
        for base_method, variants in pfa.METHODS_OF_INTEREST.items():
            for variant_method in variants:
                result = pfa.paired_comparison(all_df, base_method,
                                               variant_method)
                if result is None:
                    continue
                family, variant = family_and_variant(base_method,
                                                     variant_method)
                rows.append({
                    "arch": ARCH_NAME[backbone],
                    "base_csf": family,
                    "variant": variant,
                    "delta_augrc": round(result["mean_diff"], 3),
                    "significant_improvement": bool(
                        result["significant"] and result["mean_diff"] > 0),
                    "p_value": f"{result['p_value']:.3e}",
                    "median_diff": round(result["median_diff"], 3),
                    "n_total": result["n_total"],
                })
    rows.sort(key=lambda r: (r["arch"], -float(r["delta_augrc"])))
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    n_sig = sum(r["significant_improvement"] for r in rows)
    print(f"{len(rows)} (family, variant) rows -> {out_path} "
          f"({n_sig} significant improvements)")


if __name__ == "__main__":
    main()
