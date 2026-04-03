"""
Generate CLIP distance CSVs with OpenOOD-style binary (near/far) grouping.

Reads the existing clip_distances_{source}.csv files from --input-dir,
replaces the k-means group column with a fixed near/far assignment
that mirrors OpenOOD v1.5 (Zhang et al., 2024, Table 1), and writes
new CSVs to --output-dir.

OpenOOD defines:
  CIFAR-10  : Near = {CIFAR-100, TIN}       Far = {MNIST, SVHN, Textures, Places365}
  CIFAR-100 : Near = {CIFAR-10, TIN}        Far = {MNIST, SVHN, Textures, Places365}

We extend this to our dataset pool (no MNIST; includes iSUN, LSUN variants):
  - Near-OOD datasets share the same domain (natural object images at similar
    scale) as the source.  Following OpenOOD, this is CIFAR-10/100 and
    TinyImageNet for CIFAR sources, and CIFAR-10/100 for TinyImageNet.
  - Far-OOD datasets are scene images, digit images, or texture images that
    differ in both semantics and low-level statistics: SVHN, Textures,
    Places365, iSUN, LSUN-resize, LSUN-cropped.

Usage:
    python generate_openood_grouping.py \
        --input-dir clip_scores \
        --output-dir clip_scores_openood
"""
import os
import argparse
import pandas as pd
from loguru import logger


# OpenOOD-style binary grouping: source -> set of near-OOD dataset names
# Dataset names must match the index values in the CLIP CSV (spaces, not underscores)
OPENOOD_NEAR = {
    "cifar10":       {"cifar100", "tinyimagenet"},
    "cifar100":      {"cifar10", "tinyimagenet"},
    "supercifar100": {"cifar10", "tinyimagenet"},
    "tinyimagenet":  {"cifar10", "cifar100"},
}

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenOOD-style binary near/far grouping CSVs."
    )
    parser.add_argument("--input-dir", type=str, default="clip_scores",
                        help="Directory with original clip_distances CSVs")
    parser.add_argument("--output-dir", type=str, default="clip_scores_openood",
                        help="Directory for output CSVs with binary grouping")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for source in SOURCES:
        input_path = os.path.join(args.input_dir, f"clip_distances_{source}.csv")
        if not os.path.exists(input_path):
            logger.warning(f"Missing {input_path}, skipping")
            continue

        df = pd.read_csv(input_path, header=[0, 1])
        df.columns = df.columns.droplevel(0)
        df = df.rename(columns={
            "Unnamed: 0_level_1": "dataset",
            "Unnamed: 5_level_1": "group",
        })

        near_set = OPENOOD_NEAR[source]

        def assign_group(row):
            name = row["dataset"]
            if name == "test":
                return "0"
            # Normalize: CSV index uses spaces, our sets use no underscores
            name_norm = name.replace(" ", "").lower()
            for near_name in near_set:
                if near_name.replace("_", "").lower() == name_norm:
                    return "1"  # near
            return "3"  # far (skip 2=mid to keep the same far code)

        df["group"] = df.apply(assign_group, axis=1)

        # Rebuild multi-level header to match original format
        out_df = df[["dataset", "kid mean", "fid",
                      "inv text alignment mean", "img centroid dist mean", "group"]]
        out_df.columns = pd.MultiIndex.from_tuples([
            ("", ""),
            ("global", "kid mean"),
            ("global", "fid"),
            ("class-aware", "inv text alignment mean"),
            ("class-aware", "img centroid dist mean"),
            ("group", ""),
        ])

        output_path = os.path.join(args.output_dir, f"clip_distances_{source}.csv")
        out_df.to_csv(output_path, index=False)
        logger.success(f"Saved {output_path}")

        # Log assignments
        for _, row in df.iterrows():
            name = row["dataset"]
            grp = row["group"]
            label = {"0": "ID", "1": "Near", "3": "Far"}.get(grp, grp)
            logger.info(f"  {name}: {label}")


if __name__ == "__main__":
    main()
