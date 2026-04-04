"""
Generate CLIP distance CSVs with OpenOOD-style binary (near/far) grouping.

Reads the existing clip_distances_{source}.csv files from --input-dir,
replaces the k-means group column with a fixed near/far assignment
that mirrors OpenOOD v1.5 (Zhang et al., 2024, Table 1), and writes
new CSVs to --output-dir.

Only the intersection of OOD datasets shared between OpenOOD and our
benchmark is retained:
  CIFAR-10       : Near = {CIFAR-100, TinyImageNet}  Far = {SVHN, Textures, Places365}
  CIFAR-100      : Near = {CIFAR-10, TinyImageNet}   Far = {SVHN, Textures, Places365}
  SuperCIFAR-100 : Near = {CIFAR-10, TinyImageNet}   Far = {SVHN, Textures, Places365}

TinyImageNet is excluded as a source because OpenOOD does not define
near/far groupings for it.

Usage:
    python generate_openood_grouping.py \
        --input-dir clip_scores \
        --output-dir clip_scores_openood
"""
import os
import argparse
import pandas as pd
from loguru import logger


# OpenOOD-style binary grouping using the intersection of datasets
# present in both OpenOOD and our benchmark.
OPENOOD_NEAR = {
    "cifar10":       {"cifar100", "tinyimagenet"},
    "cifar100":      {"cifar10", "tinyimagenet"},
    "supercifar100": {"cifar10", "tinyimagenet"},
}

OPENOOD_FAR = {
    "cifar10":       {"svhn", "textures", "places365"},
    "cifar100":      {"svhn", "textures", "places365"},
    "supercifar100": {"svhn", "textures", "places365"},
}

SOURCES = ["cifar10", "cifar100", "supercifar100"]


def _normalize(name):
    """Lowercase, strip spaces/underscores for matching."""
    return name.replace(" ", "").replace("_", "").lower()


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

        near_set = {_normalize(n) for n in OPENOOD_NEAR[source]}
        far_set = {_normalize(n) for n in OPENOOD_FAR[source]}
        allowed = near_set | far_set | {"test"}

        def assign_group(row):
            name = row["dataset"]
            name_norm = _normalize(name)
            if name == "test":
                return "0"
            if name_norm in near_set:
                return "1"  # near
            if name_norm in far_set:
                return "3"  # far
            return None  # not in the intersection -> drop

        df["group"] = df.apply(assign_group, axis=1)

        # Drop datasets not in the OpenOOD intersection
        dropped = df[df["group"].isna()]["dataset"].tolist()
        if dropped:
            logger.info(f"  Dropping datasets not in OpenOOD intersection: {dropped}")
        df = df[df["group"].notna()].copy()

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
