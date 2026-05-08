"""Build long-format datasets for Track 1 and Track 2.

Track 1 (per-config rows): reads scores_all_*_fix-config files for AUGRC and
AURC, pivots eval-OOD columns long, attaches CLIP regime groupings, applies
the §5 base-CSF filter, joins per-config NC metrics on
(architecture, paradigm, source, run, dropout, reward).

Track 2 (per-(paradigm, source) rows): reads non-fix-config aggregated AUGRC
files (per-CSF best-config selection done upstream by retrieve_scores.py),
attaches regime, applies the same filter, joins NC aggregated by mean and
median across all configs/runs in each (paradigm, source) cell. ResNet18 is
absent from Track 2 because no non-fix-config file exists for it.

Outputs:
  outputs/track1/dataset/long.parquet
  outputs/track2/dataset/long.parquet
  outputs/01_build_check.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Use pyarrow directly; pandas 3.0.2's parquet engine resolution is broken."""
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent

NC_PATH = CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv"
SCORES_VGG_VIT_DIR = CODE_DIR / "scores_risk"
SCORES_R18_DIR = CODE_DIR / "scores_risk_resnet18"
CLIP_DIR = CODE_DIR / "clip_scores"
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
REGIME_MAP = {0: "test", 1: "near", 2: "mid", 3: "far"}

ARCH_LAYOUT = [
    ("VGG13", SCORES_VGG_VIT_DIR, "Conv"),
    ("ViT", SCORES_VGG_VIT_DIR, "ViT"),
    ("ResNet18", SCORES_R18_DIR, "Conv"),
]

HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}

CTM_BLACKLIST = {"CTMmean", "CTMmeanOC", "MCD-CTMmean", "MCD-CTMmeanOC"}
GLOBAL_CLASS_KEEP = {
    "KPCA RecError global", "PCA RecError global",
    "MCD-KPCA RecError global", "MCD-PCA RecError global",
}

NC_DATASET_TO_SOURCE = {"supercifar": "supercifar100"}
NC_STUDY_TO_PARADIGM = {"vit": "modelvit"}
DROP_OUT_TO_BOOL = {"do0": False, "do1": True}

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
NC_EXTRA = ["bias_collapse", "cdnv_score", "w_etf_diff", "M_etf_diff", "wM_etf_diff"]
NC_ALL = NC_PRIMARY + NC_EXTRA


def parse_reward(s) -> float:
    return float(str(s).replace("rew", ""))


def csf_passes_filter(name: str) -> bool:
    if name in CTM_BLACKLIST:
        return False
    if name.startswith("MCD-"):
        return False
    if re.search(r"global|class", name, re.IGNORECASE) and name not in GLOBAL_CLASS_KEEP:
        return False
    return True


def csf_side(name: str) -> str:
    if name in HEAD_SIDE_CSFS:
        return "head"
    if name in FEATURE_SIDE_CSFS:
        return "feature"
    return "unknown"


def load_clip_groupings(source: str) -> pd.DataFrame:
    df = pd.read_csv(CLIP_DIR / f"clip_distances_{source}.csv", header=[0, 1])
    df.columns = df.columns.droplevel(0)
    df = df.rename(columns={
        "Unnamed: 0_level_1": "eval_dataset",
        "Unnamed: 5_level_1": "group",
    })
    df["group"] = df["group"].astype(int)
    df["regime"] = df["group"].map(REGIME_MAP)
    return df[["eval_dataset", "group", "regime"]]


def load_score_long(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_cols = ["model", "drop out", "methods", "reward", "run"]
    eval_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(
        id_vars=id_cols, value_vars=eval_cols,
        var_name="eval_dataset", value_name=value_name,
    )
    long = long.rename(columns={
        "model": "paradigm",
        "drop out": "dropout_str",
        "methods": "csf",
    })
    long["dropout"] = long["dropout_str"].map(DROP_OUT_TO_BOOL)
    long["reward"] = long["reward"].apply(parse_reward)
    return long.drop(columns=["dropout_str"])


def load_score_long_aggregated(path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_cols = ["model", "drop out", "methods", "reward"]
    eval_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(
        id_vars=id_cols, value_vars=eval_cols,
        var_name="eval_dataset", value_name=value_name,
    )
    long = long.rename(columns={
        "model": "paradigm",
        "drop out": "dropout_best_str",
        "methods": "csf",
        "reward": "reward_best_str",
    })
    long["dropout_best"] = long["dropout_best_str"].map(DROP_OUT_TO_BOOL)
    long["reward_best"] = long["reward_best_str"].apply(parse_reward)
    return long.drop(columns=["dropout_best_str", "reward_best_str"])


def normalize_nc(nc: pd.DataFrame) -> pd.DataFrame:
    nc = nc.copy()
    nc["paradigm"] = nc["study"].replace(NC_STUDY_TO_PARADIGM)
    nc["source"] = nc["dataset"].replace(NC_DATASET_TO_SOURCE)
    return nc


def load_regimes() -> pd.DataFrame:
    pieces = []
    for source in SOURCES:
        clip = load_clip_groupings(source)
        clip["source"] = source
        pieces.append(clip)
    return pd.concat(pieces, ignore_index=True)


def build_track1(verbose: bool = True) -> pd.DataFrame:
    nc = normalize_nc(pd.read_csv(NC_PATH, index_col=0))

    augrc_pieces, aurc_pieces = [], []
    for arch, score_dir, bb in ARCH_LAYOUT:
        for source in SOURCES:
            for metric, bucket in [("AUGRC", augrc_pieces), ("AURC", aurc_pieces)]:
                path = score_dir / f"scores_all_{metric}_MCD-False_{bb}_{source}_fix-config.csv"
                if not path.exists():
                    if verbose:
                        print(f"  missing {path}")
                    continue
                long = load_score_long(path, metric.lower())
                long["architecture"] = arch
                long["source"] = source
                bucket.append(long)
    augrc = pd.concat(augrc_pieces, ignore_index=True)
    aurc = pd.concat(aurc_pieces, ignore_index=True)

    join_keys = [
        "architecture", "paradigm", "source", "csf",
        "reward", "run", "dropout", "eval_dataset",
    ]
    scores = augrc.merge(aurc, on=join_keys, how="outer", validate="one_to_one")

    scores = scores[scores["csf"].apply(csf_passes_filter)].copy()
    scores = scores[~((scores["paradigm"] == "modelvit") & (scores["csf"] == "Confidence"))].copy()
    scores["side"] = scores["csf"].map(csf_side)

    scores = scores.merge(load_regimes(), on=["source", "eval_dataset"],
                          how="left", validate="many_to_one")

    nc_keys = ["architecture", "paradigm", "source", "run", "dropout", "reward"]
    nc_cols = nc_keys + NC_ALL + ["lr"]
    long = scores.merge(nc[nc_cols], on=nc_keys, how="left", validate="many_to_one")
    return long


def build_track2(verbose: bool = True) -> pd.DataFrame:
    nc = normalize_nc(pd.read_csv(NC_PATH, index_col=0))

    cell_keys = ["architecture", "paradigm", "source"]
    nc_mean = nc.groupby(cell_keys)[NC_ALL].mean().reset_index()
    nc_median = nc.groupby(cell_keys)[NC_ALL].median().reset_index()
    nc_mean = nc_mean.rename(columns={c: f"nc_mean_{c}" for c in NC_ALL})
    nc_median = nc_median.rename(columns={c: f"nc_median_{c}" for c in NC_ALL})
    nc_agg = nc_mean.merge(nc_median, on=cell_keys, validate="one_to_one")

    score_pieces = []
    for arch, score_dir, bb in ARCH_LAYOUT:
        if arch == "ResNet18":
            if verbose:
                print("  skipping ResNet18 in Track 2 (no non-fix-config file)")
            continue
        for source in SOURCES:
            path = score_dir / f"scores_AUGRC_MCD-False_{bb}_{source}.csv"
            if not path.exists():
                if verbose:
                    print(f"  missing {path}")
                continue
            long = load_score_long_aggregated(path, "augrc")
            long["architecture"] = arch
            long["source"] = source
            score_pieces.append(long)
    scores = pd.concat(score_pieces, ignore_index=True)

    scores = scores[scores["csf"].apply(csf_passes_filter)].copy()
    scores = scores[~((scores["paradigm"] == "modelvit") & (scores["csf"] == "Confidence"))].copy()
    scores["side"] = scores["csf"].map(csf_side)

    scores = scores.merge(load_regimes(), on=["source", "eval_dataset"],
                          how="left", validate="many_to_one")

    long = scores.merge(nc_agg, on=cell_keys, how="left", validate="many_to_one")
    return long


def diagnose_orphan_nc(t1: pd.DataFrame) -> pd.DataFrame:
    """NC model rows that have no score match in the joined long table."""
    nc = normalize_nc(pd.read_csv(NC_PATH, index_col=0))
    nc_keys = ["architecture", "paradigm", "source", "run", "dropout", "reward"]
    nc_models = nc[nc_keys].drop_duplicates()
    long_models = t1[nc_keys].drop_duplicates()
    merged = nc_models.merge(long_models, on=nc_keys, how="left", indicator=True)
    orphans = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return orphans


def report_build_check(t1: pd.DataFrame | None, t2: pd.DataFrame | None,
                       out_path: Path) -> None:
    lines = []
    lines.append("# Step 2/3 — Build check\n\n")
    lines.append("**Date:** 2026-05-02\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/build_dataset.py`\n\n")

    if t1 is not None:
        lines.append("## Track 1 (`outputs/track1/dataset/long.parquet`)\n\n")
        lines.append(f"- Total eval rows: {len(t1):,}\n")
        n_models = t1.groupby(
            ["architecture", "paradigm", "source", "run", "dropout", "reward"]
        ).ngroups
        lines.append(f"- Unique model configurations: {n_models}\n")
        lines.append(f"- Unique CSFs after §5 filter: {t1['csf'].nunique()}\n")
        lines.append(f"- Unique eval_datasets: {t1['eval_dataset'].nunique()}\n\n")

        lines.append("### Rows per (architecture, regime)\n\n")
        g = t1.groupby(["architecture", "regime"]).size().unstack(fill_value=0)
        lines.append("```\n" + g.to_string() + "\n```\n\n")

        lines.append("### Model rows per (architecture, paradigm, source)\n\n")
        m = t1.groupby(["architecture", "paradigm", "source"]).agg(
            n_models=("run", lambda s: s.size // (
                t1[t1["architecture"] == t1["architecture"].iloc[0]]["csf"].nunique()
            )),
        )
        gm = t1.groupby(["architecture", "paradigm", "source"]).apply(
            lambda d: d.groupby(["run", "dropout", "reward"]).ngroups
        ).rename("n_model_rows")
        lines.append("```\n" + gm.to_string() + "\n```\n\n")

        lines.append("### Side breakdown\n\n")
        s = t1["side"].value_counts()
        lines.append("```\n" + s.to_string() + "\n```\n\n")

        lines.append("### NaN counts in primary 8 NC features (must be 0)\n\n")
        nan = t1[NC_PRIMARY].isna().sum()
        lines.append("```\n" + nan.to_string() + "\n```\n\n")

        lines.append("### NaN counts in AUGRC and AURC (must be 0)\n\n")
        nan2 = t1[["augrc", "aurc"]].isna().sum()
        lines.append("```\n" + nan2.to_string() + "\n```\n\n")

        lines.append("### NC rows with no score match (orphans, dropped from long table)\n\n")
        orphans = diagnose_orphan_nc(t1)
        if orphans.empty:
            lines.append("None.\n\n")
        else:
            lines.append(f"{len(orphans)} NC model rows without an AUGRC entry. "
                         "These are excluded from the long table because the "
                         "predictor cannot be evaluated on cells without an oracle CSF.\n\n")
            lines.append("```\n" + orphans.to_string(index=False) + "\n```\n\n")
            lines.append("**Implication for protocol §13 caveat 2:** the audit "
                         "noted ResNet18 dg cifar100 as having NC rewards "
                         "{2.2, 3} that VGG13 lacks. After joining with the "
                         "actual ResNet18 AUGRC file, those NC rows are orphans "
                         "(no score data exists for them). The reward grid in "
                         "the joined long table therefore aligns between VGG13 "
                         "and ResNet18 for cifar100 dg ({6, 10, 12, 15, 20}), "
                         "and the original cross-arch concern is moot.\n\n")

    if t2 is not None:
        lines.append("## Track 2 (`outputs/track2/dataset/long.parquet`)\n\n")
        lines.append(f"- Total eval rows: {len(t2):,}\n")
        n_cells = t2.groupby(["architecture", "paradigm", "source"]).ngroups
        lines.append(f"- Unique (architecture, paradigm, source) cells: {n_cells}\n")
        lines.append(f"- Unique CSFs: {t2['csf'].nunique()}\n\n")

        lines.append("### Rows per (architecture, regime)\n\n")
        g2 = t2.groupby(["architecture", "regime"]).size().unstack(fill_value=0)
        lines.append("```\n" + g2.to_string() + "\n```\n\n")

        lines.append("### NaN counts in nc_mean primary features (must be 0)\n\n")
        nan3 = t2[[f"nc_mean_{c}" for c in NC_PRIMARY]].isna().sum()
        lines.append("```\n" + nan3.to_string() + "\n```\n\n")

        lines.append("### Caveat\n\n")
        lines.append(
            "Track 2 contains **VGG13 and ViT only**. ResNet18 has no\n"
            "non-`_fix-config` aggregated AUGRC file because\n"
            "`code/retrieve_scores.py` has not been run for ResNet18 with the\n"
            "per-CSF best-config selection. Track 2 evaluation is therefore\n"
            "restricted to LOO CV within VGG13 and the ViT ablation; the\n"
            "cross-architecture transfer evaluation lives only in Track 1.\n"
        )

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", choices=["1", "2", "all"], default="all")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    t1, t2 = None, None

    if args.track in ("1", "all"):
        t1 = build_track1()
        out1 = out_root / "track1" / "dataset"
        out1.mkdir(parents=True, exist_ok=True)
        target = out1 / "long.parquet"
        write_parquet(t1, target)
        print(f"wrote {target} ({len(t1):,} rows)")

    if args.track in ("2", "all"):
        t2 = build_track2()
        out2 = out_root / "track2" / "dataset"
        out2.mkdir(parents=True, exist_ok=True)
        target = out2 / "long.parquet"
        write_parquet(t2, target)
        print(f"wrote {target} ({len(t2):,} rows)")

    report_build_check(t1, t2, out_root / "01_build_check.md")
    print(f"wrote {out_root / '01_build_check.md'}")


if __name__ == "__main__":
    main()
