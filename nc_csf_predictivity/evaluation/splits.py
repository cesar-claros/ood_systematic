"""Step 9: Train/test split definitions.

Five primary splits from protocol §9 plus two ablations. Splits operate at
MODEL row level (Track 1) or CELL level (Track 2) — never at eval-row level —
so a (model, eval_dataset) pair never leaks across train/test.

Track 1 model row = (architecture, paradigm, source, run, dropout, reward).
Track 2 cell      = (architecture, paradigm, source).

Splits:
  xarch          (Track 1, primary headline)
                 train: VGG13 CNN paradigms
                 test : ResNet18 CNN paradigms

  lopo           (Track 1, headline) — leave-one-paradigm-out, 4 folds
                 pool: VGG13 + ResNet18 + ViT, all paradigms
                 fold f: test = paradigm f rows, train = other paradigm rows
                 modelvit fold = pure CNN→Transformer test

  lodo_vgg13     (Track 1, internal CV) — leave-one-source-out on VGG13 CNN
                 4 folds, one per source

  pxs_vgg13      (Track 1, internal CV) — paradigm × source CV on VGG13 CNN
                 12 folds (3 paradigms × 4 sources)

  single_vgg13   (Track 1, diagnostic) — resubstitution
                 1 fold; train = test = all VGG13 CNN

  track2_loo     (Track 2) — leave-one-cell-out on VGG13 cells, 12 folds

  xarch_vit_in   (ablation, Track 1) — train VGG13 + ViT, test ResNet18
  lopo_cnn_only  (ablation, Track 1) — LOPO restricted to CNN paradigms (3 folds)

Outputs:
  outputs/splits/<split_name>.parquet
  outputs/07_splits_check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

CNN_PARADIGMS = ["confidnet", "devries", "dg"]
ALL_PARADIGMS = ["confidnet", "devries", "dg", "modelvit"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def make_model_id(row: pd.Series) -> str:
    return (f"{row['architecture']}|{row['paradigm']}|{row['source']}|"
            f"{int(row['run'])}|{int(bool(row['dropout']))}|{row['reward']:g}")


def make_cell_id(row: pd.Series) -> str:
    return f"{row['architecture']}|{row['paradigm']}|{row['source']}"


def load_track1_models(out_root: Path) -> pd.DataFrame:
    oracle = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    keep = ["architecture", "paradigm", "source", "run", "dropout", "reward"]
    models = oracle[keep].drop_duplicates().reset_index(drop=True)
    models["model_id"] = models.apply(make_model_id, axis=1)
    return models


def load_track2_cells(out_root: Path) -> pd.DataFrame:
    oracle = pq.read_table(out_root / "track2" / "dataset" / "oracle.parquet").to_pandas()
    keep = ["architecture", "paradigm", "source"]
    cells = oracle[keep].drop_duplicates().reset_index(drop=True)
    cells["cell_id"] = cells.apply(make_cell_id, axis=1)
    return cells


def folds_to_long(folds: list[dict], split_name: str, id_col: str) -> pd.DataFrame:
    rows = []
    for f in folds:
        for u in f["train"]:
            rows.append({"split_name": split_name, "fold_id": f["fold_id"],
                         "fold_label": f["fold_label"], "role": "train", id_col: u})
        for u in f["test"]:
            rows.append({"split_name": split_name, "fold_id": f["fold_id"],
                         "fold_label": f["fold_label"], "role": "test", id_col: u})
    return pd.DataFrame(rows)


# ---- Track 1 splits ----

def split_xarch(models: pd.DataFrame) -> list[dict]:
    train = models[(models["architecture"] == "VGG13")
                   & models["paradigm"].isin(CNN_PARADIGMS)]["model_id"].tolist()
    test = models[(models["architecture"] == "ResNet18")
                  & models["paradigm"].isin(CNN_PARADIGMS)]["model_id"].tolist()
    return [{"fold_id": 0, "fold_label": "vgg13_to_resnet18",
             "train": train, "test": test}]


def split_lopo(models: pd.DataFrame, paradigms: list[str]) -> list[dict]:
    pool = models[models["paradigm"].isin(paradigms)]
    folds = []
    for i, p in enumerate(paradigms):
        train = pool[pool["paradigm"] != p]["model_id"].tolist()
        test = pool[pool["paradigm"] == p]["model_id"].tolist()
        folds.append({"fold_id": i, "fold_label": f"lopo_{p}",
                      "train": train, "test": test})
    return folds


def split_lodo_vgg13(models: pd.DataFrame) -> list[dict]:
    pool = models[(models["architecture"] == "VGG13")
                  & models["paradigm"].isin(CNN_PARADIGMS)]
    sources = sorted(pool["source"].unique())
    folds = []
    for i, s in enumerate(sources):
        train = pool[pool["source"] != s]["model_id"].tolist()
        test = pool[pool["source"] == s]["model_id"].tolist()
        folds.append({"fold_id": i, "fold_label": f"lodo_{s}",
                      "train": train, "test": test})
    return folds


def split_pxs_vgg13(models: pd.DataFrame) -> list[dict]:
    pool = models[(models["architecture"] == "VGG13")
                  & models["paradigm"].isin(CNN_PARADIGMS)]
    folds = []
    fold_id = 0
    for p in CNN_PARADIGMS:
        for s in sorted(pool["source"].unique()):
            mask = (pool["paradigm"] == p) & (pool["source"] == s)
            train = pool[~mask]["model_id"].tolist()
            test = pool[mask]["model_id"].tolist()
            folds.append({"fold_id": fold_id,
                          "fold_label": f"pxs_{p}_{s}",
                          "train": train, "test": test})
            fold_id += 1
    return folds


def split_single_vgg13(models: pd.DataFrame) -> list[dict]:
    pool = models[(models["architecture"] == "VGG13")
                  & models["paradigm"].isin(CNN_PARADIGMS)]["model_id"].tolist()
    return [{"fold_id": 0, "fold_label": "single_vgg13_resub",
             "train": pool, "test": pool}]


def split_xarch_vit_in(models: pd.DataFrame) -> list[dict]:
    vgg = models[(models["architecture"] == "VGG13")
                 & models["paradigm"].isin(CNN_PARADIGMS)]
    vit = models[models["architecture"] == "ViT"]
    train = pd.concat([vgg, vit])["model_id"].tolist()
    test = models[(models["architecture"] == "ResNet18")
                  & models["paradigm"].isin(CNN_PARADIGMS)]["model_id"].tolist()
    return [{"fold_id": 0, "fold_label": "vgg13_vit_to_resnet18",
             "train": train, "test": test}]


# ---- Track 2 splits ----

def split_track2_loo(cells: pd.DataFrame) -> list[dict]:
    pool = cells[(cells["architecture"] == "VGG13")
                 & cells["paradigm"].isin(CNN_PARADIGMS)]
    folds = []
    for i, (_, row) in enumerate(pool.iterrows()):
        held = row["cell_id"]
        train = pool[pool["cell_id"] != held]["cell_id"].tolist()
        folds.append({"fold_id": i,
                      "fold_label": f"loo_{held.replace('|', '_')}",
                      "train": train, "test": [held]})
    return folds


# ---- Verification ----

def verify_no_leak(folds: list[dict]) -> dict:
    """Per-fold check: train ∩ test = ∅. Returns summary stats."""
    bad = []
    for f in folds:
        s_tr = set(f["train"])
        s_te = set(f["test"])
        inter = s_tr & s_te
        if inter:
            bad.append({"fold_label": f["fold_label"],
                        "n_intersection": len(inter)})
    return {"folds": len(folds), "leaks": bad}


# ---- Worked example renderer ----

def worked_examples_section(models: pd.DataFrame, cells: pd.DataFrame) -> str:
    lines = ["## Worked example — what each split does\n\n"]
    n_models = len(models)
    by_arch_par = (models.groupby(["architecture", "paradigm"]).size()
                   .rename("n_models").reset_index())
    lines.append("Model row counts in the Track 1 universe (376 total):\n\n")
    lines.append("```\n" + by_arch_par.to_string(index=False) + "\n```\n\n")

    lines.append("### `xarch` (one fold)\n\n")
    f0 = split_xarch(models)[0]
    lines.append(
        f"Fold `{f0['fold_label']}`: train = {len(f0['train'])} model rows "
        f"(VGG13 CNN), test = {len(f0['test'])} (ResNet18 CNN). The predictor "
        "sees no ResNet18 rows during training; this is the cross-architecture "
        "transfer headline.\n\n"
    )

    lines.append("### `lopo` (4 folds, pooled VGG13 + ResNet18 + ViT)\n\n")
    folds_lopo = split_lopo(models, ALL_PARADIGMS)
    rows = []
    for f in folds_lopo:
        rows.append({"fold_label": f["fold_label"],
                     "n_train": len(f["train"]), "n_test": len(f["test"])})
    lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "The `lopo_modelvit` fold trains on CNN paradigms only and tests on "
        "ViT — pure CNN→Transformer paradigm transfer. The other three folds "
        "test cross-paradigm generalization with both CNN architectures in "
        "the training pool.\n\n"
    )

    lines.append("### `lodo_vgg13` (4 folds)\n\n")
    folds_lodo = split_lodo_vgg13(models)
    rows = []
    for f in folds_lodo:
        rows.append({"fold_label": f["fold_label"],
                     "n_train": len(f["train"]), "n_test": len(f["test"])})
    lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "Each fold holds out one source dataset (with all VGG13 paradigms / "
        "configs / runs trained on it) — tests dataset transfer with "
        "architecture and paradigm pooled.\n\n"
    )

    lines.append("### `pxs_vgg13` (12 folds — 3 paradigms × 4 sources)\n\n")
    folds_pxs = split_pxs_vgg13(models)
    rows = []
    for f in folds_pxs:
        rows.append({"fold_label": f["fold_label"],
                     "n_train": len(f["train"]), "n_test": len(f["test"])})
    pxs_df = pd.DataFrame(rows)
    lines.append("```\n" + pxs_df.to_string(index=False) + "\n```\n\n")
    lines.append(
        "The smallest test fold is whichever (paradigm, source) cell has the "
        "fewest model rows — typically `confidnet` or `devries` cells "
        "(10 rows = 5 runs × 2 dropouts × 1 reward).\n\n"
    )

    lines.append("### `single_vgg13` (1 fold, diagnostic)\n\n")
    f0 = split_single_vgg13(models)[0]
    lines.append(
        f"Fold `{f0['fold_label']}`: train = test = {len(f0['train'])} "
        "VGG13 CNN model rows. This is a resubstitution estimate — the upper "
        "bound on what the predictor can learn from VGG13 alone. Used only as "
        "a within-population ceiling for the cross-population evaluations.\n\n"
    )

    lines.append("### `track2_loo` (12 folds, Track 2 cells)\n\n")
    folds_t2 = split_track2_loo(cells)
    rows = []
    for f in folds_t2:
        rows.append({"fold_label": f["fold_label"],
                     "n_train": len(f["train"]), "n_test": len(f["test"])})
    lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "Track 2 has 12 VGG13 cells (3 paradigms × 4 sources). With only 8 NC "
        "features, LOO is the only viable evaluation — every fold trains on "
        "11 cells and tests on 1. Reported as a sanity check, not a "
        "production predictor.\n\n"
    )

    lines.append("### Ablations\n\n")
    f0 = split_xarch_vit_in(models)[0]
    lines.append(
        f"`xarch_vit_in`: 1 fold, train = {len(f0['train'])} (VGG13 CNN + ViT), "
        f"test = {len(f0['test'])} (ResNet18 CNN). Same test set as primary "
        "xarch but with ViT in the training pool — measures whether ViT's NC "
        "diversity helps cross-arch transfer.\n\n"
    )
    folds_lopo_cnn = split_lopo(models, CNN_PARADIGMS)
    rows = [{"fold_label": f["fold_label"],
             "n_train": len(f["train"]), "n_test": len(f["test"])}
            for f in folds_lopo_cnn]
    lines.append(
        f"`lopo_cnn_only`: 3 folds (CNN paradigms only). "
        "Train pool excludes ViT, so this is a cleaner CNN-paradigm transfer "
        "without architecture-mediated NC differences confounding the test.\n\n"
    )
    lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    return "".join(lines)


# ---- Driver ----

def write_split(folds: list[dict], split_name: str, id_col: str,
                splits_dir: Path) -> tuple[int, dict]:
    long = folds_to_long(folds, split_name, id_col)
    write_parquet(long, splits_dir / f"{split_name}.parquet")
    leak = verify_no_leak(folds)
    return len(long), leak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    models = load_track1_models(out_root)
    cells = load_track2_cells(out_root)

    summary_rows = []

    track1_jobs = [
        ("xarch", split_xarch(models)),
        ("lopo", split_lopo(models, ALL_PARADIGMS)),
        ("lodo_vgg13", split_lodo_vgg13(models)),
        ("pxs_vgg13", split_pxs_vgg13(models)),
        ("single_vgg13", split_single_vgg13(models)),
        ("xarch_vit_in", split_xarch_vit_in(models)),
        ("lopo_cnn_only", split_lopo(models, CNN_PARADIGMS)),
    ]
    for name, folds in track1_jobs:
        n_rows, leak = write_split(folds, name, "model_id", splits_dir)
        # single_vgg13 is resubstitution by design (train == test); skip leak warning
        leak_count = sum(b["n_intersection"] for b in leak["leaks"])
        if name == "single_vgg13":
            leak_status = "expected (resubstitution)"
        elif leak_count == 0:
            leak_status = "clean"
        else:
            leak_status = f"LEAK {leak_count}"
        summary_rows.append({"split": name, "n_folds": leak["folds"],
                             "n_rows_in_parquet": n_rows,
                             "leakage": leak_status})
        print(f"wrote {splits_dir / (name + '.parquet')} "
              f"({leak['folds']} folds, {n_rows:,} rows, leakage: {leak_status})")

    # Track 2 split
    folds_t2 = split_track2_loo(cells)
    n_rows, leak = write_split(folds_t2, "track2_loo", "cell_id", splits_dir)
    leak_status = "clean" if not leak["leaks"] else f"LEAK {sum(b['n_intersection'] for b in leak['leaks'])}"
    summary_rows.append({"split": "track2_loo", "n_folds": leak["folds"],
                         "n_rows_in_parquet": n_rows,
                         "leakage": leak_status})
    print(f"wrote {splits_dir / 'track2_loo.parquet'} "
          f"({leak['folds']} folds, {n_rows:,} rows, leakage: {leak_status})")

    # Report
    out_path = out_root / "07_splits_check.md"
    lines = ["# Step 9 — Train/test splits\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/evaluation/splits.py`\n\n")

    lines.append(worked_examples_section(models, cells))

    lines.append("## Run summary\n\n")
    lines.append("```\n" + pd.DataFrame(summary_rows).to_string(index=False) + "\n```\n\n")

    lines.append("## Leakage check\n\n")
    lines.append(
        "For every fold of every split (excluding `single_vgg13` which is "
        "resubstitution by design), the intersection of `train` and `test` "
        "model/cell IDs must be empty. Any non-zero intersection above is a "
        "leak that would invalidate downstream evaluation.\n\n"
    )

    out_path.write_text("".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
