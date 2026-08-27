"""Materialize the BREEDS Entity-13 ImageNet subset from Hugging Face
ILSVRC2012 parquet shards (source-expansion Stage-1 prerequisite; HPC side).

The fd-shifts BREEDS loader (fd_shifts/loaders/dataset_collection.py,
BREEDImageNet) reads the Kaggle-style layout
    <data_dir>/ILSVRC/Data/CLS-LOC/train/<wnid>/<original>.JPEG
and uses ONLY the train split: the ID test set is a 10,000-image slice of
the seed-12345 shuffled sorted file listing of the source subclasses, and
the OOD test set is the full listing of the target subclasses. Two
consequences drive this script's design:

1. Only the 260 Entity-13 wnids are needed (union of both subclass
   partitions of make_entity13(hierarchy_dir, split="rand"), the exact
   call the loader makes), roughly 26% of ImageNet train.
2. ORIGINAL FILENAMES ARE LOAD-BEARING: the train/id_test partition is a
   seeded shuffle of the sorted listing, so file names must match the
   original ILSVRC names for the partition to reproduce the one the
   released checkpoints were evaluated/trained against. The HF parquet
   preserves the original basename in the embedded image struct's `path`
   field; this script refuses to run without it unless
   --allow_synthetic_names is passed (which is recorded loudly, because it
   silently moves training images into the ID-test slice).

Self-checks (abort on failure): the hierarchy's dataset_class_info.json
must have 1,000 entries with wnids in ascending order (validates the
assumption that the HF integer label equals the sorted-wnid index), and
every extracted train row's basename prefix must equal the wnid mapped
from its label. The computed subclass partition is written to
<out_dir>/entity13_partition.json for the provenance record; the loader
recomputes it at runtime, and comparing the two verifies determinism.

Resumable (existing non-empty files are skipped). Only `train-*.parquet`
shards are read; HF validation/test shards are ignored.

Usage (HPC, inside the campaign container):
    python pilot0/build_breeds_from_parquet.py \
        --parquet_dir $DATASET_ROOT_DIR/imagetnet1k_raw \
        --out_dir $DATASET_ROOT_DIR/breeds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def entity13_wnids() -> tuple[dict[int, str], set[int], dict]:
    """(class_number -> wnid) map, Entity-13 class-number whitelist, and the
    partition record, computed exactly as the fd-shifts loader does."""
    from fd_shifts.loaders import breeds_hierarchies
    from robustness.tools.breeds_helpers import make_entity13

    info_dir = os.path.abspath(os.path.dirname(breeds_hierarchies.__file__))
    info = json.loads(
        (Path(info_dir) / "dataset_class_info.json").read_text())
    num_to_wnid = {int(row[0]): str(row[1]) for row in info}
    assert len(num_to_wnid) == 1000, f"class info has {len(num_to_wnid)} rows"
    wnids_in_order = [num_to_wnid[i] for i in range(1000)]
    assert wnids_in_order == sorted(wnids_in_order), (
        "dataset_class_info.json is not in sorted-wnid order; the HF "
        "label -> wnid assumption does not hold, aborting")

    # The loader consumes ONLY subclass_split (source groups for
    # train/id_test, target groups for ood_test); the first return value is
    # superclass metadata and is not part of any grouping.
    _, subclass_split, _ = make_entity13(info_dir, split="rand")
    train_sub, test_sub = subclass_split
    wnid_to_num = {w: n for n, w in num_to_wnid.items()}

    def to_num(c) -> int:
        s = str(c)
        if s.startswith("n") and s in wnid_to_num:
            return wnid_to_num[s]
        return int(c)

    flat = {to_num(c) for group in train_sub for c in group}
    flat |= {to_num(c) for group in test_sub for c in group}
    record = {
        "call": 'make_entity13(hierarchy_dir, split="rand")',
        "n_whitelist_classes": len(flat),
        "source_subclasses": [[num_to_wnid[to_num(c)] for c in g]
                              for g in train_sub],
        "target_subclasses": [[num_to_wnid[to_num(c)] for c in g]
                              for g in test_sub],
    }
    return num_to_wnid, flat, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="the BREEDS data_dir; images land under "
                             "ILSVRC/Data/CLS-LOC/train/<wnid>/")
    parser.add_argument("--allow_synthetic_names", action="store_true",
                        help="proceed without embedded original filenames "
                             "(NOT recommended: silently changes the "
                             "seed-12345 id_test partition)")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    shards = sorted(Path(args.parquet_dir).rglob("train-*.parquet"))
    if not shards:
        sys.exit(f"no train-*.parquet under {args.parquet_dir}")
    num_to_wnid, whitelist, record = entity13_wnids()
    train_root = Path(args.out_dir) / "ILSVRC/Data/CLS-LOC/train"
    train_root.mkdir(parents=True, exist_ok=True)
    part_path = Path(args.out_dir) / "entity13_partition.json"
    part_path.write_text(json.dumps(record, indent=1))
    print(f"[breeds] {len(shards)} train shards; whitelist "
          f"{len(whitelist)} classes; writing under {train_root}",
          flush=True)

    t0 = time.time()
    written = skipped = filtered = 0
    synthetic = 0
    mismatches = 0
    for si, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=args.batch_size,
                                     columns=["image", "label"]):
            labels = batch.column("label").to_pylist()
            images = batch.column("image").to_pylist()
            for img, label in zip(images, labels):
                label = int(label)
                if label not in whitelist:
                    filtered += 1
                    continue
                wnid = num_to_wnid[label]
                name = os.path.basename(str(img.get("path") or ""))
                if not name:
                    if not args.allow_synthetic_names:
                        sys.exit(
                            "parquet rows carry no embedded filename "
                            "(image.path empty). The id_test partition is "
                            "a seeded shuffle of the sorted listing, so "
                            "original names are required; rerun with "
                            "--allow_synthetic_names only if you accept a "
                            "changed partition (recorded).")
                    synthetic += 1
                    name = f"{wnid}_{written + skipped:08d}.JPEG"
                elif not name.startswith(wnid):
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"[breeds] WARNING name/label mismatch: "
                              f"{name} vs {wnid}", flush=True)
                out = train_root / wnid / name
                if out.exists() and out.stat().st_size > 0:
                    skipped += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(img["bytes"])
                written += 1
        print(f"[breeds] shard {si}/{len(shards)} done: written {written}, "
              f"skipped {skipped}, filtered {filtered}, "
              f"{time.time() - t0:.0f}s", flush=True)

    if mismatches:
        sys.exit(f"[breeds] ABORT-LEVEL: {mismatches} filename/label "
                 f"mismatches; the HF label mapping assumption failed. "
                 f"Extracted files are suspect; do not use.")
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    counts = {c: len(list((train_root / c).iterdir())) for c in classes}
    print(f"[breeds] DONE: {len(classes)} classes (expect 260), "
          f"{sum(counts.values())} images, min/class "
          f"{min(counts.values()) if counts else 0}, written {written}, "
          f"skipped {skipped}, synthetic names {synthetic}", flush=True)
    (Path(args.out_dir) / "build_summary.json").write_text(json.dumps(
        {"n_classes": len(classes), "n_images": sum(counts.values()),
         "written": written, "skipped": skipped,
         "synthetic_names": synthetic, "per_class_min":
         min(counts.values()) if counts else 0}, indent=1))


if __name__ == "__main__":
    main()
