"""Materialize the stratified train superset as parquet shards.

Reads the downloaded HF train shards (data/train-*.parquet, original JPEG
bytes + integer labels), takes the FIRST --per-class images of each class in
deterministic shard order, and repacks them into --n-shards output shards:
columns (image_bytes, label, source_shard, source_row). A manifest parquet
records the same tuple per selected row; the fit set, selection set, and
G3's alternative-seed subsample are all seeded row selections WITHIN this
superset (documentation/imagenet_scale_plan.md), so the full train set is
never read again.

Deterministic by construction (shard order + first-N-per-class); the
manifest, not the sampling rule, is the reproducibility artifact.

Run (CPU, login node or any job; ~150 GB sequential read, one pass):
  python x9_imagenet/build_superset.py \
      --raw-dir  $DATASET_ROOT_DIR/imagenet1k_raw/data \
      --out-dir  $DATASET_ROOT_DIR/imagenet1k_superset \
      --per-class 250 --n-shards 12
"""
from __future__ import annotations

import argparse
import collections
import glob
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

N_CLASSES = 1000
SCHEMA = pa.schema([
    ("image_bytes", pa.binary()),
    ("label", pa.int64()),
    ("source_shard", pa.string()),
    ("source_row", pa.int64()),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--n-shards", type=int, default=12)
    args = ap.parse_args()

    shards = sorted(glob.glob(str(pathlib.Path(args.raw_dir) / "train-*.parquet")))
    if not shards:
        raise SystemExit(f"no train-*.parquet under {args.raw_dir}")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_target = args.per_class * N_CLASSES
    per_out = -(-total_target // args.n_shards)  # ceil

    counts: collections.Counter = collections.Counter()
    full_classes = 0
    buf: list[dict] = []
    manifest: list[dict] = []
    out_idx = 0
    n_written = 0

    def flush() -> None:
        nonlocal buf, out_idx, n_written
        if not buf:
            return
        path = out_dir / f"superset-{out_idx:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buf, schema=SCHEMA), path,
                       compression="none")  # JPEG bytes are already compressed
        n_written += len(buf)
        logger.info(f"wrote {path.name} ({len(buf)} rows; total {n_written:,})")
        buf, out_idx = [], out_idx + 1

    done = False
    for shard in shards:
        if done:
            break
        name = pathlib.Path(shard).name
        pf = pq.ParquetFile(shard)
        row_base = 0
        for batch in pf.iter_batches(columns=["image", "label"],
                                     batch_size=512):
            imgs = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for j, (img, y) in enumerate(zip(imgs, labels)):
                if counts[y] >= args.per_class:
                    continue
                counts[y] += 1
                if counts[y] == args.per_class:
                    full_classes += 1
                rec = {"image_bytes": img["bytes"], "label": y,
                       "source_shard": name, "source_row": row_base + j}
                buf.append(rec)
                manifest.append({"label": y, "source_shard": name,
                                 "source_row": row_base + j,
                                 "out_shard": out_idx,
                                 "class_rank": counts[y] - 1})
                if len(buf) >= per_out:
                    flush()
                if full_classes == N_CLASSES:
                    done = True
                    break
            row_base += len(labels)
            if done:
                break
        logger.info(f"{name}: {full_classes}/{N_CLASSES} classes full, "
                    f"{sum(counts.values()):,} selected")
    flush()

    if full_classes < N_CLASSES:
        short = [c for c in range(N_CLASSES) if counts[c] < args.per_class]
        logger.warning(f"{len(short)} classes below {args.per_class} images "
                       f"(min {min(counts[c] for c in short)}); recorded "
                       "as-is, draws must respect per-class availability")
    mpath = out_dir / "superset_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest), mpath)
    logger.info(f"wrote {mpath} ({len(manifest):,} rows); "
                f"superset complete: {n_written:,} images in {out_idx} shards")


if __name__ == "__main__":
    main()
