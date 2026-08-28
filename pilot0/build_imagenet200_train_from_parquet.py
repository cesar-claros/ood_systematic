"""Materialize the OpenOOD ImageNet-200 TRAIN images from Hugging Face
ILSVRC2012 parquet shards (source-expansion protocol stage 3 prerequisite).

OpenOOD's ImageNet-200 references images by explicit imglist paths
(`imagenet_1k/train/<wnid>/<file>.JPEG label`), so this materializer is
imglist-driven: it writes EXACTLY the files listed in
`benchmark_imglist/imagenet200/train_imagenet200.txt` under
`<out_root>/images_largescale/`, using each row's embedded original
filename to match. No directory-listing semantics are involved (unlike
BREEDS), so only the listed files are needed. The ID val/test images come
from OpenOOD's own `imagenet_1k` download (validation split); this script
handles train only.

Self-checks: every written file matches its listed wnid directory; the
final coverage report lists any imglist entry that was not found in the
parquet shards (abort-level if any are missing).

Usage (HPC, container with pyarrow):
    python pilot0/build_imagenet200_train_from_parquet.py \
        --imglist $DATASET_ROOT_DIR/openood/data/benchmark_imglist/imagenet200/train_imagenet200.txt \
        --parquet_dir $DATASET_ROOT_DIR/imagetnet1k_raw \
        --out_root $DATASET_ROOT_DIR/openood/data
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

WNID_RE = re.compile(r"^n\d{8}$")


def canon(basename: str) -> str:
    """Canonical match key for an ImageNet train file name.

    Normalizes extension case and the Hugging Face re-packaging convention
    that renames `<wnid>_<num>.JPEG` to `<wnid>_<num>_<wnid>.JPEG` (the
    trailing duplicate wnid is stripped when it matches the leading one).
    """
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    parts = stem.split("_")
    if len(parts) >= 3 and WNID_RE.match(parts[-1]) and parts[0] == parts[-1]:
        stem = "_".join(parts[:-1])
    return stem.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imglist", type=str, required=True)
    parser.add_argument("--parquet_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    needed: dict[str, str] = {}
    samples: list[str] = []
    for line in Path(args.imglist).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.rsplit(" ", 1)[0]
        key = canon(os.path.basename(rel))
        if key in needed:
            sys.exit(f"[in200] ABORT: canonical-key collision on {key}")
        needed[key] = rel
        if len(samples) < 3:
            samples.append(rel)
    print(f"[in200] imglist lists {len(needed)} train files; sample "
          f"entries: {samples}", flush=True)

    out_root = Path(args.out_root) / "images_largescale"
    shards = sorted(Path(args.parquet_dir).rglob("train-*.parquet"))
    if not shards:
        sys.exit(f"no train-*.parquet under {args.parquet_dir}")

    t0 = time.time()
    written = skipped = 0
    pq_samples: list[str] = []
    for si, shard in enumerate(shards, 1):
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=args.batch_size,
                                     columns=["image"]):
            for img in batch.column("image").to_pylist():
                name = os.path.basename(str(img.get("path") or ""))
                if len(pq_samples) < 3:
                    pq_samples.append(name)
                rel = needed.get(canon(name))
                if rel is None:
                    continue
                wnid = rel.split("/")[-2]
                if not name.startswith(wnid):
                    sys.exit(f"[in200] ABORT: {name} listed under {wnid}")
                out = out_root / "imagenet_1k" / rel.split(
                    "imagenet_1k/", 1)[-1]
                if out.exists() and out.stat().st_size > 0:
                    skipped += 1
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(img["bytes"])
                written += 1
        print(f"[in200] shard {si}/{len(shards)}: written {written}, "
              f"skipped {skipped}, {time.time() - t0:.0f}s", flush=True)
        if si == 1:
            print(f"[in200] parquet sample basenames: {pq_samples}",
                  flush=True)
        if si == 5 and written + skipped == 0:
            sys.exit("[in200] ABORT: zero matches after 5 shards even with "
                     "normalization. Compare the imglist sample entries "
                     "and parquet sample basenames printed above and "
                     "report both.")

    present = sum(1 for rel in needed.values()
                  if (out_root / "imagenet_1k"
                      / rel.split("imagenet_1k/", 1)[-1]).exists())
    missing = len(needed) - present
    print(f"[in200] DONE: {present}/{len(needed)} listed files on disk, "
          f"written {written}, skipped {skipped}, missing {missing}",
          flush=True)
    (Path(args.out_root) / "imagenet200_train_build_summary.json"
     ).write_text(json.dumps({"listed": len(needed), "present": present,
                              "missing": missing}, indent=1))
    if missing:
        sys.exit(f"[in200] {missing} listed files NOT found in the parquet "
                 f"shards; do not run the pilot until resolved")


if __name__ == "__main__":
    main()
