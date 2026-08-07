"""Repack the downloaded OOD suite into parquet shards (one set at a time).

For the five OpenOOD-benchmark sets (ssb_hard, ninco, inaturalist, texture,
openimage_o) rows follow the OpenOOD imglist (benchmark_imglist/imagenet/
test_<set>.txt), so the G1 gate evaluates on their exact lists; files are
resolved on disk by path-suffix matching (the zips' inner layouts differ)
and every unresolved imglist entry is reported. The three extra sets
(imagenet_o, sun, places) have no OpenOOD ImageNet list: all images found
under the set directory are packed, label -1.

Output: $out_dir/<set>-NNNNN.parquet with columns
(image_bytes, label, relpath), ~12.5k rows per shard so DataLoader workers
can split ssb_hard; original bytes, no re-encode.

Run (CPU):
  python x9_imagenet/repack_ood.py --suite-dir $DATASET_ROOT_DIR/ood_suite \
      --out-dir $DATASET_ROOT_DIR/ood_parquet
"""
from __future__ import annotations

import argparse
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMGLIST_SETS = ["ssb_hard", "ninco", "inaturalist", "texture", "openimage_o"]
GLOB_SETS = ["imagenet_o", "sun", "places"]
IMGLIST_NAME = {"texture": "test_textures.txt"}  # others: test_<set>.txt
ROWS_PER_SHARD = 12500
SCHEMA = pa.schema([("image_bytes", pa.binary()), ("label", pa.int64()),
                    ("relpath", pa.string())])


def disk_index(root: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Keyed by LOWERCASED basename: imglist entries and zip contents differ
    in case for some sets (.jpg vs .JPG etc.), which silently dropped ~10%
    of openimage_o/texture on the first pass."""
    idx: dict[str, list[pathlib.Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx.setdefault(p.name.lower(), []).append(p)
    return idx


def resolve(list_path: str, idx: dict[str, list[pathlib.Path]]) -> pathlib.Path | None:
    cands = idx.get(pathlib.PurePosixPath(list_path).name.lower())
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # basename collision: prefer the candidate sharing the longest path suffix
    want = pathlib.PurePosixPath(list_path).parts
    best, best_n = cands[0], -1
    for c in cands:
        have = c.parts
        n = 0
        while (n < min(len(want), len(have))
               and want[-1 - n] == have[-1 - n]):
            n += 1
        if n > best_n:
            best, best_n = c, n
    return best


def write_shards(rows: list[dict], out_dir: pathlib.Path, name: str) -> int:
    n = 0
    for i in range(0, len(rows), ROWS_PER_SHARD):
        chunk = rows[i:i + ROWS_PER_SHARD]
        path = out_dir / f"{name}-{i // ROWS_PER_SHARD:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk, schema=SCHEMA), path,
                       compression="none")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", nargs="*", default=IMGLIST_SETS + GLOB_SETS)
    args = ap.parse_args()
    suite = pathlib.Path(args.suite_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imglist_dir = suite / "benchmark_imglist" / "benchmark_imglist" / "imagenet"

    for name in args.sets:
        root = suite / name
        if not root.is_dir():
            logger.error(f"{name}: missing directory {root}, skipped")
            continue
        rows, missing = [], 0
        if name in IMGLIST_SETS:
            lp = imglist_dir / IMGLIST_NAME.get(name, f"test_{name}.txt")
            if not lp.exists():
                raise SystemExit(f"imglist not found: {lp}")
            idx = disk_index(root)
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                pth, lab = line.rsplit(" ", 1)
                f = resolve(pth, idx)
                if f is None:
                    missing += 1
                    continue
                rows.append({"image_bytes": f.read_bytes(), "label": int(lab),
                             "relpath": str(f.relative_to(root))})
            src = f"imglist {lp.name}"
        else:
            for f in sorted(p for p in root.rglob("*")
                            if p.is_file() and p.suffix.lower() in IMG_EXTS):
                rows.append({"image_bytes": f.read_bytes(), "label": -1,
                             "relpath": str(f.relative_to(root))})
            src = "directory glob"
        n_shards = write_shards(rows, out_dir, name)
        msg = (f"{name}: {len(rows):,} images -> {n_shards} shards ({src})"
               + (f"; {missing} imglist entries UNRESOLVED on disk"
                  if missing else ""))
        (logger.warning if missing else logger.info)(msg)


if __name__ == "__main__":
    main()
