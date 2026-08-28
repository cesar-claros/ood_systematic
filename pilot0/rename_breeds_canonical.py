"""Rename the BREEDS ImageNet train tree from the Hugging Face
re-packaging convention to canonical ILSVRC names (contamination fix,
2026-08-27).

The HF parquet names train images `<wnid>_<num>_<wnid>.JPEG`; the BREEDS
tree was materialized with those names. The BREEDS loader's ID-test split
is a seed-12345 shuffle of the SORTED file listing, and the suffixed names
sort differently from the canonical `<wnid>_<num>.JPEG`, so the shuffled
10k ID-test slice was essentially unrelated to the released checkpoints'
held-out slice (~94% of it was in their training data). Renaming to
canonical names restores the canonical listing order and therefore the
authors' exact partition; the BREEDS Stage-2 extraction must be rerun
afterwards.

Rules: strip the trailing `_<wnid>` only when it equals the file's leading
wnid AND the directory wnid; abort on any target collision; already
canonical files are left untouched. --dry_run previews.

Usage (HPC):
    python pilot0/rename_breeds_canonical.py \
        --train_root $DATASET_ROOT_DIR/breeds/ILSVRC/Data/CLS-LOC/train [--dry_run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WNID_RE = re.compile(r"^n\d{8}$")


def canonical_name(name: str, dir_wnid: str) -> str | None:
    """Canonical file name, or None if already canonical / not renamable."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return None
    parts = stem.split("_")
    if (len(parts) >= 3 and WNID_RE.match(parts[-1])
            and parts[0] == parts[-1] == dir_wnid):
        return "_".join(parts[:-1]) + "." + ext
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train_root", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.train_root)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    renamed = untouched = 0
    for wdir in sorted(root.iterdir()):
        if not wdir.is_dir() or not WNID_RE.match(wdir.name):
            continue
        for f in sorted(wdir.iterdir()):
            new = canonical_name(f.name, wdir.name)
            if new is None:
                untouched += 1
                continue
            target = wdir / new
            if target.exists():
                sys.exit(f"ABORT: collision {f} -> {target}")
            if not args.dry_run:
                f.rename(target)
            renamed += 1
        if renamed and renamed % 50000 < 1300:
            print(f"[rename] progress: {renamed} renamed", flush=True)
    mode = "DRY RUN: would rename" if args.dry_run else "renamed"
    print(f"[rename] DONE: {mode} {renamed}, untouched {untouched}")
    if not args.dry_run and renamed:
        print("[rename] now delete the BREEDS stage-2 records and rerun "
              "the extractor (see instructions)")


if __name__ == "__main__":
    main()
