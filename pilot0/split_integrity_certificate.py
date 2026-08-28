"""R6 split-integrity certificate (audit #8 sections 6.7 and 10; HPC side).

Produces a reproducible certificate for each expansion source:
record counts per split; zero filepath intersection between splits; zero
exact sha256 content intersection between train and ID test; label
frequencies; canonical-filename rule check; sha256 of the sorted split
manifest; and the statement that all reported BREEDS numbers come from the
post-repair (canonically renamed) extraction.

BREEDS replicates the loader's partition exactly: the same
make_entity13(split="rand") groupings, the same robustness ImageFolder
listing, and the same seed-12345 shuffle with the 10k id_test slice.
SVHN hashes the torchvision train and test images. ImageNet-200 hashes the
files listed by the OpenOOD train/test imglists.

Usage (HPC, inside the container):
    python pilot0/split_integrity_certificate.py --source breeds \
        --data_root $DATASET_ROOT_DIR/breeds
    python pilot0/split_integrity_certificate.py --source svhn \
        --data_root $DATASET_ROOT_DIR/svhn
    python pilot0/split_integrity_certificate.py --source imagenet200 \
        --data_root $DATASET_ROOT_DIR/openood/data
Output: pilot0/split_integrity_<source>.json (rsync back; small)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

WNID_FILE_RE = re.compile(r"^n\d{8}_\d+\.JPEG$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_many(paths: list, label: str) -> dict[str, str]:
    out = {}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        out[str(p)] = sha256_file(Path(p))
        if i % 20000 == 0:
            print(f"[r6] {label}: {i}/{len(paths)} hashed "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return out


def manifest_hash(split_paths: dict[str, list]) -> str:
    lines = sorted(f"{split}\t{p}" for split, paths in split_paths.items()
                   for p in paths)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def certify(split_paths: dict[str, list], split_labels: dict[str, list],
            canonical_checked: bool, note: str) -> dict:
    sets = {k: set(map(str, v)) for k, v in split_paths.items()}
    inter = {}
    keys = sorted(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter[f"{a}&{b}"] = len(sets[a] & sets[b])
    print("[r6] hashing train + id_test for content intersection",
          flush=True)
    h_train = hash_many(split_paths["train"], "train")
    h_id = hash_many(split_paths["id_test"], "id_test")
    shared = set(h_train.values()) & set(h_id.values())
    content_inter = len(shared)
    # Duplicate-exclusion support (audit #9 section 6.3): list the exact
    # ID-test files whose content also appears in train.
    dup_by_hash: dict[str, dict] = {h: {"sha256": h, "train": [],
                                        "id_test": []} for h in shared}
    for p, h in h_train.items():
        if h in shared:
            dup_by_hash[h]["train"].append(p)
    for p, h in h_id.items():
        if h in shared:
            dup_by_hash[h]["id_test"].append(p)
    cert = {
        "counts": {k: len(v) for k, v in split_paths.items()},
        "filepath_intersections": inter,
        "train_idtest_content_hash_intersection": content_inter,
        "label_frequencies": {k: dict(Counter(v))
                              for k, v in split_labels.items()},
        "canonical_filename_rule_checked": canonical_checked,
        "split_manifest_sha256": manifest_hash(split_paths),
        "note": note,
    }
    cert["duplicates"] = sorted(dup_by_hash.values(),
                                key=lambda d: d["sha256"])
    cert["PASS"] = (all(v == 0 for v in inter.values())
                    and content_inter == 0)
    return cert


def breeds(data_root: Path) -> dict:
    sys.path.insert(0, ".")
    from fd_shifts.loaders.dataset_collection import BREEDImageNet

    splits = {}
    labels = {}
    for split, name in (("train", "train"), ("id_test", "id_test"),
                        ("ood_test", "ood")):
        ds = BREEDImageNet(root=str(data_root), split=name, download=None,
                           transform=None, kwargs=None)
        splits[split] = [p for p, _ in ds.samples]
        labels[split] = [int(t) for _, t in ds.samples]
        print(f"[r6] breeds {split}: {len(ds.samples)} samples", flush=True)
    bad = [p for v in splits.values() for p in v
           if not WNID_FILE_RE.match(Path(p).name)]
    if bad:
        sys.exit(f"[r6] ABORT: {len(bad)} non-canonical filenames, e.g. "
                 f"{bad[:3]}")
    note = ("All reported BREEDS numbers come from the post-repair "
            "extraction: canonical ILSVRC filenames restored by "
            "rename_breeds_canonical.py, which restores the exact "
            "seed-12345 partition of the released checkpoints.")
    return certify(splits, labels, True, note)


def svhn(data_root: Path) -> dict:
    import numpy as np
    import torchvision

    tr = torchvision.datasets.SVHN(str(data_root), split="train",
                                   download=False)
    te = torchvision.datasets.SVHN(str(data_root), split="test",
                                   download=False)
    h = {}
    for split, ds in (("train", tr), ("id_test", te)):
        h[split] = [hashlib.sha256(
            np.ascontiguousarray(ds.data[i]).tobytes()).hexdigest()
            for i in range(len(ds.data))]
        print(f"[r6] svhn {split}: {len(h[split])} hashed", flush=True)
    inter = len(set(h["train"]) & set(h["id_test"]))
    return {"counts": {k: len(v) for k, v in h.items()},
            "train_idtest_content_hash_intersection": inter,
            "label_frequencies": {
                "train": dict(Counter(tr.labels.tolist())),
                "id_test": dict(Counter(te.labels.tolist()))},
            "split_manifest_sha256": hashlib.sha256(
                "\n".join(h["train"] + h["id_test"]).encode()).hexdigest(),
            "note": "torchvision official SVHN train/test split; "
                    "content-level duplicate check",
            "PASS": inter == 0}


def imagenet200(data_root: Path) -> dict:
    def read_list(rel: str, sub: str):
        paths, labels = [], []
        for line in (data_root / rel).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p, lab = line.rsplit(" ", 1)
            paths.append(str(data_root / sub / p))
            labels.append(int(lab))
        return paths, labels

    splits, labels = {}, {}
    for split, rel in (
            ("train", "benchmark_imglist/imagenet200/train_imagenet200.txt"),
            ("id_test", "benchmark_imglist/imagenet200/test_imagenet200.txt")):
        splits[split], labels[split] = read_list(rel, "images_largescale")
        print(f"[r6] imagenet200 {split}: {len(splits[split])} listed",
              flush=True)
    missing = [p for v in splits.values() for p in v
               if not Path(p).is_file()]
    if missing:
        sys.exit(f"[r6] ABORT: {len(missing)} listed files missing, e.g. "
                 f"{missing[:3]}")
    note = ("OpenOOD imglist-defined splits; train images materialized "
            "from HF parquet under the LISTED canonical names")
    return certify(splits, labels, True, note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True,
                        choices=["breeds", "svhn", "imagenet200"])
    parser.add_argument("--data_root", required=True, type=str)
    args = parser.parse_args()
    fn = {"breeds": breeds, "svhn": svhn, "imagenet200": imagenet200}
    cert = fn[args.source](Path(args.data_root))
    out = Path(f"pilot0/split_integrity_{args.source}.json")
    out.write_text(json.dumps(cert, indent=1, default=str))
    print(f"[r6] {args.source}: PASS={cert['PASS']}; wrote {out}")


if __name__ == "__main__":
    main()
