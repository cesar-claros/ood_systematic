"""Stage-2 Option B (audit #5 section 12; frozen design in
documentation/heldout_theory_validation_design.md): pool-wide OOD-coordinate
extraction for the 280-checkpoint VGG-13 pool (HPC side).

Per checkpoint: forwards the train split, ID test, and every paper OOD set;
fits the frozen pilot0 feature model on train; writes ONE small JSON with the
measured geometry (Papyan panel, logit scale, SNR, anisotropy), the ID-test
coordinates (its rho is the train-to-test generalization ratio), and the
frozen H-estimator coordinates (gamma, a, rho, w_perp, ...) for every OOD
set. No feature cache is retained (pass --keep_npz to also save the
extract_pilot0-style NPZ).

The coordinate estimators are the FROZEN pilot0 definitions
(pilot0/ood_coords.py, frozen 2026-08-15); nothing here refits or retunes
them. Failures are isolated per checkpoint (FAILED_<slug>.json records the
error; the sweep continues), and the sweep is resumable (existing outputs are
skipped).

Usage (from code/, inside the campaign container, .env with
EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR):
    # one checkpoint
    python pilot0/extract_pool_coords.py --model_path \
        cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2
    # full pool sweep, optionally sharded across GPUs
    python pilot0/extract_pool_coords.py --sweep --shard 1/2
    python pilot0/extract_pool_coords.py --sweep --shard 2/2
    # enumerate without running
    python pilot0/extract_pool_coords.py --sweep --list
Outputs: pilot0/pool_coords/<slug>.json  (rsync the whole folder back)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords

OOD_TEST_SETS: dict[str, str] = {
    "ood_sncs": "test_3",
    "ood_nsncs_svhn": "test_4",
    "ood_nsncs_ti": "test_5",
    "ood_nsncs_lsun_cropped": "test_6",
    "ood_nsncs_lsun_resize": "test_7",
    "ood_nsncs_isun": "test_8",
    "ood_nsncs_textures": "test_9",
    "ood_nsncs_places365": "test_10",
}

CNN_RE = re.compile(
    r"^(?P<src>[a-z0-9\-]+)_paper_sweep/(?P<paradigm>[a-z]+)_bb"
    r"(?P<bb>vgg13)_do(?P<do>\d)_run(?P<run>\d+)_rew(?P<rew>[\d.]+)$")
SRC_KEY = {"cifar10": "cifar10", "cifar100": "cifar100",
           "supercifar": "supercifar100", "tiny-imagenet-200": "tinyimagenet"}
MANIFEST = Path(__file__).resolve().parent / "pool_manifest.json"
OUT_DIR_DEFAULT = "pilot0/pool_coords"


def extract_one(model_path: str, out_dir: Path, use_cuda: bool,
                keep_npz: bool) -> dict:
    import torch
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    slug = model_path.replace("/", "__")
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    if study_name == "vit":
        w, b = utils.get_model_and_last_layer(module, study_name,
                                              return_model=False)
    else:
        _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_train = ev["encoded"].cpu().numpy().astype(np.float32)
    y_train = ev["labels"].cpu().numpy().astype(np.int64)

    fm = fit_feature_model(h_train, y_train, n_classes)
    record: dict = {
        "model_path": model_path, "slug": slug, "study": study_name,
        "n_classes": n_classes, "dim": int(h_train.shape[1]),
        "n_train": int(len(h_train)),
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm),
        "ood": {},
    }
    arrays = ({"w": w_np, "b": b_np, "h_train": h_train, "y_train": y_train}
              if keep_npz else {})

    logger.info(f"{slug}: forward iid test")
    ev = utils.compute_model_evaluations(model, datamodule, "test_1")
    h_iid = ev["encoded"].cpu().numpy().astype(np.float32)
    record["iid_test"] = dict(estimate_ood_coords(h_iid, fm),
                              n=int(len(h_iid)))
    if keep_npz:
        arrays["h_iid_test"] = h_iid
    del h_iid

    for mode, test_set in OOD_TEST_SETS.items():
        logger.info(f"{slug}: forward {mode} ({test_set})")
        try:
            ev = utils.compute_model_evaluations(model, datamodule, test_set)
        except (FileNotFoundError, NotImplementedError) as err:
            logger.error(f"{slug}: skipping {mode}: {err}")
            record["ood"][mode] = {"error": str(err)}
            continue
        h_ood = ev["encoded"].cpu().numpy().astype(np.float32)
        record["ood"][mode] = dict(estimate_ood_coords(h_ood, fm),
                                   n=int(len(h_ood)))
        if keep_npz:
            arrays[f"h_{mode}"] = h_ood
        del ev, h_ood

    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(record, indent=1))
    if keep_npz:
        np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    logger.info(f"{slug}: wrote coords JSON "
                f"({len([k for k, v in record['ood'].items() if 'error' not in v])} "
                f"OOD sets, {record['runtime_sec']}s)")
    del arrays, h_train, y_train, model, module, datamodule
    if use_cuda:
        torch.cuda.empty_cache()
    return record


def sweep_targets() -> tuple[list[str], list[dict], list[str]]:
    """Enumerate experiment dirs on disk, match against the frozen manifest.

    Returns (matched experiment paths, unmatched manifest cells, extra dirs).
    """
    import os

    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    manifest = json.loads(MANIFEST.read_text())["cells"]
    want = {(c["paradigm"], c["source"], c["run"], round(c["reward"], 4),
             int(c["dropout"])): c for c in manifest}
    matched: dict[tuple, str] = {}
    extra: list[str] = []
    for src_dir in sorted({c["src_dir"] for c in manifest}):
        study = root / f"{src_dir}_paper_sweep"
        if not study.is_dir():
            continue
        for p in sorted(study.iterdir()):
            rel = f"{src_dir}_paper_sweep/{p.name}"
            m = CNN_RE.match(rel)
            if not m:
                continue
            key = (m["paradigm"], SRC_KEY[m["src"]], int(m["run"]),
                   round(float(m["rew"]), 4), int(m["do"]))
            if key in want:
                matched[key] = rel
            else:
                extra.append(rel)
    missing = [c for k, c in want.items() if k not in matched]
    return sorted(matched.values()), missing, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--shard", type=str, default="1/1",
                        help="k/n: run the k-th of n interleaved shards")
    parser.add_argument("--list", action="store_true",
                        help="with --sweep: enumerate and exit")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--keep_npz", action="store_true")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    import torch
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())

    if args.model_path and not args.sweep:
        extract_one(args.model_path, out_dir, use_cuda, args.keep_npz)
        return

    if not args.sweep:
        parser.error("pass --model_path or --sweep")
    targets, missing, extra = sweep_targets()
    k, n = (int(x) for x in args.shard.split("/"))
    shard = targets[k - 1::n]
    print(f"[sweep] manifest 280; matched on disk {len(targets)}; "
          f"MISSING {len(missing)}; unrelated dirs {len(extra)}; "
          f"shard {k}/{n} -> {len(shard)} checkpoints")
    for c in missing:
        print(f"[sweep] MISSING from disk: {c}")
    if args.list:
        for t in shard:
            print(t)
        return
    done = skipped = failed = 0
    for i, rel in enumerate(shard, 1):
        slug = rel.replace("/", "__")
        if (out_dir / f"{slug}.json").exists():
            skipped += 1
            continue
        print(f"[sweep {k}/{n}] {i}/{len(shard)}: {rel}")
        try:
            extract_one(rel, out_dir, use_cuda, args.keep_npz)
            done += 1
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"model_path": rel,
                 "traceback": traceback.format_exc()}, indent=1))
            print(f"[sweep] FAILED {rel} (recorded, continuing)")
    print(f"[sweep {k}/{n}] finished: {done} new, {skipped} skipped, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
