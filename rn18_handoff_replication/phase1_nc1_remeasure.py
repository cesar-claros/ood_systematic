"""RN18 handoff-replication plan v3, PHASE 1: D-R4 NC1 re-measurement
(HPC, GPU; TRAINING FEATURES ONLY; opens no test set, score, or outcome).

Per checkpoint in manifests/expected_panel.json (resolved on disk):
- deterministic PRIMARY view (v3 section 5.1): the source-training
  examples under the source's test transform, canonical dataset order,
  no shuffle; feature model, papyan panel, geometry record;
  `nc1_corrected` (= papyan var_collapse, the estimator that filled the
  VGG corrected column), `s_dict` = (C-1)/sqrt(C*NC1), `snr_iso` = the
  geometry record's R/sigma_iso (v3 section 5.2 names);
- AUG-VIEW sensitivity: the datamodule's train dataset (train
  augmentations) in canonical order, no shuffle, num_workers 0, torch /
  numpy / random seeded 20260908 before the pass, so every checkpoint of
  a source sees the identical augmented view (verified by the label hash
  and a first-batch checksum);
- 500 common class-stratified bootstrap resamples of the deterministic
  view (seed 1201, drawn from the label vector, so identical across
  checkpoints of a source when the label hash matches): NC1 per
  resample with the SAME arithmetic as papyan_metrics (equal-class
  Sigma_B = M'M/C, sample-weighted Sigma_W, hermitian pinv rcond 1e-6);
  the unweighted case is asserted equal to the panel value.
Writes <slug>.json + <slug>.npz (centered class means, global mean,
radii, Sigma_W, bootstrap NC1 array) under outputs/phase1_nc1/.
Resumable; sharded; FAILED_ isolation.

Usage (HPC, inside the container, from code/):
    python rn18_handoff_replication/phase1_nc1_remeasure.py --list
    python rn18_handoff_replication/phase1_nc1_remeasure.py [--shard k/n] [--bootstrap 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.extract_pool_coords import forward_loader
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)

MANIFEST = Path("rn18_handoff_replication/manifests/expected_panel.json")
OUT_DIR_DEFAULT = "rn18_handoff_replication/outputs/phase1_nc1"
PREFIXES = ("", "fd-shifts/")
AUG_SEED, BOOT_SEED = 20260908, 1201
SCHEMA = 1


def nc1_weighted(H, y, w, n_classes, gram_fn) -> float:
    """NC1 under integer example weights w (bootstrap multiplicities),
    mirroring papyan_metrics' arithmetic exactly for w == 1."""
    n = float(w.sum())
    gmean = (w @ H) / n
    means = np.zeros((n_classes, H.shape[1]))
    counts = np.zeros(n_classes)
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        wc = w[idx]
        counts[c] = wc.sum()
        means[c] = (wc @ H[idx]) / counts[c]
    gram = gram_fn(w)                              # H' diag(w) H
    sigma_w = (gram - (means.T * counts) @ means) / n
    m = means - gmean
    sigma_b = m.T @ m / n_classes
    sigma_b = (sigma_b + sigma_b.T) / 2.0
    return float(np.trace(sigma_w @ np.linalg.pinv(sigma_b, rcond=1e-6,
                                                   hermitian=True))
                 / n_classes)


def make_gram_fn(H: np.ndarray, use_cuda: bool):
    try:
        import torch
        dev = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        Ht = torch.as_tensor(H, dtype=torch.float64, device=dev)

        def gram(w):
            wt = torch.as_tensor(w, dtype=torch.float64, device=dev)
            return ((Ht * wt[:, None]).T @ Ht).cpu().numpy()
        return gram
    except Exception:  # noqa: BLE001 - numpy fallback
        return lambda w: (H * w[:, None]).T @ H


def stratified_weights(rng, y: np.ndarray, n_classes: int) -> np.ndarray:
    w = np.zeros(len(y))
    for c in range(n_classes):
        idx = np.flatnonzero(y == c)
        draw = rng.choice(idx, size=len(idx), replace=True)
        w += np.bincount(draw, minlength=len(y))
    return w


def view_record(h, y, w_np, b_np, n_classes) -> tuple[dict, object]:
    fm = fit_feature_model(h, y, n_classes)
    pap = papyan_metrics(w_np, fm)
    geo = geometry_record(w_np, b_np, fm)
    assert "snr" in geo, "geometry_record lacks the historical snr field"
    nc1 = float(pap["var_collapse"])
    rec = {"papyan": pap, "geometry": geo, "nc1_corrected": nc1,
           "s_dict": float((n_classes - 1) / np.sqrt(n_classes * max(nc1, 1e-300))),
           "snr_iso": float(geo["snr"]), "n": int(len(h)),
           "labels_sha256": hashlib.sha256(y.astype(np.int64).tobytes()).hexdigest()}
    return rec, fm


def first_batch_checksum(loader) -> float:
    x0, _ = next(iter(loader))
    return float(x0.double().sum())


def seed_all(s: int) -> None:
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def extract_one(cell: dict, root: Path, out_dir: Path, use_cuda: bool,
                n_boot: int) -> None:
    import torch
    from torch.utils.data import DataLoader
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.loaders.dataset_collection import get_dataset
    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = None
    for p in PREFIXES:
        if (root / p / cell["model_path"]).is_dir():
            model_path = f"{p}{cell['model_path']}"
            break
    assert model_path, f"not on disk: {cell['model_path']}"
    slug = cell["model_path"].replace("/", "__")
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]

    # --- deterministic primary view ---------------------------------------
    ds_det = get_dataset(name=datamodule.dataset_name, root=datamodule.data_dir,
                         train=True, download=True,
                         target_transform=datamodule.target_transforms.get("train"),
                         transform=datamodule.augmentations["test"],
                         kwargs=datamodule.dataset_kwargs)
    ld_det = DataLoader(ds_det, batch_size=datamodule.batch_size, shuffle=False,
                        num_workers=datamodule.num_workers,
                        pin_memory=datamodule.pin_memory)
    det_ck = first_batch_checksum(ld_det)
    logger.info(f"{slug}: forward train (deterministic view)")
    ev = forward_loader(model, ld_det)
    h = ev["encoded"].cpu().numpy().astype(np.float32)
    y = ev["labels"].cpu().numpy().astype(np.int64)
    counts = np.bincount(y, minlength=n_classes)
    assert counts.min() > 0, "empty train class"
    det, fm = view_record(h, y, w_np, b_np, n_classes)
    det["first_batch_checksum"] = det_ck

    # --- bootstrap on the deterministic view ------------------------------
    H = h.astype(np.float64)
    gram = make_gram_fn(H, use_cuda)
    base = nc1_weighted(H, y, np.ones(len(y)), n_classes, gram)
    rel = abs(base - det["nc1_corrected"]) / max(det["nc1_corrected"], 1e-300)
    assert rel < 1e-8, f"bootstrap arithmetic mismatch: {base} vs {det['nc1_corrected']}"
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(n_boot)
    for bi in range(n_boot):
        boot[bi] = nc1_weighted(H, y, stratified_weights(rng, y, n_classes),
                                n_classes, gram)
    del H, gram
    npz = {"class_means_centered": fm.class_means, "global_mean": fm.global_mean,
           "radii": fm.radii, "sigma_w": fm.sigma_w, "boot_nc1": boot}
    del h, ev

    # --- AUG-VIEW sensitivity ---------------------------------------------
    ld_aug = DataLoader(datamodule.train_dataset, batch_size=datamodule.batch_size,
                        shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(AUG_SEED))
    seed_all(AUG_SEED)
    aug_ck = first_batch_checksum(ld_aug)
    seed_all(AUG_SEED)
    logger.info(f"{slug}: forward train (AUG-VIEW)")
    ev = forward_loader(model, ld_aug)
    ha = ev["encoded"].cpu().numpy().astype(np.float32)
    ya = ev["labels"].cpu().numpy().astype(np.int64)
    aug, _ = view_record(ha, ya, w_np, b_np, n_classes)
    aug["first_batch_checksum"] = aug_ck
    del ha, ev

    rec = {"schema_phase1": SCHEMA, **{k: cell[k] for k in cell},
           "resolved_model_path": model_path, "slug": slug, "study": study_name,
           "n_classes": n_classes, "dim": int(fm.global_mean.shape[0]),
           "n_train": int(len(y)),
           "class_support": {"min": int(counts.min()),
                             "median": float(np.median(counts)),
                             "n_empty": int((counts == 0).sum())},
           "deterministic_view": det, "aug_view": aug,
           "bootstrap": {"B": n_boot, "seed": BOOT_SEED,
                         "sd_log_nc1": float(np.std(np.log(boot), ddof=1)),
                         "labels_sha256": det["labels_sha256"]},
           "runtime_sec": round(time.time() - t0, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(out_dir / f"{slug}.npz", **npz)
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    logger.info(f"{slug}: NC1 det {det['nc1_corrected']:.4f} aug "
                f"{aug['nc1_corrected']:.4f} ({rec['runtime_sec']}s)")
    del model, module, datamodule
    if use_cuda:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--shard", type=str, default="1/1")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    cells = json.loads(MANIFEST.read_text())["cells"]
    k, n = (int(x) for x in args.shard.split("/"))
    cells = cells[k - 1::n]
    out_dir = Path(args.out_dir)
    todo = [c for c in cells if not (
        out_dir / f"{c['model_path'].replace('/', '__')}.json").exists()]
    present = [c for c in todo if any((root / p / c["model_path"]).is_dir()
                                      for p in PREFIXES)]
    print(f"[phase1] shard {args.shard}: {len(cells)} cells, {len(todo)} to run, "
          f"{len(todo) - len(present)} not on disk", flush=True)
    if args.list:
        for c in todo:
            print("  ", c["model_path"], "" if c in present else "[MISSING]")
        return
    failures = 0
    for i, c in enumerate(present, 1):
        print(f"[phase1] {i}/{len(present)}: {c['model_path']}", flush=True)
        try:
            extract_one(c, root, out_dir, args.use_cuda, args.bootstrap)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{c['model_path'].replace('/', '__')}.json").write_text(
                json.dumps({"model_path": c["model_path"],
                            "error": traceback.format_exc()}, indent=1))
            print(f"[phase1] FAILED {c['model_path']}", flush=True)
    print(f"[phase1] done: {len(present) - failures} ok, {failures} failed", flush=True)


if __name__ == "__main__":
    main()
