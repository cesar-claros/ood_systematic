"""Pilot 0 feature extraction for one checkpoint (HPC side).

Loads an fd-shifts experiment exactly as x6_spectral/measure_checkpoint.py
does (no surgeries, MCD stripped), forwards the train split, the ID test
set, and every paper OOD set for the source, and writes one compressed NPZ
plus a JSON sidecar that pilot0/run_pilot0.py consumes locally.

Usage (from code/, inside the campaign container):
    python pilot0/extract_pilot0.py --model_path=<experiment> \
        [--use_cuda] [--out_dir=pilot0/caches]
Environment: EXPERIMENT_ROOT_DIR, DATASET_ROOT_DIR (code/.env), optional
CSF_BATCH_SIZE / CSF_NUM_WORKERS as in csf_fit.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fd_shifts import logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader

from src import utils
from src.trained_module import TrainedModule
from x6_spectral.measure_checkpoint import load_model

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot 0 feature extraction for one checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Experiment folder, e.g. cifar100_paper_sweep/"
                             "confidnet_bbvgg13_do0_run1_rew2.2")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--out_dir", type=str, default="pilot0/caches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_path.replace("/", "__")

    t0 = time.time()
    cf, module, study_name = load_model(args.model_path, use_cuda)
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

    arrays: dict[str, np.ndarray] = {"w": w_np, "b": b_np}
    ood_sets: list[str] = []

    logger.info("Forward pass: train split")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    arrays["h_train"] = ev["encoded"].cpu().numpy().astype(np.float32)
    arrays["y_train"] = ev["labels"].cpu().numpy().astype(np.int64)

    logger.info("Forward pass: iid test (test_1)")
    ev = utils.compute_model_evaluations(model, datamodule, "test_1")
    arrays["h_iid_test"] = ev["encoded"].cpu().numpy().astype(np.float32)
    arrays["y_iid_test"] = ev["labels"].cpu().numpy().astype(np.int64)
    arrays["logits_iid_test"] = ev["logits"].cpu().numpy().astype(np.float32)

    for mode, test_set in OOD_TEST_SETS.items():
        logger.info(f"Forward pass: {mode} ({test_set})")
        try:
            ev = utils.compute_model_evaluations(model, datamodule, test_set)
        except (FileNotFoundError, NotImplementedError) as err:
            logger.error(f"Skipping {mode}: {err}")
            continue
        arrays[f"h_{mode}"] = ev["encoded"].cpu().numpy().astype(np.float32)
        ood_sets.append(mode)

    np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    meta = {"slug": slug, "model_path": args.model_path,
            "study": study_name, "dataset": str(cf.data.dataset),
            "n_classes": n_classes, "dim": int(arrays["h_train"].shape[1]),
            "ood_sets": ood_sets,
            "runtime_sec": round(time.time() - t0, 1)}
    (out_dir / f"{slug}.json").write_text(json.dumps(meta, indent=1))
    logger.info(f"Wrote {out_dir / slug}.npz ({len(ood_sets)} OOD sets, "
                f"{meta['runtime_sec']}s)")


if __name__ == "__main__":
    main()
