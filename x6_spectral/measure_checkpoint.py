"""Tier-A spectral measurement for one checkpoint (X6 campaign, stage 1).

Loads an fd-shifts experiment exactly as csf_fit.py does (no surgeries, MCD
confids stripped so only deterministic single-pass forwards run), computes
penultimate activations on the train and val splits, and writes:
  <out_dir>/<slug>.json  ID-side diagnostics (centered-covariance census by
      default, all-sample and correct-only arms, optional standardized
      robustness arm), Tier A predictions, spike alignments, metadata;
  <out_dir>/<slug>.npz   the objects Tier B needs later without re-forwarding
      ID data: feature means, top eigenvectors and eigenvalues of the
      correct-only arm, and the classifier head (w, b).

Runs inside the campaign container on the HPC (src.utils.get_conf resolves
the experiments root there). Measurement only: no outcome table is read, so
running this before the rule freeze is safe.

Usage (from code/):
    python x6_spectral/measure_checkpoint.py --model_path=<experiment> \
        [--use_cuda] [--out_dir=x6_spectral/outputs] [--k_class=12] \
        [--het_splits=2] [--skip_standardized]
Environment: EXPERIMENT_ROOT_DIR, DATASET_ROOT_DIR (code/.env), optional
CSF_BATCH_SIZE and CSF_NUM_WORKERS overrides as in csf_fit.py.
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
from fd_shifts.models import get_model
from fd_shifts.utils import exp_utils

from src import utils
from src.trained_module import TrainedModule

from spectra_campaign_harness import measure, tier_a
from spectral_diagnostics import spike_alignments

import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="X6 Tier-A spectral measurement for one checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Experiment folder, e.g. "
                             "cifar10_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True, help="Use GPU when available")
    parser.add_argument("--out_dir", type=str, default="x6_spectral/outputs",
                        help="Output directory for JSON/NPZ")
    parser.add_argument("--k_class", type=int, default=12,
                        help="Per-class subspace rank for the heterogeneity "
                             "statistic")
    parser.add_argument("--het_splits", type=int, default=2,
                        help="Splits for class_projector_heterogeneity "
                             "(runtime scales with C * splits)")
    parser.add_argument("--skip_standardized", action="store_true",
                        help="Skip the standardized robustness arm")
    return parser.parse_args()


def load_model(path: str, use_cuda: bool) -> tuple:
    """Mirror csf_fit.py loading, without surgeries and without MCD."""
    study_name = utils.get_study_name(path)
    do_enabled = utils.is_dropout_enabled(path)
    cf = utils.get_conf(path, study_name)
    ckpt_path = exp_utils._get_path_to_best_ckpt(
        cf.exp.dir, "last", cf.test.selection_mode)
    if "super" in path:
        cf.eval.query_studies.noise_study = ["corrupt_cifar100"]
        cf.eval.query_studies.new_class_study = [
            "cifar10", "svhn", "tinyimagenet_resize"]
        if do_enabled and "vgg" in path:
            logger.info("Disabling average pooling for VGG-13 supercifar "
                        "with dropout enabled (matches csf_fit.py)")
            cf.model.avg_pool = False
    if "vit" in path:
        cf.data.num_workers = 12

    module = get_model(cf.model.name)(cf)
    module.load_only_state_dict(ckpt_path, device="cpu")
    if study_name == "confidnet":
        module.backbone.encoder.disable_dropout()
        module.network.encoder.disable_dropout()
    elif study_name in ("devries", "dg"):
        module.model.encoder.disable_dropout()
    elif study_name == "vit":
        module.disable_dropout()
    else:
        raise NotImplementedError(study_name)

    for split in ("test", "val", "train"):
        confids = getattr(cf.eval.confidence_measures, split, None)
        if confids is not None:
            setattr(cf.eval.confidence_measures, split,
                    [c for c in confids if "mcd" not in c])

    if study_name == "vit" and not use_cuda:
        cf.trainer.batch_size = 128
    if os.environ.get("CSF_BATCH_SIZE"):
        logger.info(f"CSF_BATCH_SIZE override: {os.environ['CSF_BATCH_SIZE']}")
        cf.trainer.batch_size = int(os.environ["CSF_BATCH_SIZE"])
    if os.environ.get("CSF_NUM_WORKERS"):
        cf.data.num_workers = int(os.environ["CSF_NUM_WORKERS"])
    return cf, module, study_name


def to_f64(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def jsonable(obj):
    """Recursively convert numpy objects for json.dump."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def prune_census(diag: dict) -> dict:
    """Drop the full eigenvalue vector from JSON (kept in the NPZ)."""
    out = dict(diag)
    census = dict(out["census"])
    census.pop("eigs", None)
    out["census"] = census
    return out


def main() -> None:
    args = parse_args()
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    if args.use_cuda and not use_cuda:
        logger.warning("CUDA requested but unavailable; running on CPU")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_path.replace("/", "__")

    t0 = time.time()
    cf, module, study_name = load_model(args.model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)

    logger.info("Forward pass: train split")
    eval_train = utils.compute_model_evaluations(model, datamodule, "train")
    logger.info("Forward pass: val split")
    eval_val = utils.compute_model_evaluations(model, datamodule, "val")
    t_forward = time.time() - t0

    h_train = to_f64(eval_train["encoded"])
    y_train = to_f64(eval_train["labels"]).astype(np.int64)
    preds_train = to_f64(eval_train["logits"]).argmax(1)
    correct_mask = preds_train == y_train
    val_acc = float((to_f64(eval_val["logits"]).argmax(1)
                     == to_f64(eval_val["labels"]).astype(np.int64)).mean())

    if study_name == "vit":
        w, b = utils.get_model_and_last_layer(module, study_name,
                                              return_model=False)
    else:
        _, w, b = utils.get_model_and_last_layer(module, study_name)
    w_np, b_np = to_f64(w), to_f64(b)
    n_classes = int(cf.data.num_classes)
    w_np, b_np = w_np[:n_classes], b_np[:n_classes]

    t1 = time.time()
    logger.info("Diagnostics: correct-only arm (implementation-faithful)")
    h_cor, y_cor = h_train[correct_mask], y_train[correct_mask]
    diag_correct = measure(h_cor, y_cor, w_np, n_classes,
                           k_class=args.k_class, het_splits=args.het_splits)
    tier_a_out = tier_a(diag_correct, id_val_accuracy=val_acc)
    logger.info("Diagnostics: all-sample arm")
    diag_all = measure(h_train, y_train, w_np, n_classes,
                       k_class=args.k_class, het_splits=args.het_splits)
    diag_std = None
    if not args.skip_standardized:
        logger.info("Diagnostics: standardized robustness arm")
        diag_std = measure(h_train, y_train, w_np, n_classes,
                           k_class=args.k_class, het_splits=args.het_splits,
                           standardize=True)
    aligns = spike_alignments(h_cor, y_cor, w_np, n_classes)
    t_diag = time.time() - t1

    centered = h_cor - h_cor.mean(0)
    eigvals, eigvecs = np.linalg.eigh(centered.T @ centered / len(h_cor))
    k_save = min(h_cor.shape[1], (n_classes - 1) + 64)
    np.savez_compressed(
        out_dir / f"{slug}.npz",
        mean_correct=h_cor.mean(0), mean_all=h_train.mean(0),
        eigvals_correct=eigvals, top_eigvecs_correct=eigvecs[:, -k_save:],
        w=w_np, b=b_np)

    record = {
        "model_path": args.model_path, "study": study_name,
        "dataset": str(cf.data.dataset), "n_classes": n_classes,
        "n_train": int(len(h_train)), "dim": int(h_train.shape[1]),
        "effective_n_correct": int(correct_mask.sum()),
        "train_acc": float(correct_mask.mean()), "id_val_accuracy": val_acc,
        "k_class": args.k_class,
        "arms": {
            "correct_only": prune_census(diag_correct),
            "all": prune_census(diag_all),
            "all_standardized": prune_census(diag_std) if diag_std else None,
        },
        "tier_a": tier_a_out,
        "spike_alignments": {k: v[:40] for k, v in aligns.items()},
        "runtime_sec": {"forward": round(t_forward, 1),
                        "diagnostics": round(t_diag, 1)},
    }
    with open(out_dir / f"{slug}.json", "w") as fh:
        json.dump(jsonable(record), fh, indent=1)
    logger.info(f"Wrote {out_dir / slug}.json and .npz "
                f"(forward {t_forward:.0f}s, diagnostics {t_diag:.0f}s)")


if __name__ == "__main__":
    main()
