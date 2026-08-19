"""Geometry/nuisance extraction for the Pilot 1 manipulation report (HPC).

Blinded stage of code/pilot1/MANIFEST.md section 3: forwards the TRAIN and
VAL splits only (no OOD data is ever loaded), measures the geometry vector
(eight Papyan coordinates + the G extensions) and the nuisance vector, and
writes one JSON per (experiment, checkpoint). The train forward uses the
same dataloader convention as the paper's NC protocol.

Usage (from code/, inside the frozen container):
    python nc_csf_predictivity/interventions/extract_manipulation.py \
        --group cifar100_intervention [--cadence] [--use_cuda] \
        [--out_dir nc_csf_predictivity/interventions/geometry]
Environment: EXPERIMENT_ROOT_DIR, DATASET_ROOT_DIR.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fd_shifts import logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
from fd_shifts.models import get_model
from fd_shifts.utils import exp_utils

from pilot0.geometry import fit_feature_model, geometry_record, papyan_metrics
from src import utils

NAME_RE = re.compile(r"(?P<kind>etfreg|etfhard)_bb\w+_do\d+_run(?P<run>\d+)"
                     r"_lam(?P<lam>[-\w.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pilot 1 manipulation-stage geometry extraction")
    parser.add_argument("--group", type=str, default="cifar100_intervention")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Explicit experiment names; default: all in "
                             "the group directory")
    parser.add_argument("--out_dir", type=str,
                        default="nc_csf_predictivity/interventions/geometry")
    parser.add_argument("--cadence", action="store_true",
                        help="Also measure every cadence checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-measure checkpoints with existing JSONs "
                             "(default: skip, making reruns resumable)")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


@torch.no_grad()
def forward_split(module, loader, device: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encoded features, labels, logits for one dataloader."""
    feats, labels, logits = [], [], []
    for x, y in loader:
        z = module.model.forward_features(x.to(device))
        g = module.model.head(z)
        feats.append(z.cpu().numpy().astype(np.float32))
        labels.append(y.numpy())
        logits.append(g.cpu().numpy().astype(np.float32))
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(logits))


def ece_15bin(logits: np.ndarray, labels: np.ndarray) -> float:
    """Standard 15-bin expected calibration error."""
    g = logits.astype(np.float64)
    p = np.exp(g - g.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    conf = p.max(1)
    correct = (g.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def measure_checkpoint_state(module, datamodule, device: str,
                             n_classes: int) -> dict:
    """Geometry + nuisance record for the module's current weights."""
    module.eval()
    h_tr, y_tr, g_tr = forward_split(module, datamodule.train_dataloader(),
                                     device)
    _, y_val, g_val = forward_split(module, datamodule.val_dataloader(),
                                    device)
    for label, arr in (("train features", h_tr), ("train logits", g_tr),
                       ("val logits", g_val)):
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite values in {label}; checkpoint is "
                             "a candidate for the manifest failure rule")
    w, b = utils.get_model_and_last_layer(module, "intervention",
                                          return_model=False)
    w_np = w.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)

    model = fit_feature_model(h_tr, y_tr, n_classes)
    record = geometry_record(w_np, b_np, model)
    record.update(papyan_metrics(w_np, model))
    log_p = g_val.astype(np.float64)
    log_p -= log_p.max(1, keepdims=True)
    nll = float(np.mean(np.log(np.exp(log_p).sum(1))
                        - log_p[np.arange(len(y_val)), y_val]))
    record.update({
        "train_acc": float((g_tr.argmax(1) == y_tr).mean()),
        "val_acc": float((g_val.argmax(1) == y_val).mean()),
        "val_nll": nll,
        "val_ece15": ece_15bin(g_val, y_val),
        "weight_norm_fro": float(np.linalg.norm(w_np)),
        "n_train": len(y_tr),
        "dim": int(h_tr.shape[1]),
    })
    return record


def main() -> None:
    args = parse_args()
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    names = args.experiments or sorted(
        p.name for p in (exp_root / args.group).iterdir() if p.is_dir())
    for name in names:
        meta = NAME_RE.match(name)
        if meta is None:
            logger.warning(f"Skipping unparseable experiment name: {name}")
            continue
        path = f"{args.group}/{name}"
        logger.info(f"=== {path} ===")
        cf = utils.get_conf(path, "intervention")
        datamodule = FDShiftsDataLoader(cf)
        datamodule.setup()
        module = get_model(cf.model.name)(cf)

        ckpts = {"last": exp_utils._get_path_to_best_ckpt(
            cf.exp.dir, "last", cf.test.selection_mode)}
        if args.cadence:
            for p in sorted(Path(cf.exp.dir).glob(
                    "version_*/cadence_epoch=*.ckpt")):
                ckpts[p.stem] = str(p)

        for tag, ckpt_path in ckpts.items():
            out_path = out_dir / f"{name}__{tag}.json"
            fail_path = out_dir / f"{name}__{tag}.FAILED.json"
            if out_path.exists() and not args.overwrite:
                logger.info(f"skipping existing {out_path}")
                continue
            try:
                module.load_only_state_dict(ckpt_path, device="cpu")
                module.model.encoder.disable_dropout()
                module.to(device)
                record = measure_checkpoint_state(
                    module, datamodule, device, int(cf.data.num_classes))
            except (np.linalg.LinAlgError, ValueError, RuntimeError) as err:
                logger.error(f"measurement failed for {path} [{tag}]: {err}")
                fail_path.write_text(json.dumps(
                    {"experiment": path, "checkpoint": tag,
                     "error": str(err)}, indent=1))
                continue
            record.update({
                "experiment": path, "checkpoint": tag,
                "kind": meta["kind"], "run": int(meta["run"]),
                "lam": meta["lam"],
            })
            out_path.write_text(json.dumps(record, indent=1))
            fail_path.unlink(missing_ok=True)
            logger.info(f"wrote {out_path} (self_duality="
                        f"{record['self_duality']:.4f}, val_acc="
                        f"{record['val_acc']:.4f})")


if __name__ == "__main__":
    main()
