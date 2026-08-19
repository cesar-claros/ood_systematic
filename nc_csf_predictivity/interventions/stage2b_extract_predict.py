"""Stage 2b, HPC side: H coordinates + frozen plug-in predictions per model.

Manifest Addendum A item 3: for each Pilot 1 model, forward the train
split and the eight OOD sets (activations only -- NO detector score is
computed anywhere in this stage), measure the frozen H coordinates
(pilot0.ood_coords), and generate the frozen plug-in AUROC predictions
for the six endpoint scores (MLS, Energy, MSR, CTM_head, CTM_mean, Maha)
under both noise arms. One JSON per model; `stage2b_signs.py` aggregates
them into the committed directional-prediction table.

Usage (from code/, inside the frozen container):
    python nc_csf_predictivity/interventions/stage2b_extract_predict.py \
        --group cifar100_intervention [--use_cuda] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fd_shifts import logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
from fd_shifts.models import get_model
from fd_shifts.utils import exp_utils

from nc_csf_predictivity.interventions.extract_manipulation import (
    NAME_RE,
    forward_split,
)
from pilot0.geometry import fit_feature_model
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer
from pilot0.theory import (
    HeadContext,
    NoiseModel,
    predicted_aurocs,
    predicted_ctm_mean_auroc,
    predicted_maha_auroc,
)
from src import utils

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


class _ForwardAdapter:
    """Minimal TrainedModule-like interface for compute_model_evaluations."""

    def __init__(self, module, cf, device: str):
        self.module = module
        self.study_name = "intervention"
        self.dataset = cf.data.dataset
        self.device = device

    @torch.no_grad()
    def __call__(self, batch, _idx):
        x = batch[0]
        z = self.module.model.forward_features(x.to(self.device))
        return {"encoded": z.cpu()}


def predictions_for_set(h_ood: np.ndarray, model, means_unc: np.ndarray,
                        class_freq: np.ndarray, precision: np.ndarray,
                        ctx: HeadContext, dim: int) -> dict:
    """H coordinates and six-score predictions (iso + emp) for one OOD set."""
    h64 = h_ood.astype(np.float64)
    m_o = h64.mean(0)
    resid = h64 - m_o
    sigma_o = float(np.sqrt((resid**2).sum(1).mean() / dim))
    cov_o = resid.T @ resid / len(resid)

    covs = {"iso": (model.sigma_iso**2 * np.eye(dim),
                    sigma_o**2 * np.eye(dim)),
            "emp": (model.sigma_w, cov_o)}
    noise_id = {"iso": NoiseModel.isotropic(model.sigma_iso, ctx, dim),
                "emp": NoiseModel.empirical(model.sigma_w, ctx)}
    noise_ood = {"iso": NoiseModel.isotropic(sigma_o, ctx, dim),
                 "emp": NoiseModel.empirical(cov_o, ctx)}

    preds: dict[str, dict[str, float]] = {}
    for arm in ("iso", "emp"):
        cov_id_arm, cov_ood_arm = covs[arm]
        head = predicted_aurocs(means_unc, class_freq, noise_id[arm],
                                m_o, noise_ood[arm], ctx)
        head["CTM_mean"] = predicted_ctm_mean_auroc(
            means_unc, class_freq, cov_id_arm, m_o, cov_ood_arm)
        head["Maha"] = predicted_maha_auroc(
            means_unc, precision, cov_id_arm, m_o, cov_ood_arm)
        preds[arm] = {k: float(v) for k, v in head.items()}
    return {"n_ood": len(h64),
            "h_coords": estimate_ood_coords(h_ood, model),
            "preds": preds}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2b: H coordinates + plug-in predictions")
    parser.add_argument("--group", type=str, default="cifar100_intervention")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--out_dir", type=str,
                        default="nc_csf_predictivity/interventions/stage2b")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
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
        out_path = out_dir / f"{name}.json"
        if out_path.exists() and not args.overwrite:
            logger.info(f"skipping existing {out_path}")
            continue
        path = f"{args.group}/{name}"
        logger.info(f"=== {path} ===")
        try:
            cf = utils.get_conf(path, "intervention")
            datamodule = FDShiftsDataLoader(cf)
            datamodule.setup()
            module = get_model(cf.model.name)(cf)
            module.load_only_state_dict(exp_utils._get_path_to_best_ckpt(
                cf.exp.dir, "last", cf.test.selection_mode), device="cpu")
            module.model.encoder.disable_dropout()
            module.to(device)
            module.eval()

            h_tr, y_tr, _ = forward_split(
                module, datamodule.train_dataloader(), device)
            n_classes = int(cf.data.num_classes)
            model = fit_feature_model(h_tr, y_tr, n_classes)
            means_unc = model.class_means + model.global_mean
            maha = MahalanobisScorer(h_tr, y_tr, n_classes)
            w, b = utils.get_model_and_last_layer(module, "intervention",
                                                  return_model=False)
            ctx = HeadContext.from_head(w.detach().cpu().numpy(),
                                        b.detach().cpu().numpy())
            adapter = _ForwardAdapter(module, cf, device)
            dim = h_tr.shape[1]

            record = {"experiment": path, "kind": meta["kind"],
                      "run": int(meta["run"]), "lam": meta["lam"],
                      "sets": {}}
            for mode, test_set in OOD_TEST_SETS.items():
                ev = utils.compute_model_evaluations(adapter, datamodule,
                                                     test_set)
                h_ood = ev["encoded"].numpy().astype(np.float32)
                record["sets"][mode] = predictions_for_set(
                    h_ood, model, means_unc, model.class_freq,
                    maha.precision, ctx, dim)
                logger.info(f"  {mode}: n={record['sets'][mode]['n_ood']}")
        except (np.linalg.LinAlgError, ValueError, RuntimeError,
                FileNotFoundError) as err:
            logger.error(f"stage 2b failed for {path}: {err}")
            (out_dir / f"{name}.FAILED.json").write_text(json.dumps(
                {"experiment": path, "error": str(err)}, indent=1))
            continue
        out_path.write_text(json.dumps(record, indent=1))
        logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
