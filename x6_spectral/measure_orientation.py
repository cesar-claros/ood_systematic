"""Stage-2 Tier-B orientation measurement for one checkpoint (X6 campaign).

For each OOD evaluation dataset of the checkpoint's source: collect the
first N_DRAWS x BATCH_PER_DRAW samples deterministically (loader order, no
shuffling), estimate the displacement orientation per draw against the
stage-1 projector, apply the pre-registered r2-tierB rules, and run the
registered batch-trial scores with validation features as the ID reference.
All constants and rules are pre-registered in FREEZE.md (Tier-B addendum)
before any orientation was measured; this script is plumbing around
spectra_campaign_harness (the math) and measure_checkpoint.load_model (the
loading). Explicitly OOD-side: reads evaluation IMAGES, never outcome
tables, so dev-pool runs are calibration and held-out runs happen only
after the freeze tag.

Reads <out_dir>/<slug>.npz (stage 1) and writes
<out_dir>/orientation/<slug>.json.

Usage (from code/):
    python x6_spectral/measure_orientation.py --model_path=<experiment> \
        [--use_cuda] [--out_dir=x6_spectral/outputs]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from fd_shifts import logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader

from src import utils
from src.trained_module import TrainedModule

from src.csfs._utils import TorchStandardScaler  # noqa: F401 (unpickling)
from torch_pca import PCA  # noqa: F401 (unpickling of saved PF params)

from measure_checkpoint import jsonable, load_model, to_f64
from spectra_campaign_harness import (batch_trial, deployed_pf_rank,
                                      deployed_trial, estimate_orientation,
                                      make_backprojector, tier_b)

BATCH_PER_DRAW = 128
N_DRAWS = 5
Q_COVERAGE = 0.90
ID_REF_MAX = 4000
MODEL_OPTS = "_RW0_RF0_ASHNone"


def load_deployed_pf(cf) -> dict:
    """Load the deployed ProjectionFiltering params saved by csf_fit (r7).

    Returns numpy back-projection ingredients for the global and per-class
    estimators: exact fitted components, the tuned variance_explained, and
    the deployed component-count rule, so the trial replicates
    get_backprojection verbatim instead of reconstructing a projector.
    """
    base = Path(cf.exp.dir) / "params"
    out: dict = {"found_global": False, "found_class": False}
    g_path = base / f"ProjectionFiltering_global_params{MODEL_OPTS}.pt"
    if g_path.exists():
        params = torch.load(g_path, map_location="cpu")
        mean = params["tss"].mean_.numpy().astype(np.float64)[0]
        pca = params["pca_estimator"]
        comps = pca.components_.numpy().astype(np.float64)
        n_g = deployed_pf_rank(pca.explained_variance_ratio_.numpy(),
                               float(params["variance_explained"]))
        out.update(found_global=True, global_mean=mean, global_comps=comps,
                   n_global=int(n_g),
                   variance_explained=float(params["variance_explained"]))
    c_path = base / f"ProjectionFiltering_class_params{MODEL_OPTS}.pt"
    if c_path.exists():
        params = torch.load(c_path, map_location="cpu")
        v = float(params["variance_explained"])
        class_bp = []
        for tss_c, pca_c in zip(params["tss"], params["pca_estimator"]):
            mean_c = tss_c.mean_.numpy().astype(np.float64)[0]
            comps_c = pca_c.components_.numpy().astype(np.float64)
            n_c = deployed_pf_rank(
                pca_c.explained_variance_ratio_.numpy(), v)
            class_bp.append((mean_c, comps_c, int(n_c)))
        out.update(found_class=True, class_bp=class_bp,
                   variance_explained_class=v)
    return out


def refit_maha(z: np.ndarray, labels: np.ndarray, n_classes: int,
               fallback_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Val-side tied-covariance Mahalanobis refit (rcond as deployed)."""
    means = np.stack([z[labels == c].mean(0) if (labels == c).any()
                      else fallback_mean for c in range(n_classes)])
    centered = np.concatenate([z[labels == c] - means[c]
                               for c in range(n_classes)
                               if (labels == c).any()])
    cov = centered.T @ centered / max(len(centered), 1)
    return means, np.linalg.pinv(cov, hermitian=True, rcond=1e-6)
NAME_NORMALIZATION = {
    "tinyimagenet_resize": "tinyimagenet", "tinyimagenet_384": "tinyimagenet",
    "cifar10_384": "cifar10", "cifar100_384": "cifar100",
    "svhn_384": "svhn",
}
FAR_OOD = {6: ("LSUN", "lsun cropped"), 7: ("LSUN_resize", "lsun resize"),
           8: ("iSUN", "isun"), 9: ("dtd/images", "textures"),
           10: ("places365", "places365")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="X6 Tier-B orientation measurement for one checkpoint")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--out_dir", type=str, default="x6_spectral/outputs")
    return parser.parse_args()


def far_ood_loader(datamodule, folder: str, resize_img: tuple[int, int]
                   ) -> DataLoader:
    """ImageFolder loaders for test sets 6-10, mirroring src.utils."""
    transform = transforms.Compose([
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    root = datamodule.data_root_dir
    for part in folder.split("/"):
        root = root.joinpath(part)
    dataset = torchvision.datasets.ImageFolder(root, transform=transform)
    return DataLoader(dataset, batch_size=datamodule.batch_size,
                      num_workers=datamodule.num_workers, shuffle=False)


def collect_features(model: TrainedModule, loader: DataLoader,
                     n_max: int) -> np.ndarray:
    """Forward only enough batches to collect n_max feature rows."""
    chunks, n_have = [], 0
    for i, batch in enumerate(islice(loader, 0, None)):
        out = model(batch, i)
        chunks.append(to_f64(out["encoded"]))
        n_have += len(chunks[-1])
        if n_have >= n_max:
            break
    if not chunks:
        return np.empty((0, 0))
    return np.concatenate(chunks)[:n_max]


def main() -> None:
    args = parse_args()
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    out_dir = Path(args.out_dir)
    slug = args.model_path.replace("/", "__")
    npz_path = out_dir / f"{slug}.npz"
    assert npz_path.exists(), f"stage-1 artifact missing: {npz_path}"
    ori_dir = out_dir / "orientation"
    ori_dir.mkdir(parents=True, exist_ok=True)

    art = np.load(npz_path)
    eigvecs = art["top_eigvecs_correct"]
    eigvals_desc = art["eigvals_correct"][::-1]
    w_full, b_full = art["w"], art["b"]
    dim, k_save = eigvecs.shape

    cum = np.cumsum(eigvals_desc) / max(eigvals_desc.sum(), 1e-30)
    q90 = int(np.searchsorted(cum, Q_COVERAGE) + 1)
    q_used = min(q90, k_save)
    coverage_at_q = float(cum[q_used - 1])
    projector = eigvecs[:, -q_used:]

    t0 = time.time()
    cf, module, study_name = load_model(args.model_path, use_cuda)
    n_classes = int(cf.data.num_classes)
    w, b = w_full[:n_classes], b_full[:n_classes]
    proj_mean = eigvecs[:, -(n_classes - 1):]
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)

    logger.info("Forward pass: validation split (ID reference)")
    eval_val = utils.compute_model_evaluations(model, datamodule, "val")
    feats_val = to_f64(eval_val["encoded"])
    y_val = to_f64(eval_val["labels"]).astype(np.int64)

    # r8: out-of-sample ID reference. All val-side REFIT statistics use the
    # second half of val; all ID reference blocks come from the first half,
    # so no reference sample is in-sample for any refit. The r7 round showed
    # the in-sample bias hits refit-based scores asymmetrically (the D-dim
    # raw refit overfits the shared reference more than the k-dim projected
    # refit, biasing base-minus-variant negative for Maha).
    split_at = len(feats_val) // 2
    ref_feats, ref_y = feats_val[:split_at], y_val[:split_at]
    fit_feats, fit_y = feats_val[split_at:], y_val[split_at:]

    class_means_val = np.stack([
        fit_feats[fit_y == c].mean(0) if (fit_y == c).any()
        else art["mean_correct"] for c in range(n_classes)])
    centered_cls = np.concatenate([
        fit_feats[fit_y == c] - class_means_val[c]
        for c in range(n_classes) if (fit_y == c).any()])
    cov_within = centered_cls.T @ centered_cls / max(len(centered_cls), 1)
    precision_val = np.linalg.pinv(cov_within, hermitian=True)
    proj_fit = art["mean_correct"] \
        + ((fit_feats - art["mean_correct"]) @ projector) @ projector.T
    class_means_proj = np.stack([
        proj_fit[fit_y == c].mean(0) if (fit_y == c).any()
        else art["mean_correct"] for c in range(n_classes)])
    centered_proj = np.concatenate([
        proj_fit[fit_y == c] - class_means_proj[c]
        for c in range(n_classes) if (fit_y == c).any()])
    cov_proj = centered_proj.T @ centered_proj / max(len(centered_proj), 1)
    precision_proj = np.linalg.pinv(cov_proj, hermitian=True)
    id_ref = ref_feats[:ID_REF_MAX]

    pf = load_deployed_pf(cf)
    if pf["found_global"]:
        bp_global = make_backprojector(pf["global_mean"],
                                       pf["global_comps"], pf["n_global"])
    else:
        logger.warning("Deployed global PF params missing; r7 trial falls "
                       "back to the stage-1 projector")
        bp_global = make_backprojector(art["mean_correct"], projector.T,
                                       q_used)
    class_bp = pf.get("class_bp") if pf["found_class"] else None

    logits_ref = ref_feats @ w.T + b
    correct_ref = logits_ref.argmax(1) == ref_y
    n_id_blocks = max(1, min(N_DRAWS, len(ref_feats) // BATCH_PER_DRAW))
    id_blocks = [ref_feats[d * BATCH_PER_DRAW:(d + 1) * BATCH_PER_DRAW]
                 for d in range(n_id_blocks)]
    id_fail_blocks = [~correct_ref[d * BATCH_PER_DRAW:
                                   (d + 1) * BATCH_PER_DRAW]
                      for d in range(n_id_blocks)]

    logger.info("r7/r8 precompute: per-arm Mahalanobis refits on the val "
                "fit half")
    z_g_fit = bp_global(fit_feats)
    maha_sets = {"raw": refit_maha(fit_feats, fit_y, n_classes,
                                   art["mean_correct"]),
                 "global": refit_maha(z_g_fit, fit_y, n_classes,
                                      art["mean_correct"])}
    if class_bp is not None:
        preds_fit = (fit_feats @ w.T + b).argmax(1)
        z_cp_fit = np.empty_like(fit_feats)
        for c, (mean_c, comps_c, n_c) in enumerate(class_bp):
            mask = preds_fit == c
            if mask.any():
                z_cp_fit[mask] = make_backprojector(mean_c, comps_c,
                                                    n_c)(fit_feats[mask])
        maha_sets["cp"] = refit_maha(z_cp_fit, fit_y, n_classes,
                                     art["mean_correct"])

    if study_name == "vit":
        resize_img = (384, 384)
    elif str(cf.data.dataset) == "tiny-imagenet-200":
        resize_img = (64, 64)
    else:
        resize_img = (32, 32)

    ood_sets: list[tuple[str, object]] = []
    new_class = list(cf.eval.query_studies.new_class_study or [])
    test_loaders = datamodule.test_dataloader()
    for offset, raw_name in enumerate(new_class[:3]):
        idx = 3 + offset
        if idx < len(test_loaders):
            label = NAME_NORMALIZATION.get(raw_name, raw_name)
            ood_sets.append((label, test_loaders[idx]))
    for idx, (folder, label) in FAR_OOD.items():
        try:
            ood_sets.append((label, far_ood_loader(datamodule, folder,
                                                   resize_img)))
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(f"far-OOD loader {label} unavailable: {exc}")

    record = {
        "model_path": args.model_path, "study": study_name,
        "dataset": str(cf.data.dataset), "n_classes": n_classes, "dim": dim,
        "q90": q90, "q_used": q_used, "q_capped": bool(q90 > k_save),
        "coverage_at_q": coverage_at_q, "k_save": int(k_save),
        "n_val": int(len(feats_val)), "id_ref_n": int(len(id_ref)),
        "constants": {"batch_per_draw": BATCH_PER_DRAW, "n_draws": N_DRAWS,
                      "q_coverage": Q_COVERAGE,
                      "augrc_prevalence": "1:1 per-draw val blocks",
                      "r8_split": {"n_ref": int(split_at),
                                   "n_fit": int(len(feats_val) - split_at)}},
        "pf_params": {"found_global": pf["found_global"],
                      "found_class": pf["found_class"],
                      "n_global": pf.get("n_global"),
                      "variance_explained": pf.get("variance_explained"),
                      "class_ranks": [n for _, _, n in class_bp]
                      if class_bp else None},
        "datasets": {},
    }

    for label, loader in ood_sets:
        t1 = time.time()
        feats = collect_features(model, loader, BATCH_PER_DRAW * N_DRAWS)
        if len(feats) < BATCH_PER_DRAW // 2:
            record["datasets"][label] = {"skipped": f"only {len(feats)} "
                                         "samples collected"}
            continue
        n_draws = max(1, len(feats) // BATCH_PER_DRAW)
        draws = []
        for d in range(n_draws):
            block = feats[d * BATCH_PER_DRAW:(d + 1) * BATCH_PER_DRAW]
            delta = block.mean(0) - feats_val.mean(0)
            ori = estimate_orientation(feats_val, block, projector)
            tb = tier_b({"dim": dim}, ori, w, q_used, delta=delta,
                        projector=projector)
            ori_mean = estimate_orientation(feats_val, block, proj_mean)
            tb_mean = tier_b({"dim": dim}, ori_mean, w, n_classes - 1,
                             delta=delta, projector=proj_mean)
            trial = batch_trial(id_ref, block, art["mean_correct"],
                                projector, w, b, class_means_val,
                                precision=precision_val,
                                projected_class_means=class_means_proj,
                                projected_precision=precision_proj)
            trial7 = deployed_trial(id_blocks[d % n_id_blocks],
                                    id_fail_blocks[d % n_id_blocks],
                                    block, w, b, bp_global, class_bp,
                                    maha_sets)
            draws.append({"orientation": ori, "tier_b": tb,
                          "tier_b_meanspan": {k: tb_mean[k] for k in
                                              ("kept", "complement",
                                               "undetermined", "a_hat")},
                          "trial": trial, "trial_deployed": trial7})

        def majority(key: str) -> int:
            votes = [d["tier_b"][key] for d in draws
                     if not d["tier_b"]["undetermined"]]
            if not votes:
                return 0
            total = int(np.sign(sum(votes)))
            return total

        summary = {
            "n_draws": n_draws,
            "a_hat_mean": float(np.mean([d["orientation"]["a_hat"]
                                         for d in draws])),
            "a_hat_sd": float(np.std([d["orientation"]["a_hat"]
                                      for d in draws])),
            "lam_hat_mean": float(np.mean([d["orientation"]["lam_hat"]
                                           for d in draws])),
            "sign_kept": majority("kept"),
            "sign_complement": majority("complement"),
            "sign_logit": majority("logit"),
            "n_undetermined": sum(d["tier_b"]["undetermined"]
                                  for d in draws),
            "trial_mean": {k: float(np.mean([d["trial"][k] for d in draws]))
                           for k in draws[0]["trial"]},
            "trial_deployed_mean": {
                k: float(np.mean([d["trial_deployed"][k] for d in draws]))
                for k in draws[0]["trial_deployed"]},
            "runtime_sec": round(time.time() - t1, 1),
        }
        record["datasets"][label] = {"summary": summary, "draws": draws}
        logger.info(f"{label}: a_hat={summary['a_hat_mean']:.3f} "
                    f"kept={summary['sign_kept']:+d} "
                    f"comp={summary['sign_complement']:+d} "
                    f"logit={summary['sign_logit']:+d}")

    record["runtime_total_sec"] = round(time.time() - t0, 1)
    with open(ori_dir / f"{slug}.json", "w") as fh:
        json.dump(jsonable(record), fh, indent=1)
    logger.info(f"Wrote {ori_dir / slug}.json "
                f"({record['runtime_total_sec']}s)")


if __name__ == "__main__":
    main()
