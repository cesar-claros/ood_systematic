"""Pilot 1 deep-gradient GradPCA stage for one checkpoint (X6-GradPCA campaign).

Loads an fd-shifts experiment exactly as csf_fit.py does (via
x6_spectral/measure_checkpoint.load_model), then in two passes:

  fit pass (train split): one batched forward + two short backwards per
      batch (sum and max-logit aggregations) yield exact per-sample
      gradients of the pre-declared deep layer via the hook trick
      (per_sample_grads.py); class-mean gradients are accumulated per true
      class, and the GradPCA construction (unweighted-mean centering, dual
      C x C Gram in float64, trace cut, lift) is fit for both aggregations.
      Penultimate activations are collected in the same pass and the three
      HEAD variants (head_sum, head_max, act_cmeans) are fit on them via
      src/csfs/gradpca.py, giving perfectly matched baselines from the same
      forwards.
  score pass (iid + OOD sets): same forward/backward machinery scores the
      two deep variants; head variants score from the collected activations.

Pre-declared deep parameter subsets (X6_gradpca_pilot1_protocol.md):
  VGG-13 studies:  the last Conv2d in encoder.features (weight + bias) —
                   the same layer RankWeight targets;
  ViT study:       model.blocks[-1].mlp.fc2 (weight + bias).

Metrics replicate csf_pipeline.stats(): failure AUGRC/AURC (RiskCoverageStats),
failure AUROC / FPR@95 / AP (fd_shifts metrics.StatsCache), and OOD modes use
the joint convention of compute_metrics (correct-only iid_test rows + full
OOD set with correct := 0). Outputs go to <out_dir>/<slug>.json (config,
resolved parameter names/shapes, k*, runtimes, peak memory, per-mode metric
rows for all five variants + matched deep-minus-head deltas) and
<slug>_scores.npz (per-set per-variant scores for later aggregation).

Runtime self-check: on the first batch, the hook-trick gradients are compared
against a per-sample autograd loop on the real architecture; the stage aborts
on disagreement.

Usage (from code/, inside the paper container on the HPC):
    python x6_gradpca/deep_gradpca.py --model_path=<experiment> [--use_cuda]
        [--modes=iid_test,ood_nsncs_svhn,...] [--chunk=N] [--smoke=512]
        [--list_params] [--out_dir=x6_gradpca/outputs]
Environment: EXPERIMENT_ROOT_DIR, DATASET_ROOT_DIR (code/.env).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "x6_spectral"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fd_shifts import logger
from fd_shifts.analysis import metrics
from fd_shifts.loaders.data_loader import FDShiftsDataLoader

from measure_checkpoint import load_model  # x6_spectral loader, csf_fit-faithful
from per_sample_grads import LayerGradCapture, aggregation_scalar, reference_per_sample_grads
from src.csfs.gradpca import GradPCA
from src.rc_stats import RiskCoverageStats
from src.trained_module import TrainedModule

DEEP_VARIANTS = ("GradPCA_lastlayer_sum", "GradPCA_lastlayer_max")
HEAD_VARIANTS = {"GradPCA_head_sum": "head_sum", "GradPCA_head_max": "head_max",
                 "ActPCA_cmeans": "act_cmeans"}

#: eval mode -> compute_model_evaluations set name (mirrors csf_eval.py).
MODE_TO_SET = {
    "iid_val": "val", "iid_test": "test_1",
    "ood_sncs_c10": "test_3", "ood_sncs_c100": "test_3",
    "ood_nsncs_svhn": "test_4", "ood_nsncs_ti": "test_5",
    "ood_nsncs_lsun_cropped": "test_6", "ood_nsncs_lsun_resize": "test_7",
    "ood_nsncs_isun": "test_8", "ood_nsncs_textures": "test_9",
    "ood_nsncs_places365": "test_10",
}
DEFAULT_MODES = ("iid_val,iid_test,ood_sncs_c10,ood_nsncs_svhn,ood_nsncs_ti,"
                 "ood_nsncs_lsun_cropped,ood_nsncs_lsun_resize,ood_nsncs_isun,"
                 "ood_nsncs_textures,ood_nsncs_places365")


def parse_args():
    p = argparse.ArgumentParser(description="X6-GradPCA Pilot 1 deep-gradient stage")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_dir", type=str, default="x6_gradpca/outputs")
    p.add_argument("--trace_threshold", type=float, default=0.99)
    p.add_argument("--modes", type=str, default=DEFAULT_MODES)
    p.add_argument("--chunk", type=int, default=None,
                   help="forward batch size (default: 128 CNN / 16 ViT)")
    p.add_argument("--smoke", type=int, default=None,
                   help="cap samples per split (smoke test only)")
    p.add_argument("--selfcheck_n", type=int, default=8,
                   help="samples for the hook-vs-autograd runtime check (0 = skip)")
    p.add_argument("--list_params", action="store_true",
                   help="print the resolved target layer and exit")
    return p.parse_args()


def resolve_target_layer(T: TrainedModule, study_name: str):
    if study_name == "vit":
        layer = T.module.model.blocks[-1].mlp.fc2
        name = f"model.blocks.{len(T.module.model.blocks) - 1}.mlp.fc2"
    else:
        idx = int(T.conv_layers_name[-1])
        layer = T.model.encoder.features[idx]
        name = f"encoder.features.{idx}"
    return layer, name


def make_forward(T: TrainedModule, study_name: str, ext_confid_name: str):
    """images -> (encoded z, logits), grad-enabled (mirrors TrainedModule.__call__
    minus the confidence branches and without detaching)."""
    def forward(x):
        z = T.forward_features_vit(x) if study_name == "vit" else T.forward_features(x)
        if ext_confid_name == "devries":
            logits, _ = T.model.head(z)
        elif ext_confid_name == "dg":
            logits = T.model.head(z)  # aggregation slices off the reservation logit
        elif ext_confid_name == "tcp":
            logits = T.model.head(z)
        elif ext_confid_name == "maha":
            logits = T.module.model.head(z)
        else:
            raise NotImplementedError(ext_confid_name)
        return z, logits
    return forward


def get_dataloader(datamodule, set_name: str, study_name: str, dataset_name: str):
    """Mirror utils.compute_model_evaluations' loader dispatch (shuffle=False
    everywhere; ImageFolder OOD sets 6-10 with the ImageNet normalization)."""
    if study_name == "vit":
        resize_img = (384, 384)
    elif dataset_name == "tiny-imagenet-200":
        resize_img = (64, 64)
    else:
        resize_img = (32, 32)
    parts = set_name.split("_")
    if parts[0] == "train":
        return datamodule.train_dataloader()
    if parts[0] == "val":
        return datamodule.val_dataloader()
    assert parts[0] == "test"
    n = int(parts[1])
    if n <= 5:
        return datamodule.test_dataloader()[n]
    folder = {6: ("LSUN",), 7: ("LSUN_resize",), 8: ("iSUN",),
              9: ("dtd", "images"), 10: ("places365",)}[n]
    transform = transforms.Compose([
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(
        datamodule.data_root_dir.joinpath(*folder), transform=transform)
    return DataLoader(dataset, batch_size=datamodule.batch_size,
                      num_workers=datamodule.num_workers, shuffle=False)


def spectral_fit_dual(cmeans: torch.Tensor, eps: float):
    """GradPCA dual construction on a (C, P) float64 class-mean matrix."""
    mean = cmeans.mean(dim=0)
    M = cmeans - mean
    G = (M @ M.T)
    evals, evecs = torch.linalg.eigh(G)
    evals, evecs = evals.flip(0).clamp(min=0.0), evecs.flip(1)
    cum = torch.cumsum(evals, 0) / evals.sum()
    k = int((cum >= eps).nonzero()[0].item()) + 1
    k = min(k, int((evals > evals[0] * 1e-12).sum().item()))
    U = M.T @ evecs[:, :k]
    U = U / U.norm(dim=0, keepdim=True)
    return mean, U, k


def stats_row(scores: np.ndarray, correct: np.ndarray, n_bins: int = 20) -> dict:
    """Replicates csf_pipeline.stats() metric computation for one score column."""
    confids = torch.from_numpy(scores.astype(np.float64))
    correct_t = torch.from_numpy(correct.astype(np.int64))
    rcs = RiskCoverageStats(confids=confids, residuals=1 - correct_t)
    cache = metrics.StatsCache(confids, correct_t, n_bins)
    return {
        "AUGRC": float(rcs.augrc), "AURC": float(rcs.aurc),
        "AUROC_f": float(metrics.failauc(cache)),
        "FPR@95TPR": float(metrics.fpr_at_95_tpr(cache)),
        "AP_ferr": float(metrics.failap_err(cache)),
        "AP_fsuc": float(metrics.failap_suc(cache)),
    }


def main():
    args = parse_args()
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_path.replace("/", "__")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODE_TO_SET]
    assert not unknown, f"unknown modes {unknown}"
    ood_modes = [m for m in modes if m.startswith("ood_")]
    if ood_modes and "iid_test" not in modes:
        modes.insert(0, "iid_test")  # joint OOD eval needs the iid_test scores
    modes.sort(key=lambda m: 0 if not m.startswith("ood_") else 1)

    cf, module, study_name = load_model(args.model_path, use_cuda)
    n_classes = int(cf.data.num_classes)
    chunk = args.chunk or (16 if study_name == "vit" else 128)
    cf.trainer.batch_size = chunk
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    T = TrainedModule(module, study_name, cf, rank_weight=False, rank_feat=False,
                      ash_method=None, use_cuda=use_cuda)
    forward = make_forward(T, study_name, str(cf.eval.ext_confid_name))
    layer, layer_name = resolve_target_layer(T, study_name)
    if args.list_params:
        print(f"target layer: {layer_name} ({type(layer).__name__})")
        for n, p in layer.named_parameters():
            print(f"  {n}: {tuple(p.shape)}")
        return

    # Freeze everything except the target layer: short backward, small graph.
    for p in T.module.parameters():
        p.requires_grad_(False)
    if study_name == "confidnet":
        for p in T.network.parameters():
            p.requires_grad_(False)
    for p in layer.parameters():
        p.requires_grad_(True)
    cap = LayerGradCapture(layer)
    P = cap.flat_dim
    logger.info(f"Deep subset: {layer_name}, P = {P}, chunk = {chunk}")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    record = {
        "model_path": args.model_path, "study": study_name,
        "dataset": str(cf.data.dataset), "n_classes": n_classes,
        "target_layer": layer_name, "target_layer_type": type(layer).__name__,
        "param_names_shapes": cap.param_names_shapes(prefix=f"{layer_name}."),
        "flat_dim": int(P), "chunk": chunk,
        "trace_threshold": args.trace_threshold,
        "flatten_convention": "[weight.reshape(-1), bias] per layer",
        "aggregations": {"sum": "sum of first C logits (v=1; DG reservation excluded)",
                         "max": "maximum over first C logits"},
        "smoke": args.smoke, "runtime_sec": {}, "selfcheck": None,
        "modes": {}, "deltas": {},
    }

    def run_split(set_name, fit=False, selfcheck=False):
        """One pass over a split; returns dict of collected arrays (and, when
        fit=True, the per-class gradient sums)."""
        loader = get_dataloader(datamodule, set_name, study_name, str(cf.data.dataset))
        sums = {a: torch.zeros(n_classes, P, dtype=torch.float64, device=device)
                for a in ("sum", "max")} if fit else None
        counts = torch.zeros(n_classes, dtype=torch.float64, device=device) if fit else None
        h_list, logits_list, y_list = [], [], []
        deep_scores = {v: [] for v in DEEP_VARIANTS} if not fit else None
        n_seen = 0
        for x, y in loader:
            if args.smoke is not None and n_seen >= args.smoke:
                break
            x, y = x.to(device), y.to(device)
            z, logits = forward(x)
            aggregation_scalar(logits, n_classes, "sum").backward(retain_graph=True)
            g_sum = cap.per_sample_grads()
            aggregation_scalar(logits, n_classes, "max").backward()
            g_max = cap.per_sample_grads()
            for p in layer.parameters():
                p.grad = None
            if selfcheck and n_seen == 0 and args.selfcheck_n > 0:
                nn_ = min(args.selfcheck_n, x.shape[0])
                errs = {}
                for agg, g in (("sum", g_sum), ("max", g_max)):
                    ref = reference_per_sample_grads(
                        lambda xi: forward(xi)[1], layer, x[:nn_], n_classes, agg)
                    denom = ref.abs().max().clamp(min=1e-30)
                    errs[agg] = float((g[:nn_] - ref).abs().max() / denom)
                record["selfcheck"] = errs
                assert max(errs.values()) < 1e-3, f"hook-trick self-check failed: {errs}"
                logger.info(f"Self-check passed: {errs}")
            if fit:
                sums["sum"].index_add_(0, y, g_sum.double())
                sums["max"].index_add_(0, y, g_max.double())
                counts.index_add_(0, y, torch.ones(len(y), dtype=torch.float64, device=device))
            else:
                for name, g in (("GradPCA_lastlayer_sum", g_sum), ("GradPCA_lastlayer_max", g_max)):
                    mean, U, _ = deep_fits[name]
                    zc = g.double() - mean
                    num = (zc @ U).pow(2).sum(dim=1)
                    den = zc.pow(2).sum(dim=1)
                    deep_scores[name].append((num / den).cpu())
            h_list.append(z.detach().float().cpu())
            logits_list.append(logits.detach().float().cpu())
            y_list.append(y.cpu())
            n_seen += x.shape[0]
        out = {"h": torch.cat(h_list), "logits": torch.cat(logits_list),
               "labels": torch.cat(y_list)}
        if fit:
            return out, sums, counts
        out.update({v: torch.cat(deep_scores[v]) for v in DEEP_VARIANTS})
        return out

    # ---- fit pass -------------------------------------------------------
    t0 = time.time()
    train_out, sums, counts = run_split("train", fit=True, selfcheck=True)
    assert (counts > 0).all(), f"empty classes in train split: {(counts == 0).nonzero().flatten().tolist()}"
    deep_fits = {}
    for name, agg in (("GradPCA_lastlayer_sum", "sum"), ("GradPCA_lastlayer_max", "max")):
        cmeans = sums[agg] / counts.unsqueeze(1)
        mean, U, k = spectral_fit_dual(cmeans, args.trace_threshold)
        deep_fits[name] = (mean, U, k)
        record[f"k_{name}"] = int(k)
        logger.info(f"{name}: k* = {k}")
    del sums
    head_fits = {}
    for name, variant in HEAD_VARIANTS.items():
        gp = GradPCA(module, study_name, cf, variant=variant,
                     trace_threshold=args.trace_threshold)
        gp.compute_GradPCA_params(train_out["h"], train_out["labels"])
        head_fits[name] = gp
        record[f"k_{name}"] = int(gp.n_components)
    record["runtime_sec"]["fit"] = round(time.time() - t0, 1)
    del train_out

    # ---- score pass -----------------------------------------------------
    per_set, npz_payload = {}, {}
    for mode in modes:
        set_name = MODE_TO_SET[mode]
        if set_name in per_set:
            continue
        t1 = time.time()
        out = run_split(set_name)
        for name, gp in head_fits.items():
            out[name] = gp.get_scores(out["h"]).float()
        preds = out["logits"][:, :n_classes].argmax(dim=1)
        out["correct"] = (preds == out["labels"]).long()
        del out["h"], out["logits"]
        per_set[set_name] = out
        dt = time.time() - t1
        record["runtime_sec"][set_name] = round(dt, 1)
        record.setdefault("throughput_sps", {})[set_name] = round(len(out["labels"]) / max(dt, 1e-9), 1)

    all_variants = list(DEEP_VARIANTS) + list(HEAD_VARIANTS)
    for mode in modes:
        s = per_set[MODE_TO_SET[mode]]
        rows = {}
        for v in all_variants:
            if mode.startswith("ood_"):
                iid = per_set["test_1"]
                keep = iid["correct"] == 1  # compute_metrics joint convention
                scores = torch.cat([iid[v][keep], s[v]]).numpy()
                correct = np.concatenate([np.ones(int(keep.sum())), np.zeros(len(s[v]))])
            else:
                scores, correct = s[v].numpy(), s["correct"].numpy()
            rows[v] = stats_row(scores, correct)
        record["modes"][mode] = rows
        record["deltas"][mode] = {
            "lastlayer_sum_minus_head_sum": {
                m: rows["GradPCA_lastlayer_sum"][m] - rows["GradPCA_head_sum"][m]
                for m in ("AUGRC", "AUROC_f", "FPR@95TPR")},
            "lastlayer_max_minus_head_max": {
                m: rows["GradPCA_lastlayer_max"][m] - rows["GradPCA_head_max"][m]
                for m in ("AUGRC", "AUROC_f", "FPR@95TPR")},
        }
        for set_name in (MODE_TO_SET[mode],):
            for v in all_variants:
                npz_payload[f"{mode}__{v}"] = per_set[set_name][v].numpy().astype(np.float32)
            npz_payload[f"{mode}__correct"] = per_set[set_name]["correct"].numpy().astype(np.int8)

    if use_cuda:
        record["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    record["stored_projector_bytes"] = {
        name: int(deep_fits[name][1].numel() * 8 + deep_fits[name][0].numel() * 8)
        for name in DEEP_VARIANTS}
    cap.remove()

    np.savez_compressed(out_dir / f"{slug}_scores.npz", **npz_payload)
    with open(out_dir / f"{slug}.json", "w") as fh:
        json.dump(record, fh, indent=1)
    logger.info(f"Wrote {out_dir / slug}.json and _scores.npz")
    for mode in modes:
        d = record["deltas"][mode]
        logger.info(f"[{mode}] deep-vs-head AUROC_f deltas: "
                    f"sum {d['lastlayer_sum_minus_head_sum']['AUROC_f']:+.4f}, "
                    f"max {d['lastlayer_max_minus_head_max']['AUROC_f']:+.4f}")


if __name__ == "__main__":
    main()
