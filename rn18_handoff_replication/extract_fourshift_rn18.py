"""RN18 handoff-replication plan v3, PHASE 6 extractor (HPC, GPU): the
four new shifts through the ResNet-18 panel of record, plus the
deterministic-view VGG bridge re-extraction (--panel vgg_bridge).

FROZEN CONVENTIONS:
- Train forward = the deterministic PRIMARY view (train examples, test
  transform, canonical order), so the feature model, prototypes, and
  papyan panel equal phase 1's (asserted downstream).
- ID test = the config's iid loader (frozen pool plan); four new sets =
  the frozen roster-B loaders and transform recipe (extract_roster_b_
  newshifts.new_set_loader).
- Scores: frozen mirrors (Energy/CTM claim-bearing); outcomes via the
  frozen set_outcomes (raw + balanced AUGRC, ID-vs-OOD AUROC per score,
  rng 20260827); coords via estimate_ood_coords; P10 compact block via
  the frozen repair_stats rules; G0/G1 inputs (OOD mean, full covariance,
  component means/weights, shared residual covariance) in the npz.
- ID held-out label counts recorded (class probabilities for G0/G1).
Per the first-reader rule the outputs stay UNREAD until
rn18_analysis.py runs. Resumable; sharded; FAILED_ isolation.

Usage (HPC, inside the container, from code/):
    python rn18_handoff_replication/extract_fourshift_rn18.py --list
    python rn18_handoff_replication/extract_fourshift_rn18.py [--shard k/n]
    python rn18_handoff_replication/extract_fourshift_rn18.py --panel vgg_bridge
Output: rn18_handoff_replication/outputs/fourshift_<panel>/<slug>.json + .npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.extract_pool_coords import SRC_KEY, build_ood_plan, forward_loader
from pilot0.extract_roster_b_newshifts import NEW_SETS, new_set_loader
from pilot0.extract_stage2_expansion import set_outcomes
from pilot0.geometry import fit_feature_model, geometry_record, papyan_metrics
from pilot0.ood_coords import estimate_ood_coords
from pilot0.repair_stats import N_MIN, assign_components, compact_p10
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

MANIFESTS = {"rn18": "rn18_handoff_replication/manifests/expected_panel.json",
             "vgg_bridge": "rn18_handoff_replication/manifests/vgg_bridge_panel.json"}
PREFIXES = ("", "fd-shifts/")
SCHEMA = 1


def gaussian_inputs(h_o: np.ndarray, fm) -> tuple[dict, dict]:
    """G0/G1 inputs: uncentered OOD mean + full covariance; frozen
    nearest-prototype partition with the N_MIN merge; component means
    (uncentered), weights, shared residual covariance."""
    H = h_o.astype(np.float64)
    n = len(H)
    mean = H.mean(0)
    Hc = H - mean
    cov_glob = Hc.T @ Hc / n
    hc = H - fm.global_mean
    mu_hat = fm.class_means / fm.radii[:, None]
    labels = assign_components(hc, mu_hat)
    raw = np.bincount(labels, minlength=len(mu_hat))
    keep = np.where(raw >= N_MIN)[0]
    comp = np.where(np.isin(labels, keep), labels, -1)
    ids = sorted(set(comp.tolist()))
    means = np.stack([H[comp == k].mean(0) for k in ids])
    counts = np.array([(comp == k).sum() for k in ids], dtype=float)
    R = H - means[[ids.index(k) for k in comp]]
    cov_res = R.T @ R / n
    scal = {"n": int(n), "component_ids": ids, "weights": (counts / n).tolist()}
    arr = {"ood_mean": mean, "cov_glob": cov_glob.astype(np.float32),
           "comp_means": means, "cov_res": cov_res.astype(np.float32)}
    return scal, arr


def extract_one(cell: dict, root: Path, out_dir: Path, use_cuda: bool) -> None:
    import torch
    from torch.utils.data import DataLoader
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.loaders.dataset_collection import get_dataset
    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = next((f"{p}{cell['model_path']}" for p in PREFIXES
                       if (root / p / cell["model_path"]).is_dir()), None)
    assert model_path, f"not on disk: {cell['model_path']}"
    slug = cell["model_path"].replace("/", "__")
    source = SRC_KEY[cell["model_path"].split("_paper_sweep/")[0]]
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False, rank_feat=False,
                          ash_method=None, use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]
    iid_token, plan, _ = build_ood_plan(cf, source)
    test_loaders = datamodule.test_dataloader()

    ds_det = get_dataset(name=datamodule.dataset_name, root=datamodule.data_dir, train=True,
                         download=True, target_transform=datamodule.target_transforms.get("train"),
                         transform=datamodule.augmentations["test"], kwargs=datamodule.dataset_kwargs)
    logger.info(f"{slug}: forward train (deterministic view)")
    ev = forward_loader(model, DataLoader(ds_det, batch_size=datamodule.batch_size, shuffle=False,
                                          num_workers=datamodule.num_workers,
                                          pin_memory=datamodule.pin_memory))
    h_tr = ev["encoded"].cpu().numpy().astype(np.float32)
    y_tr = ev["labels"].cpu().numpy().astype(np.int64)
    fm = fit_feature_model(h_tr, y_tr, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_tr.astype(np.float64), y_tr, n_classes)
    train_mean = fm.global_mean
    del h_tr, y_tr, ev

    def scores_for(h):
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"], "MLS": hs_["MLS"],
                "CTM": ctm(h64, proto_unc), "Maha": maha(h64),
                "fDBD": fdbd(h64, g, w_np, train_mean), "_logits": g}

    rec = {"schema_fourshift": SCHEMA, **cell, "resolved_model_path": model_path, "slug": slug,
           "source": source, "study": study_name, "n_classes": n_classes,
           "dim": int(fm.global_mean.shape[0]), "view": "deterministic",
           "geometry": geometry_record(w_np, b_np, fm), "papyan": papyan_metrics(w_np, fm), "ood": {}}
    arrays = {"w": w_np, "b": b_np, "proto_unc": proto_unc, "global_mean": fm.global_mean,
              "class_means_centered": fm.class_means, "sigma_w": fm.sigma_w.astype(np.float32)}

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test ({iid_token})")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_id = ev["encoded"].cpu().numpy().astype(np.float32)
    y_id = ev["labels"].cpu().numpy().astype(np.int64)
    sc_id = scores_for(h_id)
    res_id = (sc_id.pop("_logits").argmax(1) != y_id).astype(float)
    rec["iid_test"] = dict(estimate_ood_coords(h_id, fm), n=int(len(h_id)),
                           id_error_rate=float(res_id.mean()),
                           label_counts=np.bincount(y_id, minlength=n_classes).tolist())
    del ev, h_id

    resize_img = (64, 64) if str(cf.data.dataset) == "tiny-imagenet-200" else (32, 32)
    for si, cname in enumerate(NEW_SETS, start=1):
        try:
            logger.info(f"{slug}: forward {cname}")
            ev = forward_loader(model, new_set_loader(cname, datamodule, resize_img))
            h_o = ev["encoded"].cpu().numpy().astype(np.float32)
            sc_o = scores_for(h_o); sc_o.pop("_logits")
            gscal, garr = gaussian_inputs(h_o, fm)
            rec["ood"][cname] = dict(estimate_ood_coords(h_o, fm),
                                     p10=compact_p10(h_o, fm, w_np, set_index=si),
                                     gaussian=gscal, **set_outcomes(sc_id, res_id, sc_o))
            for k, v in garr.items():
                arrays[f"set__{cname}__{k}"] = v
            del ev, h_o
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            rec["ood"][cname] = {"error": str(err)}
    rec["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    f = out_dir / f"FAILED_{slug}.json"
    if f.exists():
        f.unlink()
    logger.info(f"{slug}: wrote {len(rec['ood'])} sets ({rec['runtime_sec']}s)")
    del model, module, datamodule
    if use_cuda:
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", choices=list(MANIFESTS), default="rn18")
    ap.add_argument("--shard", type=str, default="1/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    man = json.loads(Path(MANIFESTS[args.panel]).read_text())
    cells = [c for c in man["cells"] if c.get("mechanically_eligible", True)]
    k, n = (int(x) for x in args.shard.split("/"))
    cells = cells[k - 1::n]
    out_dir = Path(f"rn18_handoff_replication/outputs/fourshift_{args.panel}")
    todo = [c for c in cells if not (out_dir / f"{c['model_path'].replace('/', '__')}.json").exists()]
    print(f"[fourshift/{args.panel}] shard {args.shard}: {len(cells)} eligible cells, {len(todo)} to run", flush=True)
    if args.list:
        for c in todo[:12]:
            print("  ", c["model_path"])
        return
    failures = 0
    for i, c in enumerate(todo, 1):
        print(f"[fourshift] {i}/{len(todo)}: {c['model_path']}", flush=True)
        try:
            extract_one(c, root, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{c['model_path'].replace('/', '__')}.json").write_text(
                json.dumps({"model_path": c["model_path"], "error": traceback.format_exc()}, indent=1))
            print(f"[fourshift] FAILED {c['model_path']}", flush=True)
    print(f"[fourshift] done: {len(todo) - failures} ok, {failures} failed", flush=True)


if __name__ == "__main__":
    main()
