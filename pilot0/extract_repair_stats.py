"""Phase-1 repair-statistics extraction (saturation plan 2026-08-28,
section 12 Phase 1; HPC side).

Consumes the FROZEN subset manifest
(nc_csf_predictivity/outputs/track1/repair_phase1_manifest.json, built by
repair_phase1_manifest.py from the development half of the repair split
plus the five deterministic BREEDS checkpoints; sha256 committed). All
component, covariance, and shrinkage rules are frozen in
pilot0/repair_stats.py. NO detector score, outcome, gap, or ranking is
computed or consulted anywhere in this pipeline.

Per checkpoint: forwards the train split, the ID test, and the plan's
OOD suite with the SAME loader machinery as the frozen extractions
(pool: extract_pool_coords schema-2 plan incl. custom and torchvision
fallbacks; breeds: extract_stage1_pilot.flatten_externals). Writes ONE
JSON (meta, measured geometry, frozen continuity coordinates from
estimate_ood_coords, per-set scalar diagnostics) and ONE compressed NPZ
(the section-8.4 sufficient-statistic arrays: ID class/logit/prototype
projections and Sigma_W projections; per set the global-mean and
per-component mean projections, the global and pooled-residual
covariance projections, and top-16 eigenvalues). The ID test receives
the same mixture decomposition as the OOD sets (reference structure).
set_index for the batch-occupancy seed = position in the recorded
"set_order" list (ID test = 0).

Failures are isolated per checkpoint (FAILED_<slug>.json; sweep
continues); the sweep is resumable (existing <slug>.json skipped).

Usage (from code/, inside the campaign container, .env with
EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR):
    python pilot0/extract_repair_stats.py --list
    python pilot0/extract_repair_stats.py [--only <substr>] [--no-use_cuda]
Outputs: pilot0/repair_phase1_stats/<slug>.json + <slug>.npz  (rsync back)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.extract_pool_coords import (SRC_KEY, build_ood_plan,
                                        fallback_loader, forward_loader)
from pilot0.extract_stage1_pilot import flatten_externals
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.repair_stats import (assign_components, id_stats, mixture_stats,
                                 split_record)

MANIFEST = ("nc_csf_predictivity/outputs/track1/"
            "repair_phase1_manifest.json")
OUT_DIR_DEFAULT = "pilot0/repair_phase1_stats"
CHUNK = 8192


def logits_argmax(h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(h), dtype=np.int64)
    for i in range(0, len(h), CHUNK):
        out[i:i + CHUNK] = (h[i:i + CHUNK].astype(np.float64) @ w.T
                            + b).argmax(1)
    return out


def process_set(record, arrays, key, set_index, h, fm, w_np, b_np, mu_hat):
    """Continuity coordinates + mixture sufficient statistics for one set."""
    coords = estimate_ood_coords(h, fm)
    hc = h.astype(np.float64) - fm.global_mean
    mx = mixture_stats(hc, w_np, mu_hat,
                       logits_argmax=logits_argmax(h, w_np, b_np),
                       set_index=set_index, chunk=CHUNK)
    scalars, arrs = split_record(mx)
    record["sets"][key] = {"set_index": set_index, "n": int(len(h)),
                           "coords": coords, "stats": scalars}
    for name, a in arrs.items():
        arrays[f"set__{key.replace(' ', '_')}__{name}"] = a


def extract_one(target: dict, out_dir: Path, use_cuda: bool) -> None:
    import torch  # noqa: F401 - torch env before fd_shifts imports
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = target["model_path"]
    kind = target["kind"]
    slug = model_path.replace("/", "__")
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]
    test_loaders = datamodule.test_dataloader()

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_train = ev["encoded"].cpu().numpy().astype(np.float32)
    y_train = ev["labels"].cpu().numpy().astype(np.int64)
    fm = fit_feature_model(h_train, y_train, n_classes)
    mu_hat = fm.class_means / fm.radii[:, None]

    record: dict = {
        "phase": "repair_phase1", "model_path": model_path, "slug": slug,
        "kind": kind, "study": study_name, "n_classes": n_classes,
        "dim": int(h_train.shape[1]), "n_train": int(len(h_train)),
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm),
        "sets": {}, "set_order": [],
    }
    id_scalars, arrays = split_record(
        {"id": id_stats(fm, w_np, b_np)})
    record["id_scalars"] = id_scalars
    train_labels = assign_components(
        h_train.astype(np.float64) - fm.global_mean, mu_hat, chunk=CHUNK)
    record["id_prototype_switch_rate"] = float(
        (train_labels != y_train).mean())
    del train_labels, h_train, y_train

    # ---- build the ordered set plan -----------------------------------
    if kind == "pool":
        source = SRC_KEY[model_path.split("_paper_sweep/")[0]]
        iid_token, plan, notes = build_ood_plan(cf, source)
        record["plan_notes"] = notes
        resize_img = ((64, 64) if str(cf.data.dataset) == "tiny-imagenet-200"
                      else (32, 32))
    else:  # breeds
        iid_token, specs = flatten_externals(cf)
        plan = {s["raw"]: {"kind": "config", "token": s["token"]}
                for s in specs if s["forward"]}
        resize_img = None

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test ({iid_token})")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_iid = ev["encoded"].cpu().numpy().astype(np.float32)
    record["set_order"].append("iid_test")
    process_set(record, arrays, "iid_test", 0, h_iid, fm, w_np, b_np,
                mu_hat)
    del ev, h_iid

    for set_index, (cname, spec) in enumerate(sorted(plan.items()), 1):
        record["set_order"].append(cname)
        try:
            if spec["kind"] == "config":
                idx = int(spec["token"].split("_")[1])
                logger.info(f"{slug}: forward {cname} ({spec['token']})")
                ev = forward_loader(model, test_loaders[idx])
            elif spec["kind"] == "custom":
                logger.info(f"{slug}: forward {cname} (custom)")
                ev = utils.compute_model_evaluations(model, datamodule,
                                                     spec["token"])
            else:
                logger.info(f"{slug}: forward {cname} (fallback)")
                ev = forward_loader(model,
                                    fallback_loader(cname, datamodule,
                                                    resize_img))
            h_ood = ev["encoded"].cpu().numpy().astype(np.float32)
            process_set(record, arrays, cname, set_index, h_ood, fm,
                        w_np, b_np, mu_hat)
            del ev, h_ood
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            record["sets"][cname] = {"error": str(err)}

    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(record, indent=1, default=float))
    np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    logger.info(f"{slug}: wrote stats ({len(record['sets'])} sets, "
                f"{record['runtime_sec']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=str, default=MANIFEST)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    targets = ([dict(e, kind="pool") for e in man["pool"]]
               + [dict(e, kind="breeds") for e in man["breeds"]
                  if "error" not in e])
    if args.only:
        targets = [t for t in targets if args.only in t["model_path"]]
    out_dir = Path(args.out_dir)
    todo = [t for t in targets
            if not (out_dir
                    / f"{t['model_path'].replace('/', '__')}.json").exists()]
    print(f"[repair-p1] {len(targets)} targets, {len(todo)} to run",
          flush=True)
    if args.list:
        for t in todo:
            print(f"  {t['kind']:6s} {t['model_path']}")
        return
    failures = 0
    for i, t in enumerate(todo, 1):
        print(f"[repair-p1] {i}/{len(todo)}: {t['model_path']}", flush=True)
        try:
            extract_one(t, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            slug = t["model_path"].replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"model_path": t["model_path"],
                 "error": traceback.format_exc()}, indent=1))
            print(f"[repair-p1] FAILED {slug} (recorded)", flush=True)
    print(f"[repair-p1] done: {len(todo) - failures} ok, "
          f"{failures} failed", flush=True)


if __name__ == "__main__":
    main()
