"""iWildCam Stage-2 extraction (source-expansion protocol amendment 7,
2026-09-01; HPC side). Five Stage-1 checkpoints, ONE configured shift,
and the CLEAN custom ID test.

FROZEN SPECIFICATION (amendment 7; declared before any Stage-2 outcome
is computed):
- Clean ID test: rebuild the iid test subset exactly as the
  datamodule's setup() does (same get_dataset call, same test
  transforms), apply the datamodule's own pre-slices if active
  (test_iid_split == "tenPercent"), then EXCLUDE the first 1000 indices
  (the devries-val training-time selection slice) and keep the rest.
  Expected n = 7,154 with no pre-slice; the constructed n, the excluded
  count, and the pre-slice state are asserted and recorded. Rationale:
  the fd-shifts default truncates the WILDS iid test to indices
  [100:150] (n = 50) inside that same selection slice.
- OOD: the config-declared wilds_animals_ood_test by loader index
  (stage-1 convention), n = 42,791.
- Scores: frozen feature-level mirrors (Energy/CTM claim-bearing;
  MSR/MLS/Maha/fDBD secondary); outcomes via the frozen set_outcomes
  (raw + prevalence-balanced AUGRC, rng 20260827, SIGNED gaps per the
  amendment-5 Stage-2 convention); frozen geometry/coordinates.
Per the voluntarily adopted GR-5 discipline the outputs stay UNREAD
until the committed stage2_iwildcam_analysis.py runs.

Usage (HPC, inside the container, from code/, .env loaded):
    python pilot0/extract_stage2_iwildcam.py --list
    python pilot0/extract_stage2_iwildcam.py
Output: pilot0/stage2_iwildcam_coords/<slug>.json
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

from pilot0.extract_pool_coords import forward_loader
from pilot0.extract_stage1_pilot import flatten_externals, resolve_targets
from pilot0.extract_stage2_expansion import set_outcomes
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

SCHEMA_S2_IWC = 1
OUT_DIR_DEFAULT = "pilot0/stage2_iwildcam_coords"
DEVRIES_VAL_SLICE = 1000


def clean_id_loader(datamodule) -> tuple:
    """The amendment-7 clean ID test loader. Returns (loader, meta)."""
    import torch
    from fd_shifts.loaders.dataset_collection import get_dataset

    ds = get_dataset(
        name=datamodule.dataset_name,
        root=datamodule.data_dir,
        train=False,
        download=True,
        target_transform=datamodule.target_transforms.get("test"),
        transform=datamodule.augmentations["test"],
        kwargs=datamodule.dataset_kwargs,
    )
    assert hasattr(ds, "indices"), (
        "clean ID construction expects a WILDS subset with .indices")
    n_full = len(ds.indices)
    pre_slice = getattr(datamodule, "test_iid_split", None)
    if pre_slice == "tenPercent":
        split = int(n_full * 0.1)
        ds.indices = ds.indices[split:]
    n_pre = len(ds.indices)
    ds.indices = ds.indices[DEVRIES_VAL_SLICE:]
    n_clean = len(ds.indices)
    assert n_clean == n_pre - DEVRIES_VAL_SLICE > 0, (n_pre, n_clean)
    meta = {"n_id_test_full": int(n_full),
            "pre_slice": str(pre_slice),
            "n_after_pre_slice": int(n_pre),
            "n_excluded_devries_val": DEVRIES_VAL_SLICE,
            "n_clean": int(n_clean)}
    loader = torch.utils.data.DataLoader(
        ds, batch_size=datamodule.batch_size, shuffle=False,
        pin_memory=datamodule.pin_memory,
        num_workers=datamodule.num_workers)
    return loader, meta


def extract_one(target: dict, out_dir: Path, use_cuda: bool) -> None:
    import torch  # noqa: F401 - torch env before fd_shifts imports
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = target["model_path"]
    slug = model_path.replace("/", "__")
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None,
                          use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]
    iid_token, specs = flatten_externals(cf)
    test_loaders = datamodule.test_dataloader()

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_tr = ev["encoded"].cpu().numpy().astype(np.float32)
    y_tr = ev["labels"].cpu().numpy().astype(np.int64)
    counts = np.bincount(y_tr, minlength=n_classes)
    assert counts.min() > 0, "empty train class; frozen model undefined"
    fm = fit_feature_model(h_tr, y_tr, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_tr.astype(np.float64), y_tr, n_classes)
    train_mean = fm.global_mean
    record: dict = {
        "schema_stage2_iwildcam": SCHEMA_S2_IWC,
        "model_path": model_path, "slug": slug,
        "source": "iwildcam", "paradigm": target["paradigm"],
        "reward": target["reward"], "n_classes": n_classes,
        "dim": int(h_tr.shape[1]), "n_train": int(len(h_tr)),
        "class_support": {"min": int(counts.min()),
                          "median": float(np.median(counts)),
                          "n_empty": int((counts == 0).sum())},
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm), "ood": {},
    }
    del h_tr, y_tr

    def scores_for(h: np.ndarray) -> dict:
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"],
                "MLS": hs_["MLS"], "CTM": ctm(h64, proto_unc),
                "Maha": maha(h64), "fDBD": fdbd(h64, g, w_np, train_mean),
                "_logits": g}

    logger.info(f"{slug}: forward CLEAN iid test (amendment 7)")
    id_loader, id_meta = clean_id_loader(datamodule)
    ev = forward_loader(model, id_loader)
    h_id = ev["encoded"].cpu().numpy().astype(np.float32)
    y_id = ev["labels"].cpu().numpy().astype(np.int64)
    assert len(h_id) == id_meta["n_clean"], (len(h_id), id_meta)
    sc_id = scores_for(h_id)
    res_id = (sc_id.pop("_logits").argmax(1) != y_id).astype(float)
    record["iid_test"] = dict(estimate_ood_coords(h_id, fm), **id_meta,
                              id_error_rate=float(res_id.mean()))
    del ev, h_id

    for spec in specs:
        if not spec["forward"]:
            continue
        try:
            idx = int(spec["token"].split("_")[1])
            logger.info(f"{slug}: forward {spec['raw']} ({spec['token']})")
            ev = forward_loader(model, test_loaders[idx])
            h_o = ev["encoded"].cpu().numpy().astype(np.float32)
            sc_o = scores_for(h_o)
            sc_o.pop("_logits")
            record["ood"][spec["raw"]] = dict(
                estimate_ood_coords(h_o, fm),
                **set_outcomes(sc_id, res_id, sc_o))
            del ev, h_o
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {spec['raw']} FAILED: {err}")
            record["ood"][spec["raw"]] = {"error": str(err)}
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(record, indent=1, default=float))
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    logger.info(f"{slug}: wrote stage-2 record "
                f"({record['runtime_sec']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()
    targets = resolve_targets("iwildcam")
    missing = [t for t in targets if t["model_path"] is None]
    assert not missing, f"unresolved checkpoints: {missing}"
    out_dir = Path(args.out_dir)
    todo = [t for t in targets if not (
        out_dir / f"{t['model_path'].replace('/', '__')}.json").exists()]
    print(f"[s2-iwc] {len(targets)} targets, {len(todo)} to run")
    for t in todo:
        print(f"   {t['paradigm']}/rew{t['reward']}: {t['model_path']}")
    if args.list:
        return
    failures = 0
    for t in todo:
        try:
            extract_one(t, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            slug = t["model_path"].replace("/", "__")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"model_path": t["model_path"],
                 "error": traceback.format_exc()}, indent=1))
            print(f"[s2-iwc] FAILED {slug}", flush=True)
    print(f"[s2-iwc] done: {len(todo) - failures} ok, {failures} failed",
          flush=True)


if __name__ == "__main__":
    main()
