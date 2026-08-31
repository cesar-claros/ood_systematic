"""ICML roster B extraction: four NEW OOD shifts through the 280 VGG-13
pool checkpoints (HPC side; frozen protocol section 8.2 + KID/FD
amendment). Per GR-5 the outputs stay UNREAD until the committed
analysis suite runs.

FROZEN SPEC:
- New sets (exactly these, none in the registered 8-set suite):
  MNIST (test), FashionMNIST (test), KMNIST (test), STL-10 (test).
- Transform recipe = the frozen custom-branch recipe: Resize to the
  source's input size ((64, 64) for the tiny-imagenet source, else
  (32, 32)), ToTensor, ImageNet normalization; grayscale sets replicate
  to 3 channels ahead of Resize. Downloads via torchvision
  (download=True) under DATASET_ROOT_DIR.
- Checkpoint roster = the 280 pool model paths enumerated from the
  committed pool_coords records (metadata only).
- Scores: the frozen feature-level mirrors (Energy/CTM claim-bearing);
  outcomes via the frozen set_outcomes (raw AUGRC carries the pool
  materiality convention downstream; balanced values recorded too);
  per-set frozen estimate_ood_coords for the predictor's coordinates.
- Transport label: shift-generalization at known checkpoints.
Resumable; sharded; per-checkpoint FAILED_ isolation.

Usage (HPC, inside the container, from code/):
    python pilot0/extract_roster_b_newshifts.py --list
    python pilot0/extract_roster_b_newshifts.py [--shard k/n]
Output: pilot0/icml_roster_b_coords/<slug>.json
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

from pilot0.extract_pool_coords import SRC_KEY, build_ood_plan, forward_loader
from pilot0.extract_stage2_expansion import set_outcomes
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

NEW_SETS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
OUT_DIR_DEFAULT = "pilot0/icml_roster_b_coords"


def pool_model_paths() -> list[str]:
    paths = []
    for p in sorted((_CODE_ROOT / "pilot0/pool_coords").glob("*.json")):
        if p.name.startswith("FAILED") or ".v1." in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("schema") == 2:
            paths.append(r["model_path"])
    assert len(paths) == 280, f"expected 280 pool records, got {len(paths)}"
    return paths


def new_set_loader(cname: str, datamodule, resize_img):
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    gray = cname in ("mnist_new", "fashionmnist_new", "kmnist_new")
    steps = ([transforms.Grayscale(num_output_channels=3)] if gray else [])
    steps += [transforms.Resize(resize_img), transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])]
    tf = transforms.Compose(steps)
    root = Path(datamodule.data_root_dir)
    if cname == "mnist_new":
        ds = torchvision.datasets.MNIST(str(root / "mnist_new"),
                                        train=False, download=True,
                                        transform=tf)
    elif cname == "fashionmnist_new":
        ds = torchvision.datasets.FashionMNIST(str(root / "fmnist_new"),
                                               train=False, download=True,
                                               transform=tf)
    elif cname == "kmnist_new":
        ds = torchvision.datasets.KMNIST(str(root / "kmnist_new"),
                                         train=False, download=True,
                                         transform=tf)
    else:
        ds = torchvision.datasets.STL10(str(root / "stl10_new"),
                                        split="test", download=True,
                                        transform=tf)
    return DataLoader(ds, batch_size=datamodule.batch_size,
                      num_workers=datamodule.num_workers, shuffle=False)


def extract_one(model_path: str, out_dir: Path, use_cuda: bool) -> None:
    import torch  # noqa: F401 - torch env before fd_shifts imports
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    slug = model_path.replace("/", "__")
    source = SRC_KEY[model_path.split("_paper_sweep/")[0]]
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
    iid_token, plan, _notes = build_ood_plan(cf, source)
    test_loaders = datamodule.test_dataloader()

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_tr = ev["encoded"].cpu().numpy().astype(np.float32)
    y_tr = ev["labels"].cpu().numpy().astype(np.int64)
    fm = fit_feature_model(h_tr, y_tr, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_tr.astype(np.float64), y_tr, n_classes)
    train_mean = fm.global_mean
    del h_tr, y_tr

    def scores_for(h: np.ndarray) -> dict:
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"],
                "MLS": hs_["MLS"], "CTM": ctm(h64, proto_unc),
                "Maha": maha(h64), "fDBD": fdbd(h64, g, w_np, train_mean),
                "_logits": g}

    record: dict = {"schema_icml_b": 1, "model_path": model_path,
                    "slug": slug, "source": source,
                    "n_classes": n_classes,
                    "geometry": geometry_record(w_np, b_np, fm),
                    "papyan": papyan_metrics(w_np, fm), "ood": {}}

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_id = ev["encoded"].cpu().numpy().astype(np.float32)
    y_id = ev["labels"].cpu().numpy().astype(np.int64)
    sc_id = scores_for(h_id)
    res_id = (sc_id.pop("_logits").argmax(1) != y_id).astype(float)
    record["iid_test"] = dict(estimate_ood_coords(h_id, fm),
                              n=int(len(h_id)),
                              id_error_rate=float(res_id.mean()))
    del ev, h_id

    resize_img = ((64, 64) if str(cf.data.dataset) == "tiny-imagenet-200"
                  else (32, 32))
    for cname in NEW_SETS:
        try:
            logger.info(f"{slug}: forward {cname}")
            ev = forward_loader(model,
                                new_set_loader(cname, datamodule,
                                               resize_img))
            h_o = ev["encoded"].cpu().numpy().astype(np.float32)
            sc_o = scores_for(h_o)
            sc_o.pop("_logits")
            record["ood"][cname] = dict(
                estimate_ood_coords(h_o, fm),
                **set_outcomes(sc_id, res_id, sc_o))
            del ev, h_o
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            record["ood"][cname] = {"error": str(err)}
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(record, indent=1, default=float))
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    logger.info(f"{slug}: wrote {len(record['ood'])} new-set records "
                f"({record['runtime_sec']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--shard", type=str, default="1/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()
    k, n = (int(x) for x in args.shard.split("/"))
    paths = pool_model_paths()[k - 1::n]
    out_dir = Path(args.out_dir)
    todo = [p for p in paths
            if not (out_dir / f"{p.replace('/', '__')}.json").exists()]
    print(f"[roster-b] shard {args.shard}: {len(paths)} targets, "
          f"{len(todo)} to run", flush=True)
    if args.list:
        for p in todo[:10]:
            print("  ", p)
        print(f"   ... ({len(todo)} total)")
        return
    failures = 0
    for i, p in enumerate(todo, 1):
        print(f"[roster-b] {i}/{len(todo)}: {p}", flush=True)
        try:
            extract_one(p, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{p.replace('/', '__')}.json").write_text(
                json.dumps({"model_path": p,
                            "error": traceback.format_exc()}, indent=1))
            print(f"[roster-b] FAILED {p}", flush=True)
    print(f"[roster-b] done: {len(todo) - failures} ok, {failures} failed",
          flush=True)


if __name__ == "__main__":
    main()
