"""ICML roster C outcome extraction: empirical detector outcomes for
the 140 repair VALIDATION-half checkpoints on the REGISTERED 8-set
suite (HPC side; frozen protocol section 8.2/8.3, endpoint E3).

WHY THIS SCRIPT EXISTS: endpoint E3 compares predicted AUROC
(1 - exp(l)) against EMPIRICAL ID-vs-OOD AUROC per (checkpoint, set,
score). The frozen artifacts hold no such AUROCs for the pool (the
harmonized parquet stores failure AUGRC/AURC; pool_coords stores
coordinates only), and the roster-C sufficient-statistics extraction
(extract_repair_stats.py) computes NO outcome by its own frozen rule.
This companion pass supplies exactly the missing empirical side.

FROZEN CONVENTIONS (identical to the pool/stage-2 machinery):
- Roster: nc_csf_predictivity/outputs/track1/repair_valhalf_manifest
  .json (committed metadata-only enumeration; the script refuses to run
  without it).
- Loaders: the frozen pool plan (build_ood_plan; config loaders by
  index, custom branches, torchvision fallbacks) - the SAME sets the
  registered pool suite used.
- Scores: the frozen feature-level mirrors (Energy/CTM claim-bearing;
  MSR/MLS/Maha/fDBD secondary); outcomes via the frozen set_outcomes
  (auroc_id_vs_ood per score, raw + prevalence-balanced AUGRC,
  materiality, rng 20260827).
Per GR-5 the outputs stay UNREAD until the committed analysis suite
runs. Resumable; sharded; per-checkpoint FAILED_ isolation.

Usage (HPC, inside the container, from code/):
    python pilot0/extract_roster_c_outcomes.py --list
    python pilot0/extract_roster_c_outcomes.py [--shard k/n]
Output: pilot0/icml_roster_c_outcomes/<slug>.json
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
from pilot0.extract_stage2_expansion import set_outcomes
from pilot0.geometry import fit_feature_model
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

MANIFEST = Path("nc_csf_predictivity/outputs/track1/"
                "repair_valhalf_manifest.json")
OUT_DIR_DEFAULT = "pilot0/icml_roster_c_outcomes"


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
    n_loaders = len(test_loaders)

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

    record: dict = {"schema_icml_c": 1, "model_path": model_path,
                    "slug": slug, "source": source,
                    "n_classes": n_classes, "ood": {}}

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_id = ev["encoded"].cpu().numpy().astype(np.float32)
    y_id = ev["labels"].cpu().numpy().astype(np.int64)
    sc_id = scores_for(h_id)
    res_id = (sc_id.pop("_logits").argmax(1) != y_id).astype(float)
    record["iid_test"] = {"n": int(len(h_id)),
                          "id_error_rate": float(res_id.mean())}
    del ev, h_id

    resize_img = ((64, 64) if str(cf.data.dataset) == "tiny-imagenet-200"
                  else (32, 32))
    for cname, spec in plan.items():
        try:
            if spec["kind"] == "config":
                idx = int(spec["token"].split("_")[1])
                if idx >= n_loaders:
                    raise IndexError(f"{spec['token']} out of range "
                                     f"({n_loaders} loaders)")
                logger.info(f"{slug}: forward {cname} (config)")
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
    logger.info(f"{slug}: wrote {len(record['ood'])} set outcomes "
                f"({record['runtime_sec']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--shard", type=str, default="1/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                    default=True)
    args = ap.parse_args()
    assert MANIFEST.exists(), ("PROTOCOL GATE: commit "
                               f"{MANIFEST} (make_repair_valhalf_"
                               "manifest.py) before extraction")
    man = json.loads(MANIFEST.read_text())
    paths = sorted(e["model_path"] for e in man["pool"])
    assert len(paths) == man["n_pool"] == 140, (len(paths), man["n_pool"])
    k, n = (int(x) for x in args.shard.split("/"))
    paths = paths[k - 1::n]
    out_dir = Path(args.out_dir)
    todo = [p for p in paths
            if not (out_dir / f"{p.replace('/', '__')}.json").exists()]
    print(f"[roster-c] shard {args.shard}: {len(paths)} targets, "
          f"{len(todo)} to run", flush=True)
    if args.list:
        for p in todo[:10]:
            print("  ", p)
        print(f"   ... ({len(todo)} total)")
        return
    failures = 0
    for i, p in enumerate(todo, 1):
        print(f"[roster-c] {i}/{len(todo)}: {p}", flush=True)
        try:
            extract_one(p, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{p.replace('/', '__')}.json").write_text(
                json.dumps({"model_path": p,
                            "error": traceback.format_exc()}, indent=1))
            print(f"[roster-c] FAILED {p}", flush=True)
    print(f"[roster-c] done: {len(todo) - failures} ok, {failures} failed",
          flush=True)


if __name__ == "__main__":
    main()
