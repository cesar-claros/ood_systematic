"""Stage-2 source-expansion extraction: full SVHN + BREEDS checkpoint
sweeps (HPC side; source-expansion protocol stage 2 items 1-3).

FROZEN SPECIFICATION (protocol frozen 2026-08-26; amendments 1-5 in
documentation/source_expansion_protocol.md; this header declared before any
Stage-2 outcome is computed)
==========================================================================

Targets: every checkpoint of `svhn_paper_sweep` (expected 60) and
`breeds_paper_sweep` (expected 28) on the experiment store, matching
`<paradigm>_bb<bb>_do<d>_run<r>_rew<w>`; enumeration is checked against the
frozen manifest counts and any missing/extra directory is reported. Each
checkpoint needs its runtime hydra config (generated once via
`_fd_shifts_exec ... exp.mode=test`, protocol amendment 4); checkpoints
without one are recorded as FAILED, never silently skipped.

Suites (amendment 5a/5b): SVHN = its 3 configured semantic externals PLUS
the five 32x32 custom sets through the identical pool custom loaders
(test_6..test_10, `src/utils.compute_model_evaluations`), 8 sets total.
BREEDS = its single configured subclass shift only. Noise studies are
counted for loader indexing, never forwarded.

Scores (amendment 5d): computed on extracted activations with the
validated feature-level mirrors (`pilot0/scores.py`; G1/G5-gated).
Claim-bearing pair: Energy and CTM (max cosine to UNcentered train class
means). Secondary (ranking diagnostics only): MSR, MLS, Mahalanobis
(shared-covariance, train-fit), fDBD. Logits come from the head sliced to
n_classes exactly as the pool extractor does.

Outcomes (amendment 5c): per (checkpoint, set) and score, failure AUGRC on
the natural mixture (raw) AND on the prevalence-balanced mixture (both
sides subsampled without replacement to k = min(n_id_test, n_ood),
`numpy.random.default_rng(20260827)` fresh per set; pi = 0.5). Residuals =
[misclassified ID test, all-ones OOD] (the G5-validated construction).
Signed Energy-CTM gaps are recorded raw and balanced; MATERIALITY
(|gap| >= 0.01) applies to the balanced gap. Stage-2 records are outcomes
of record: the Stage-1 direction-withholding rule does not apply.

Geometry and coordinates: the frozen pilot0 estimators, unchanged
(fit_feature_model / geometry_record / papyan_metrics on the train split
through the experiment's own configured train augmentations, the
registered pipeline convention; estimate_ood_coords per evaluation set;
ID-test coordinates with the train-to-test rho).

Failure isolation per checkpoint (FAILED_<slug>.json); resumable; --source
and --shard k/n compose; --list enumerates without running.

Usage (from code/, inside the campaign container, .env with
EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR):
    python pilot0/extract_stage2_expansion.py --list
    python pilot0/extract_stage2_expansion.py --source svhn
    python pilot0/extract_stage2_expansion.py --source breeds [--shard 1/2]
Outputs: pilot0/stage2_expansion_coords/<slug>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.extract_pool_coords import CUSTOM_SETS, forward_loader
from pilot0.extract_stage1_pilot import PATH_PREFIXES, flatten_externals
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer, auroc, ctm, fdbd, head_scores

SCHEMA_S2 = 1
MATERIALITY_AUGRC = 0.01
BALANCE_SEED = 20260827
OUT_DIR_DEFAULT = "pilot0/stage2_expansion_coords"
FAMILIES = {"svhn": ("svhn_paper_sweep", 60, True),
            "breeds": ("breeds_paper_sweep", 28, False)}
NAME_RE = re.compile(
    r"^(?P<paradigm>[a-z]+)_bb(?P<bb>[a-z0-9_]+)_do(?P<do>\d)_run"
    r"(?P<run>\d+)_rew(?P<rew>[\d.]+)$")
SCORE_NAMES = ("Energy", "CTM", "MSR", "MLS", "Maha", "fDBD")


def enumerate_targets(source: str) -> tuple[list[dict], list[str]]:
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    fam, expected, _ = FAMILIES[source]
    found, notes = [], []
    for prefix in PATH_PREFIXES:
        base = root / prefix / fam
        if not base.is_dir():
            continue
        for p in sorted(base.iterdir()):
            m = NAME_RE.match(p.name)
            if not m:
                continue
            found.append({
                "source": source, "model_path": f"{prefix}{fam}/{p.name}",
                "paradigm": m["paradigm"], "reward": m["rew"],
                "dropout": int(m["do"]), "run": int(m["run"]),
                "has_config": (p / "hydra" / "config.yaml").is_file()})
        break
    if len(found) != expected:
        notes.append(f"{source}: found {len(found)} checkpoints, manifest "
                     f"expects {expected}")
    n_cfg = sum(t["has_config"] for t in found)
    if n_cfg < len(found):
        notes.append(f"{source}: {len(found) - n_cfg} checkpoints lack "
                     f"hydra/config.yaml (run exp.mode=test first)")
    return found, notes


def failure_augrc(confids: np.ndarray, residuals: np.ndarray) -> float:
    from src.rc_stats import RiskCoverageStats
    rc = RiskCoverageStats(confids=confids, residuals=residuals)
    return float(rc.augrc / rc.AUC_DISPLAY_SCALE)


def set_outcomes(sc_id: dict, res_id: np.ndarray, sc_ood: dict) -> dict:
    n_id, n_ood = len(res_id), len(next(iter(sc_ood.values())))
    res = np.concatenate([res_id, np.ones(n_ood)])
    k = min(n_id, n_ood)
    rng = np.random.default_rng(BALANCE_SEED)
    id_idx = (np.arange(n_id) if n_id == k
              else rng.choice(n_id, k, replace=False))
    ood_idx = (np.arange(n_ood) if n_ood == k
               else rng.choice(n_ood, k, replace=False))
    out: dict = {"n_id": n_id, "n_ood": n_ood, "k_balanced": int(k)}
    aug_raw, aug_bal = {}, {}
    for s in SCORE_NAMES:
        cid, cood = sc_id[s], sc_ood[s]
        out[f"auroc_id_vs_ood_{s}"] = round(auroc(cid, cood), 4)
        aug_raw[s] = failure_augrc(np.concatenate([cid, cood]), res)
        aug_bal[s] = failure_augrc(
            np.concatenate([cid[id_idx], cood[ood_idx]]),
            np.concatenate([res_id[id_idx], np.ones(k)]))
        out[f"augrc_raw_{s}"] = round(aug_raw[s], 5)
        out[f"augrc_balanced_{s}"] = round(aug_bal[s], 5)
    out["gap_raw"] = round(aug_raw["Energy"] - aug_raw["CTM"], 5)
    out["gap_balanced"] = round(aug_bal["Energy"] - aug_bal["CTM"], 5)
    out["material"] = bool(abs(out["gap_balanced"]) >= MATERIALITY_AUGRC)
    return out


def extract_one(target: dict, out_dir: Path, use_cuda: bool) -> None:
    import torch  # noqa: F401
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = target["model_path"]
    slug = model_path.replace("/", "__")
    source = target["source"]
    _, _, with_customs = FAMILIES[source]
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]

    iid_token, specs = flatten_externals(cf)
    test_loaders = datamodule.test_dataloader()

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_train = ev["encoded"].cpu().numpy().astype(np.float32)
    y_train = ev["labels"].cpu().numpy().astype(np.int64)
    fm = fit_feature_model(h_train, y_train, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_train.astype(np.float64), y_train, n_classes)
    train_mean = fm.global_mean

    def scores_for(h: np.ndarray) -> dict:
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs = head_scores(g)
        return {"Energy": hs["Energy"], "MSR": hs["MSR"], "MLS": hs["MLS"],
                "CTM": ctm(h64, proto_unc), "Maha": maha(h64),
                "fDBD": fdbd(h64, g, w_np, train_mean), "_logits": g}

    record: dict = {
        "schema_stage2": SCHEMA_S2, "model_path": model_path, "slug": slug,
        "study": study_name, "source": source,
        "paradigm": target["paradigm"], "reward": target["reward"],
        "dropout": target["dropout"], "run": target["run"],
        "n_classes": n_classes, "dim": int(h_train.shape[1]),
        "n_train": int(len(h_train)), "iid_token": iid_token,
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm),
        "ood": {},
    }

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_iid = ev["encoded"].cpu().numpy().astype(np.float32)
    y_iid = ev["labels"].cpu().numpy().astype(np.int64)
    sc_id = scores_for(h_iid)
    res_id = (sc_id["_logits"].argmax(1) != y_iid).astype(float)
    record["iid_test"] = dict(estimate_ood_coords(h_iid, fm),
                              n=int(len(h_iid)),
                              id_error_rate=round(float(res_id.mean()), 4))

    plan: list[tuple[str, str]] = []
    for spec in specs:
        if spec["forward"]:
            plan.append((spec["raw"], spec["token"]))
    if with_customs:
        plan += [(cname, tok) for cname, tok in CUSTOM_SETS.items()]

    n_material = 0
    for cname, token in plan:
        try:
            idx = int(token.split("_")[1])
            if idx <= 5:
                if idx >= len(test_loaders):
                    raise IndexError(f"{token} out of range "
                                     f"({len(test_loaders)} loaders)")
                logger.info(f"{slug}: forward {cname} ({token}, config)")
                ev = forward_loader(model, test_loaders[idx])
            else:
                logger.info(f"{slug}: forward {cname} ({token}, custom)")
                ev = utils.compute_model_evaluations(model, datamodule,
                                                     token)
            h_ood = ev["encoded"].cpu().numpy().astype(np.float32)
            sc_ood = scores_for(h_ood)
            entry = dict(estimate_ood_coords(h_ood, fm), token=token)
            entry.update(set_outcomes(sc_id, res_id, sc_ood))
            n_material += int(entry["material"])
            record["ood"][cname] = entry
            del ev, h_ood, sc_ood
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            record["ood"][cname] = {"error": str(err), "token": token}

    ok = [c for c, v in record["ood"].items() if "error" not in v]
    record["n_sets_extracted"] = len(ok)
    record["n_sets_planned"] = len(plan)
    record["suite_complete"] = len(ok) == len(plan)
    record["n_material_sets"] = n_material
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(record, indent=1))
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    logger.info(f"{slug}: wrote stage-2 record ({len(ok)}/{len(plan)} sets, "
                f"{n_material} material, {record['runtime_sec']}s)")
    del h_train, y_train, h_iid, model, module, datamodule
    if use_cuda:
        import torch
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=str, default=None,
                        choices=sorted(FAMILIES))
    parser.add_argument("--shard", type=str, default="1/1")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    if args.batch_size is not None:
        os.environ["CSF_BATCH_SIZE"] = str(args.batch_size)
    if args.num_workers is not None:
        os.environ["CSF_NUM_WORKERS"] = str(args.num_workers)
    out_dir = Path(args.out_dir)

    sources = [args.source] if args.source else sorted(FAMILIES)
    targets, notes = [], []
    for s in sources:
        t, n = enumerate_targets(s)
        targets += t
        notes += n
    k, n = (int(x) for x in args.shard.split("/"))
    shard = targets[k - 1::n]
    print(f"[stage2] {len(targets)} targets over {sources}; shard {k}/{n} "
          f"-> {len(shard)}")
    for note in notes:
        print(f"[stage2] NOTE: {note}")
    if args.list:
        for t in shard:
            print(f"[stage2] {t['model_path']}"
                  + ("" if t["has_config"] else " [NO hydra/config.yaml]"))
        return

    import torch
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    done = skipped = failed = 0
    for i, t in enumerate(shard, 1):
        slug = t["model_path"].replace("/", "__")
        if (out_dir / f"{slug}.json").exists():
            skipped += 1
            continue
        print(f"[stage2 {k}/{n}] {i}/{len(shard)}: {t['model_path']}")
        try:
            extract_one(t, out_dir, use_cuda)
            done += 1
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"target": t, "traceback": traceback.format_exc()},
                indent=1))
            print(f"[stage2] FAILED {t['model_path']} (recorded, continuing)")
    print(f"[stage2 {k}/{n}] finished: {done} new, {skipped} skipped, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
