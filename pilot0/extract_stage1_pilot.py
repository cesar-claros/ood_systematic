"""Stage-1 feasibility pilot for the frozen source-expansion protocol
(documentation/source_expansion_protocol.md, frozen 2026-08-26, stage 1;
HPC side).

FROZEN SPECIFICATION (declared before any new-source outcome is computed)
=========================================================================

Purpose: per-source run/no-run feasibility ONLY (protocol stage 1). This
script verifies checkpoint loading, feature and logit availability,
class-mean support, coordinate extraction, metadata recoverability,
config-declared shift mappings, severity-condition counts, material-gap
existence, and runtime/storage. It never decides scientific favorability,
and its outputs are feasibility indicators, not claim-bearing outcomes.

Checkpoint selection (protocol section 8, deterministic, metadata only):
per source, confidnet do1 run1, devries do1 run1, and dg do1 run1 at the
lowest, median, and highest reward of that source's grid. CLARIFICATION
(2026-08-26, metadata-only, recorded in the protocol amendment log): on an
even-sized reward grid the median is the lower median, sorted index
(n-1)//2; SVHN's grid {2.2, 3, 6, 10} therefore uses reward 3. Substitution
ladder if a named checkpoint is absent on disk: do0, then run2 (do1_run1 ->
do0_run1 -> do1_run2 -> do0_run2); any substitution is recorded in the
output JSON. All fifteen declared checkpoints exist in the frozen manifest
(documentation/source_expansion_manifest.txt).

Sources and backbones (frozen): svhn -> svhn_paper_sweep (svhn_small_conv);
breeds -> breeds_paper_sweep (resnet50); iwildcam -> animals_paper_sweep
(resnet50). Path resolution under EXPERIMENT_ROOT_DIR tries the bare family
path first, then the 'fd-shifts/' prefix (upstream release layout); the
resolved path is recorded.

Shift suite (protocol section 2.6): the experiment's OWN configured
query_studies, flattened exactly as fd_shifts.loaders.data_loader does
(loader order: val-tuning?, iid, then every non-iid group in config order).
Every declared external is recorded; noise-study ('corrupt*') sets are
counted for loader indexing but not forwarded (paper-suite convention);
everything else is forwarded DIRECTLY by loader index. No CIFAR-suite
custom branches (test_6..test_10) and no torchvision fallbacks are used for
the new sources; whether comparable custom sets should exist is a Stage-2
protocol decision, not a pilot improvisation.

Measurements per checkpoint (all estimators FROZEN elsewhere, none refit):
geometry panel + Papyan metrics (pilot0/geometry.py), ID-test and per-set
OOD coordinates (pilot0/ood_coords.py), classifier head sliced to
n_classes exactly as pilot0/extract_pool_coords.py does (DeepGamblers'
reservation column dropped), per-class train counts (min / median / number
empty). If any train class is empty, the frozen feature model is not
evaluated (recorded as an error), and CTM prototypes are restricted to
present classes; Energy is unaffected.

Feasibility scores (protocol stage-1 items "Energy and CTM can be computed
identically" and "material gaps exist"): Energy = log-sum-exp of the sliced
head logits (pilot0/scores.head_scores); CTM = max cosine to the UNcentered
train class means (pilot0/scores.ctm; the deployed pipeline's CTM
convention per the MC phase audit). AUGRC uses the G5-validated
construction (pilot0/run_pilot0.py: residuals = [misclassified ID test,
all-ones OOD], src.rc_stats.RiskCoverageStats, augrc / AUC_DISPLAY_SCALE).
DIRECTION WITHHOLDING (declared): the per-set record stores each score's
ID-vs-OOD AUROC (orientation check), |AUGRC_Energy - AUGRC_CTM|, and the
materiality flag (|gap| >= 0.01), but NOT the signed gap; winner directions
first materialize in Stage 2 through the deployed pipeline. The printed
summary shows counts only.

Failure isolation: FAILED_<slug>.json per checkpoint; resumable (existing
schema-1 stage-1 outputs are skipped). Usage (from code/, inside the
campaign container, .env with EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR):
    python pilot0/extract_stage1_pilot.py --list
    python pilot0/extract_stage1_pilot.py [--source svhn|breeds|iwildcam]
Outputs: pilot0/stage1_pilot_coords/<slug>.json
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

from pilot0.extract_pool_coords import forward_loader
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import auroc, ctm, head_scores

SCHEMA_S1 = 1
MATERIALITY_AUGRC = 0.01
OUT_DIR_DEFAULT = "pilot0/stage1_pilot_coords"
PATH_PREFIXES = ("", "fd-shifts/")

FAMILY = {"svhn": ("svhn_paper_sweep", "svhn_small_conv"),
          "breeds": ("breeds_paper_sweep", "resnet50"),
          "iwildcam": ("animals_paper_sweep", "resnet50")}
DG_GRID = {"svhn": ["2.2", "3", "6", "10"],
           "breeds": ["2.2", "3", "6", "10", "15"],
           "iwildcam": ["2.2", "3", "6", "10", "15"]}


def median_reward(grid: list[str]) -> str:
    ordered = sorted(grid, key=float)
    return ordered[(len(ordered) - 1) // 2]


def pilot_cells(source: str) -> list[tuple[str, str]]:
    """(paradigm, reward) cells in the frozen order."""
    grid = DG_GRID[source]
    return [("confidnet", "2.2"), ("devries", "2.2"),
            ("dg", sorted(grid, key=float)[0]),
            ("dg", median_reward(grid)),
            ("dg", sorted(grid, key=float)[-1])]


def candidate_names(source: str, paradigm: str, rew: str) -> list[str]:
    _, bb = FAMILY[source]
    return [f"{paradigm}_bb{bb}_do{do}_run{run}_rew{rew}"
            for run in (1, 2) for do in (1, 0)]  # do1_run1,do0_run1,do1_run2,do0_run2


def resolve_targets(source: str) -> list[dict]:
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    fam, _ = FAMILY[source]
    out = []
    for paradigm, rew in pilot_cells(source):
        cands = candidate_names(source, paradigm, rew)
        chosen = None
        for i, name in enumerate(cands):
            for prefix in PATH_PREFIXES:
                d = root / prefix / fam / name
                if d.is_dir():
                    chosen = {"source": source, "paradigm": paradigm,
                              "reward": rew,
                              "model_path": f"{prefix}{fam}/{name}",
                              "substituted": i > 0,
                              "declared": cands[0],
                              "has_config": (d / "hydra"
                                             / "config.yaml").is_file()}
                    break
            if chosen:
                break
        if chosen is None:
            chosen = {"source": source, "paradigm": paradigm, "reward": rew,
                      "model_path": None, "substituted": None,
                      "declared": cands[0], "missing_all": cands}
        out.append(chosen)
    return out


def flatten_externals(cf) -> tuple[str, list[dict]]:
    """(iid_token, ordered external specs) mirroring the fd-shifts loader."""
    qs = dict(cf.eval.query_studies)
    iid_idx = 1 if bool(cf.eval.val_tuning) else 0
    specs = []
    i = 0
    for key, values in qs.items():
        if key == "iid_study" or values is None:
            continue
        if isinstance(values, str):
            vals = [values]
        else:
            try:
                vals = [str(v) for v in values]
            except TypeError:
                vals = [values]
        for raw in vals:
            raw = str(raw)
            idx = iid_idx + 1 + i
            i += 1
            forward = not raw.lower().startswith("corrupt")
            specs.append({"group": key, "raw": raw, "token": f"test_{idx}",
                          "forward": forward})
    return f"test_{iid_idx}", specs


def safe_class_means(h: np.ndarray, y: np.ndarray,
                     n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y, minlength=n_classes)
    present = counts > 0
    means = np.stack([h[y == c].mean(0) for c in range(n_classes)
                      if present[c]])
    return means.astype(np.float64), counts


def pilot_one(target: dict, out_dir: Path, use_cuda: bool) -> dict:
    import torch  # noqa: F401 - ensures torch env before fd_shifts imports
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader

    from src import utils
    from src.rc_stats import RiskCoverageStats
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model

    t0 = time.time()
    model_path = target["model_path"]
    slug = model_path.replace("/", "__")
    cf, module, study_name = load_model(model_path, use_cuda)
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    if study_name == "vit":
        w, b = utils.get_model_and_last_layer(module, study_name,
                                              return_model=False)
    else:
        _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]

    iid_token, specs = flatten_externals(cf)
    test_loaders = datamodule.test_dataloader()
    record: dict = {
        "schema_stage1": SCHEMA_S1,
        "model_path": model_path, "slug": slug, "study": study_name,
        "source": target["source"], "paradigm": target["paradigm"],
        "reward": target["reward"], "substituted": target["substituted"],
        "declared_checkpoint": target["declared"],
        "n_classes": n_classes,
        "dataset": str(cf.data.dataset),
        "img_size": str(getattr(cf.data, "img_size", None)),
        "backbone": str(getattr(cf.model, "network", None) or
                        getattr(cf.model, "name", None)),
        "iid_token": iid_token,
        "n_loaders_exposed": len(test_loaders),
        "declared_externals": specs,
        "n_forwardable_externals": sum(s["forward"] for s in specs),
    }

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_train = ev["encoded"].cpu().numpy().astype(np.float32)
    y_train = ev["labels"].cpu().numpy().astype(np.int64)
    counts = np.bincount(y_train, minlength=n_classes)
    record.update({
        "dim": int(h_train.shape[1]), "n_train": int(len(h_train)),
        "class_support": {"min": int(counts.min()),
                          "median": float(np.median(counts)),
                          "n_empty": int((counts == 0).sum())},
    })

    fm = None
    if counts.min() > 0:
        fm = fit_feature_model(h_train, y_train, n_classes)
        record["geometry"] = geometry_record(w_np, b_np, fm)
        record["papyan"] = papyan_metrics(w_np, fm)
        proto_unc = fm.class_means + fm.global_mean
    else:
        record["geometry"] = {"error": f"{int((counts == 0).sum())} empty "
                              "train classes; frozen feature model not "
                              "evaluated"}
        proto_unc, _ = safe_class_means(h_train.astype(np.float64), y_train,
                                        n_classes)

    def scores_for(h: np.ndarray) -> dict[str, np.ndarray]:
        g = h.astype(np.float64) @ w_np.T + b_np
        return {"Energy": head_scores(g)["Energy"],
                "CTM": ctm(h.astype(np.float64), proto_unc), "_logits": g}

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test ({iid_token})")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_iid = ev["encoded"].cpu().numpy().astype(np.float32)
    y_iid = ev["labels"].cpu().numpy().astype(np.int64)
    sc_id = scores_for(h_iid)
    res_id = (sc_id["_logits"].argmax(1) != y_iid).astype(float)
    record["iid_test"] = {"n": int(len(h_iid)),
                          "id_error_rate": round(float(res_id.mean()), 4)}
    if fm is not None:
        record["iid_test"].update(estimate_ood_coords(h_iid, fm))

    record["ood"] = {}
    n_material = 0
    for spec in specs:
        if not spec["forward"]:
            record["ood"][spec["raw"]] = {"skipped": "noise study",
                                          **{k: spec[k] for k in
                                             ("group", "token")}}
            continue
        try:
            idx = int(spec["token"].split("_")[1])
            if idx >= len(test_loaders):
                raise IndexError(f"{spec['token']} out of range "
                                 f"({len(test_loaders)} loaders)")
            logger.info(f"{slug}: forward {spec['raw']} ({spec['token']})")
            ev = forward_loader(model, test_loaders[idx])
            h_ood = ev["encoded"].cpu().numpy().astype(np.float32)
            sc_ood = scores_for(h_ood)
            entry: dict = {"n": int(len(h_ood)), "group": spec["group"],
                           "token": spec["token"]}
            if fm is not None:
                entry.update(estimate_ood_coords(h_ood, fm))
            res = np.concatenate([res_id, np.ones(len(h_ood))])
            augrcs = {}
            for score in ("Energy", "CTM"):
                confids = np.concatenate([sc_id[score], sc_ood[score]])
                entry[f"auroc_id_vs_ood_{score}"] = round(
                    auroc(sc_id[score], sc_ood[score]), 4)
                rc = RiskCoverageStats(confids=confids, residuals=res)
                augrcs[score] = rc.augrc / rc.AUC_DISPLAY_SCALE
            gap = abs(augrcs["Energy"] - augrcs["CTM"])
            entry["abs_augrc_gap"] = round(float(gap), 4)
            entry["material"] = bool(gap >= MATERIALITY_AUGRC)
            n_material += int(entry["material"])
            record["ood"][spec["raw"]] = entry
            del ev, h_ood, sc_ood
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {spec['raw']} FAILED: {err}")
            record["ood"][spec["raw"]] = {"error": str(err), **spec}

    n_ok = sum(1 for v in record["ood"].values()
               if "error" not in v and "skipped" not in v)
    record["n_sets_extracted"] = n_ok
    record["n_material_sets"] = n_material
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.json"
    out_path.write_text(json.dumps(record, indent=1))
    record["json_bytes"] = out_path.stat().st_size
    failed_marker = out_dir / f"FAILED_{slug}.json"
    if failed_marker.exists():
        failed_marker.unlink()
    logger.info(f"{slug}: stage-1 record written ({n_ok} sets, "
                f"{n_material} material, {record['runtime_sec']}s)")
    del h_train, y_train, h_iid, model, module, datamodule
    if use_cuda:
        import torch
        torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=str, default=None,
                        choices=sorted(FAMILY),
                        help="restrict to one source (default: all three)")
    parser.add_argument("--list", action="store_true",
                        help="resolve and print targets, then exit")
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

    sources = [args.source] if args.source else sorted(FAMILY)
    targets = [t for s in sources for t in resolve_targets(s)]
    print(f"[stage1] {len(targets)} pilot targets over {sources}")
    for t in targets:
        state = ("MISSING (all candidates)" if t["model_path"] is None
                 else t["model_path"]
                 + (" [SUBSTITUTED]" if t["substituted"] else "")
                 + ("" if t.get("has_config")
                    else " [NO hydra/config.yaml: run exp.mode=test first]"))
        print(f"[stage1] {t['source']}/{t['paradigm']}/rew{t['reward']}: "
              f"{state}")
    if args.list:
        return

    import torch
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    done = skipped = failed = 0
    for i, t in enumerate(targets, 1):
        if t["model_path"] is None:
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{t['declared']}.json").write_text(
                json.dumps({"target": t,
                            "error": "no candidate on disk"}, indent=1))
            continue
        slug = t["model_path"].replace("/", "__")
        if (out_dir / f"{slug}.json").exists():
            skipped += 1
            continue
        print(f"[stage1] {i}/{len(targets)}: {t['model_path']}")
        try:
            pilot_one(t, out_dir, use_cuda)
            done += 1
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"target": t, "traceback": traceback.format_exc()},
                indent=1))
            print(f"[stage1] FAILED {t['model_path']} (recorded, continuing)")
    print(f"[stage1] finished: {done} new, {skipped} skipped, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
