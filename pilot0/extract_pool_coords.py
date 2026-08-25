"""Stage-2 Option B (audit #5 section 12; frozen design in
documentation/heldout_theory_validation_design.md): pool-wide OOD-coordinate
extraction for the 280-checkpoint VGG-13 pool (HPC side).

Per checkpoint: forwards the train split, ID test, and the source's paper OOD
suite; fits the frozen pilot0 feature model on train; writes ONE small JSON
with the measured geometry (Papyan panel, logit scale, SNR, anisotropy), the
ID-test coordinates (its rho is the train-to-test generalization ratio), and
the frozen H-estimator coordinates (gamma, a, rho, w_perp, ...) for every OOD
set, keyed by CANONICAL parquet set names. No feature cache is retained
(--keep_npz opts in).

SCHEMA 2 (2026-08-25). The v1 hardcoded test_3/test_4/test_5 labels were only
valid for the cifar10/cifar100 loader layout and crashed (or could mislabel)
on other sources. v2 derives the mapping from the experiment's own saved
config at runtime: cf.eval.query_studies is flattened exactly as
fd_shifts.loaders.data_loader does, loader indices follow (val-tuning, iid,
externals...), the corrupt-* noise study is excluded (not part of the paper
suite), and the five source-independent custom sets stay at test_6..test_10.
The produced set list is checked against the source's expected 8-set suite
(from the harmonized table); any expected set the config does not declare is
built as a torchvision fallback with the same transform recipe as the custom
branches and tagged "custom_fallback" in the JSON, and anything still missing
is recorded as an explicit error entry, never silently skipped. Existing v1
JSONs are renamed to <slug>.v1.json and re-extracted; FAILED_ records are
cleared on later success.

The coordinate estimators are the FROZEN pilot0 definitions
(pilot0/ood_coords.py, frozen 2026-08-15); nothing here refits or retunes
them. Failures are isolated per checkpoint (FAILED_<slug>.json records the
error; the sweep continues), and the sweep is resumable (schema-2 outputs are
skipped).

Batch size / workers: --batch_size and --num_workers set the CSF_BATCH_SIZE /
CSF_NUM_WORKERS overrides read by load_model (defaults 128 / 12; the env
variables also work directly, the flags win). Extracted coordinates are
batch-size invariant; the flag only trades GPU memory against speed.

Usage (from code/, inside the campaign container, .env with
EXPERIMENT_ROOT_DIR/DATASET_ROOT_DIR):
    # one checkpoint
    python pilot0/extract_pool_coords.py --model_path \
        cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2
    # full pool sweep, optionally sharded across GPUs/jobs
    python pilot0/extract_pool_coords.py --sweep --shard 1/2
    python pilot0/extract_pool_coords.py --sweep --shard 2/2
    # sweep one ID source at a time (accepts parquet or directory names:
    # cifar10, cifar100, supercifar100/supercifar, tinyimagenet/tiny-imagenet-200)
    python pilot0/extract_pool_coords.py --sweep --source cifar100
    # --source and --shard compose: shard k/n WITHIN the filtered source
    python pilot0/extract_pool_coords.py --sweep --source supercifar100 --shard 1/2
    # enumerate without running
    python pilot0/extract_pool_coords.py --sweep --list [--source ...]
Outputs: pilot0/pool_coords/<slug>.json  (rsync the whole folder back)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(1, str(_CODE_ROOT / "x6_spectral"))

from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords

SCHEMA = 2

# Source-independent custom loaders hardcoded in
# src/utils.compute_model_evaluations (test_6..test_10).
CUSTOM_SETS = {"lsun cropped": "test_6", "lsun resize": "test_7",
               "isun": "test_8", "textures": "test_9",
               "places365": "test_10"}

# Expected 8-set paper suite per source (canonical parquet names; from the
# harmonized table, cross-checked in data_unit_report.md).
EXPECTED_SUITE = {
    "cifar10": ["cifar100", "svhn", "tinyimagenet"] + list(CUSTOM_SETS),
    "cifar100": ["cifar10", "svhn", "tinyimagenet"] + list(CUSTOM_SETS),
    "supercifar100": ["cifar10", "svhn", "tinyimagenet"] + list(CUSTOM_SETS),
    "tinyimagenet": ["cifar10", "cifar100", "svhn"] + list(CUSTOM_SETS),
}

CNN_RE = re.compile(
    r"^(?P<src>[a-z0-9\-]+)_paper_sweep/(?P<paradigm>[a-z]+)_bb"
    r"(?P<bb>vgg13)_do(?P<do>\d)_run(?P<run>\d+)_rew(?P<rew>[\d.]+)$")
SRC_KEY = {"cifar10": "cifar10", "cifar100": "cifar100",
           "supercifar": "supercifar100", "tiny-imagenet-200": "tinyimagenet"}
MANIFEST = Path(__file__).resolve().parent / "pool_manifest.json"
OUT_DIR_DEFAULT = "pilot0/pool_coords"


def canon_external(raw: str) -> str | None:
    """Config dataset token -> canonical parquet set name (None = not part
    of the paper suite: corrupt-* noise study or unknown)."""
    n = str(raw).lower()
    for suf in ("_384", "_64"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    if n.startswith("corrupt"):
        return None
    if n in ("tinyimagenet", "tinyimagenet_resize", "tiny-imagenet-200"):
        return "tinyimagenet"
    if n in ("cifar10", "cifar100", "svhn"):
        return n
    return None


def build_ood_plan(cf, source: str) -> tuple[str, dict, list[str]]:
    """Derive (iid_token, plan, notes) from the experiment's own config.

    plan: canonical set name -> {"kind": config|custom|fallback,
    "token"/"dataset": ..., "raw": ...}. Replicates the loader-list order of
    fd_shifts.loaders.data_loader: [val-tuning?, iid, *flattened externals].
    """
    notes: list[str] = []
    qs = dict(cf.eval.query_studies)
    externals: list[str] = []
    for key, values in qs.items():
        if key != "iid_study" and values is not None:
            externals.extend([str(v) for v in values])
    iid_idx = 1 if bool(cf.eval.val_tuning) else 0
    plan: dict[str, dict] = {}
    for i, raw in enumerate(externals):
        idx = iid_idx + 1 + i
        cname = canon_external(raw)
        if cname is None:
            notes.append(f"config external '{raw}' (test_{idx}) not in the "
                         f"paper suite; skipped")
            continue
        plan[cname] = {"kind": "config", "token": f"test_{idx}", "raw": raw}
    for cname, tok in CUSTOM_SETS.items():
        plan[cname] = {"kind": "custom", "token": tok, "raw": cname}
    for cname in EXPECTED_SUITE[source]:
        if cname not in plan:
            plan[cname] = {"kind": "fallback", "dataset": cname,
                           "raw": f"{cname} (not declared by config)"}
            notes.append(f"expected set '{cname}' not declared by the "
                         f"experiment config; using torchvision fallback")
    unexpected = [c for c in plan if c not in EXPECTED_SUITE[source]]
    for c in unexpected:
        notes.append(f"config declares '{c}', which is not in the expected "
                     f"{source} suite; extracting it anyway (extra)")
    return f"test_{iid_idx}", plan, notes


def fallback_loader(cname: str, datamodule, resize_img):
    """Torchvision loader for an expected set the config does not declare,
    with the SAME transform recipe as the custom test_6..test_10 branches."""
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(resize_img),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    root = Path(datamodule.data_root_dir)
    if cname == "cifar10":
        ds = torchvision.datasets.CIFAR10(str(root), train=False,
                                          download=True, transform=transform)
    elif cname == "cifar100":
        ds = torchvision.datasets.CIFAR100(str(root), train=False,
                                           download=True, transform=transform)
    elif cname == "svhn":
        ds = torchvision.datasets.SVHN(str(root / "svhn"), split="test",
                                       download=True, transform=transform)
    elif cname == "tinyimagenet":
        candidates = ["tinyimagenet_resize", "TinyImagenet_resize",
                      "Imagenet_resize", "tiny-imagenet-200/val"]
        for cand in candidates:
            if (root / cand).is_dir():
                ds = torchvision.datasets.ImageFolder(str(root / cand),
                                                      transform=transform)
                break
        else:
            raise FileNotFoundError(
                f"no tinyimagenet fallback dir under {root} "
                f"(tried {candidates})")
    else:
        raise NotImplementedError(f"no fallback for {cname}")
    return DataLoader(ds, batch_size=datamodule.batch_size,
                      num_workers=datamodule.num_workers, shuffle=False)


def forward_loader(model, dataloaders):
    """Replicates the concat loop of src.utils.compute_model_evaluations."""
    import torch
    from tqdm import tqdm

    results = [model(batch, i)
               for i, batch in enumerate(tqdm(dataloaders, position=0,
                                              leave=True))]
    out = {}
    for k in results[0].keys():
        try:
            out[k] = torch.concat([d[k] for d in results])
        except Exception:  # noqa: BLE001 - mirror utils' tolerant concat
            out[k] = None
    return out


def extract_one(model_path: str, out_dir: Path, use_cuda: bool,
                keep_npz: bool) -> dict:
    import torch
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
                          rank_feat=False, ash_method=None, use_cuda=use_cuda)
    if study_name == "vit":
        w, b = utils.get_model_and_last_layer(module, study_name,
                                              return_model=False)
    else:
        _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]

    iid_token, plan, notes = build_ood_plan(cf, source)
    test_loaders = datamodule.test_dataloader()
    n_loaders = len(test_loaders)
    for note in notes:
        logger.warning(f"{slug}: {note}")

    logger.info(f"{slug}: forward train")
    ev = utils.compute_model_evaluations(model, datamodule, "train")
    h_train = ev["encoded"].cpu().numpy().astype(np.float32)
    y_train = ev["labels"].cpu().numpy().astype(np.int64)

    fm = fit_feature_model(h_train, y_train, n_classes)
    record: dict = {
        "schema": SCHEMA,
        "model_path": model_path, "slug": slug, "study": study_name,
        "source": source, "n_classes": n_classes,
        "dim": int(h_train.shape[1]), "n_train": int(len(h_train)),
        "iid_token": iid_token,
        "ood_plan": {c: {k: v for k, v in p.items()}
                     for c, p in plan.items()},
        "plan_notes": notes,
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm),
        "ood": {},
    }
    arrays = ({"w": w_np, "b": b_np, "h_train": h_train, "y_train": y_train}
              if keep_npz else {})

    iid_idx = int(iid_token.split("_")[1])
    logger.info(f"{slug}: forward iid test ({iid_token})")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_iid = ev["encoded"].cpu().numpy().astype(np.float32)
    record["iid_test"] = dict(estimate_ood_coords(h_iid, fm),
                              n=int(len(h_iid)))
    if keep_npz:
        arrays["h_iid_test"] = h_iid
    del h_iid

    resize_img = ((384, 384) if study_name == "vit"
                  else (64, 64) if str(cf.data.dataset) == "tiny-imagenet-200"
                  else (32, 32))
    for cname, spec in plan.items():
        try:
            if spec["kind"] == "config":
                # Forward the configured loader DIRECTLY by index: immune to
                # compute_model_evaluations' custom branches at test_6+ and
                # to its index-<=5 restriction.
                idx = int(spec["token"].split("_")[1])
                if idx >= n_loaders:
                    raise IndexError(
                        f"{spec['token']} out of range: experiment exposes "
                        f"{n_loaders} configured loaders")
                logger.info(f"{slug}: forward {cname} ({spec['token']}, "
                            f"config loader)")
                ev = forward_loader(model, test_loaders[idx])
            elif spec["kind"] == "custom":
                logger.info(f"{slug}: forward {cname} ({spec['token']}, "
                            f"custom)")
                ev = utils.compute_model_evaluations(model, datamodule,
                                                     spec["token"])
            else:
                logger.info(f"{slug}: forward {cname} (torchvision fallback)")
                ev = forward_loader(model,
                                    fallback_loader(cname, datamodule,
                                                    resize_img))
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            record["ood"][cname] = {"error": str(err), **spec}
            continue
        h_ood = ev["encoded"].cpu().numpy().astype(np.float32)
        record["ood"][cname] = dict(estimate_ood_coords(h_ood, fm),
                                    n=int(len(h_ood)), loader=spec["kind"])
        if keep_npz:
            arrays[f"h_{cname.replace(' ', '_')}"] = h_ood
        del ev, h_ood

    got = [c for c, v in record["ood"].items() if "error" not in v]
    missing = [c for c in EXPECTED_SUITE[source] if c not in got]
    record["suite_complete"] = not missing
    if missing:
        logger.warning(f"{slug}: suite INCOMPLETE, missing {missing}")
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps(record, indent=1))
    failed_marker = out_dir / f"FAILED_{slug}.json"
    if failed_marker.exists():
        failed_marker.unlink()
    if keep_npz:
        np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    logger.info(f"{slug}: wrote coords JSON ({len(got)}/"
                f"{len(EXPECTED_SUITE[source])} suite sets, "
                f"{record['runtime_sec']}s)")
    del arrays, h_train, y_train, model, module, datamodule
    if use_cuda:
        torch.cuda.empty_cache()
    return record


def sweep_targets() -> tuple[list[str], list[dict], list[str]]:
    """Enumerate experiment dirs on disk, match against the frozen manifest.

    Returns (matched experiment paths, unmatched manifest cells, extra dirs).
    """
    import os

    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    manifest = json.loads(MANIFEST.read_text())["cells"]
    want = {(c["paradigm"], c["source"], c["run"], round(c["reward"], 4),
             int(c["dropout"])): c for c in manifest}
    matched: dict[tuple, str] = {}
    extra: list[str] = []
    for src_dir in sorted({c["src_dir"] for c in manifest}):
        study = root / f"{src_dir}_paper_sweep"
        if not study.is_dir():
            continue
        for p in sorted(study.iterdir()):
            rel = f"{src_dir}_paper_sweep/{p.name}"
            m = CNN_RE.match(rel)
            if not m:
                continue
            key = (m["paradigm"], SRC_KEY[m["src"]], int(m["run"]),
                   round(float(m["rew"]), 4), int(m["do"]))
            if key in want:
                matched[key] = rel
            else:
                extra.append(rel)
    missing = [c for k, c in want.items() if k not in matched]
    return sorted(matched.values()), missing, extra


def existing_schema(path: Path) -> int:
    try:
        return int(json.loads(path.read_text()).get("schema", 1))
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--source", type=str, default=None,
                        help="restrict the sweep to one ID source "
                             "(cifar10, cifar100, supercifar100/supercifar, "
                             "tinyimagenet/tiny-imagenet-200)")
    parser.add_argument("--shard", type=str, default="1/1",
                        help="k/n: run the k-th of n interleaved shards of "
                             "the (optionally source-filtered) target list; "
                             "use one shard per GPU/job, e.g. 1/2 and 2/2")
    parser.add_argument("--list", action="store_true",
                        help="with --sweep: enumerate and exit")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="forward-pass batch size (sets CSF_BATCH_SIZE; "
                             "default 128 from load_model)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="dataloader workers (sets CSF_NUM_WORKERS; "
                             "default 12)")
    parser.add_argument("--keep_npz", action="store_true")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    import os
    if args.batch_size is not None:
        os.environ["CSF_BATCH_SIZE"] = str(args.batch_size)
    if args.num_workers is not None:
        os.environ["CSF_NUM_WORKERS"] = str(args.num_workers)

    def resolve_cuda() -> bool:
        import torch
        return bool(args.use_cuda and torch.cuda.is_available())

    if args.model_path and not args.sweep:
        extract_one(args.model_path, out_dir, resolve_cuda(), args.keep_npz)
        return

    if not args.sweep:
        parser.error("pass --model_path or --sweep")
    targets, missing, extra = sweep_targets()
    src_label = "all sources"
    if args.source:
        alias = {"cifar10": "cifar10", "cifar100": "cifar100",
                 "supercifar100": "supercifar", "supercifar": "supercifar",
                 "tinyimagenet": "tiny-imagenet-200",
                 "tiny-imagenet-200": "tiny-imagenet-200"}
        token = alias.get(args.source)
        if token is None:
            parser.error(f"unknown --source {args.source!r}; use one of "
                         f"{sorted(set(alias))}")
        targets = [t for t in targets
                   if t.split("_paper_sweep/")[0] == token]
        missing = [c for c in missing if c["src_dir"] == token]
        src_label = f"source {args.source} ({token}_paper_sweep)"
    k, n = (int(x) for x in args.shard.split("/"))
    shard = targets[k - 1::n]
    print(f"[sweep] manifest 280; {src_label}: matched on disk "
          f"{len(targets)}; MISSING {len(missing)}; "
          f"unrelated dirs {len(extra)}; "
          f"shard {k}/{n} -> {len(shard)} checkpoints (schema {SCHEMA})")
    for c in missing:
        print(f"[sweep] MISSING from disk: {c}")
    if args.list:
        for t in shard:
            print(t)
        return
    use_cuda = resolve_cuda()
    done = skipped = failed = 0
    for i, rel in enumerate(shard, 1):
        slug = rel.replace("/", "__")
        out_json = out_dir / f"{slug}.json"
        if out_json.exists():
            if existing_schema(out_json) >= SCHEMA:
                skipped += 1
                continue
            stale = out_dir / f"{slug}.v1.json"
            out_json.rename(stale)
            print(f"[sweep] stale schema-1 output renamed to {stale.name}; "
                  f"re-extracting")
        print(f"[sweep {k}/{n}] {i}/{len(shard)}: {rel}")
        try:
            extract_one(rel, out_dir, use_cuda, args.keep_npz)
            done += 1
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"model_path": rel,
                 "traceback": traceback.format_exc()}, indent=1))
            print(f"[sweep] FAILED {rel} (recorded, continuing)")
    print(f"[sweep {k}/{n}] finished: {done} new, {skipped} skipped, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
