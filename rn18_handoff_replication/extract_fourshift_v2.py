"""RN18 handoff-replication extractor, VERSION 2 (lossless evidence;
repair item P1-D, status-review finding F10). HPC, GPU.

Version 1 (`extract_fourshift_rn18.py`, frozen) stored rounded summaries
only (AUROC 4 decimals, AUGRC and gaps 5 decimals), no per-example
scores, no correctness arrays, no sample identities, no balancing
indices, no checkpoint identity, and float32 covariances. This version
re-extracts the same 96 + 20 checkpoints under the SAME frozen
measurement contract (deterministic train view, frozen loaders and
transforms, frozen score mirrors, frozen outcome helper with its
balancing seed 20260827, frozen coordinate and P10 code) and ADDS:

- per-example evidence in float64: the six scores and the logits on the
  ID test set and on every new set; ID labels and correctness; new-set
  labels;
- canonical sample identities: ID test = dataset test-split position
  after the config's slices (tenPercent pre-slice, then the devries
  1000-example validation slice), asserted against the split length and
  a label-sequence digest; new sets = torchvision test-split position
  with a label-sequence digest;
- the balancing indices actually drawn by the frozen outcome helper
  (same generator, same call order), so the balanced AUGRC is
  reproducible from the arrays;
- unrounded outcomes beside the frozen rounded ones (asserted equal
  after rounding), the AUGRC identity residual per score
  (AUGRC = pi^2/2 + pi (1 - pi)(1 - AUROC_f), midrank failure-AUROC;
  asserted <= 1e-10) and the exact failure-AUROC decomposition of the
  Energy-minus-CTM gap;
- checkpoint identity: resolved checkpoint path, file sha256, a
  canonical digest of the loaded state dict (every tensor incl. BN
  buffers), epoch and global step if the checkpoint records them;
- training membership under the loaded config (sampler type, size,
  index digest; the deterministic view uses the configured membership);
- covariances in float64 (within-class, OOD global, OOD residual).

The version-1 JSON fields are reproduced unchanged (same helper) so the
two extractions can be compared field by field. Outputs go to a NEW
directory; version-1 outputs are never touched. Per the first-reader
rule the outputs stay UNREAD until the version-2 reader runs.

Usage (HPC, inside the container, from code/):
    python rn18_handoff_replication/extract_fourshift_v2.py --list
    python rn18_handoff_replication/extract_fourshift_v2.py [--shard k/n]
    python rn18_handoff_replication/extract_fourshift_v2.py --panel vgg_bridge
    python rn18_handoff_replication/extract_fourshift_v2.py --self-test   (local, no torch)
Output: rn18_handoff_replication/outputs/fourshift_v2_<panel>/<slug>.json + .npz
"""
from __future__ import annotations

import argparse
import hashlib
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

from pilot0.extract_stage2_expansion import BALANCE_SEED, MATERIALITY_AUGRC, SCORE_NAMES, failure_augrc, set_outcomes
from pilot0.scores import auroc

MANIFESTS = {"rn18": "rn18_handoff_replication/manifests/expected_panel.json",
             "vgg_bridge": "rn18_handoff_replication/manifests/vgg_bridge_panel.json"}
PREFIXES = ("", "fd-shifts/")
SCHEMA_V2 = 2
IDENTITY_TOL = 1e-10
DEVRIES_VAL_SLICE = 1000


# ---------------------------------------------------------------------------
# Pure-numpy evidence functions (self-testable locally).
# ---------------------------------------------------------------------------

def digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes() + str(a.dtype).encode() + str(a.shape).encode()).hexdigest()


def failure_auroc(conf: np.ndarray, res: np.ndarray) -> float:
    """P(confidence of a correct example > confidence of a failure), midrank ties."""
    return auroc(conf[res == 0], conf[res == 1])


def augrc_identity(conf: np.ndarray, res: np.ndarray) -> dict:
    pi = float(res.mean())
    a_f = failure_auroc(conf, res)
    closed = 0.5 * pi ** 2 + pi * (1 - pi) * (1 - a_f)
    rc = failure_augrc(conf, res)
    return {"augrc": rc, "augrc_closed_form": closed, "residual": rc - closed, "pi": pi, "failure_auroc": a_f}


def balancing_indices(n_id: int, n_ood: int) -> tuple[np.ndarray, np.ndarray, int]:
    """EXACTLY the draws of the frozen set_outcomes (same seed, same order)."""
    k = min(n_id, n_ood)
    rng = np.random.default_rng(BALANCE_SEED)
    id_idx = np.arange(n_id) if n_id == k else rng.choice(n_id, k, replace=False)
    ood_idx = np.arange(n_ood) if n_ood == k else rng.choice(n_ood, k, replace=False)
    return id_idx, ood_idx, k


def outcomes_v2(sc_id: dict, res_id: np.ndarray, sc_ood: dict) -> tuple[dict, dict, dict]:
    """Frozen rounded outcomes (identical helper) + unrounded outcomes,
    identity residuals, gap decomposition, and the balancing indices."""
    frozen = set_outcomes(sc_id, res_id, sc_ood)
    n_id, n_ood = len(res_id), len(next(iter(sc_ood.values())))
    id_idx, ood_idx, k = balancing_indices(n_id, n_ood)
    res_raw = np.concatenate([res_id, np.ones(n_ood)])
    res_bal = np.concatenate([res_id[id_idx], np.ones(k)])
    un = {"n_id": n_id, "n_ood": n_ood, "k_balanced": int(k), "pi_raw": float(res_raw.mean()), "pi_balanced": float(res_bal.mean()),
          "auroc_id_vs_ood": {}, "augrc_raw": {}, "augrc_balanced": {}, "failure_auroc_raw": {}, "failure_auroc_balanced": {},
          "identity_residual_raw": {}, "identity_residual_balanced": {}}
    for s in SCORE_NAMES:
        cid, cood = np.asarray(sc_id[s], float), np.asarray(sc_ood[s], float)
        un["auroc_id_vs_ood"][s] = auroc(cid, cood)
        r = augrc_identity(np.concatenate([cid, cood]), res_raw)
        b = augrc_identity(np.concatenate([cid[id_idx], cood[ood_idx]]), res_bal)
        un["augrc_raw"][s], un["augrc_balanced"][s] = r["augrc"], b["augrc"]
        un["failure_auroc_raw"][s], un["failure_auroc_balanced"][s] = r["failure_auroc"], b["failure_auroc"]
        un["identity_residual_raw"][s], un["identity_residual_balanced"][s] = r["residual"], b["residual"]
        assert abs(r["residual"]) <= IDENTITY_TOL and abs(b["residual"]) <= IDENTITY_TOL, (s, r["residual"], b["residual"])
        assert round(un["auroc_id_vs_ood"][s], 4) == frozen[f"auroc_id_vs_ood_{s}"], s
        assert round(r["augrc"], 5) == frozen[f"augrc_raw_{s}"] and round(b["augrc"], 5) == frozen[f"augrc_balanced_{s}"], s
    for tag, pi in (("raw", un["pi_raw"]), ("balanced", un["pi_balanced"])):
        gap = un[f"augrc_{tag}"]["Energy"] - un[f"augrc_{tag}"]["CTM"]
        dec = pi * (1 - pi) * (un[f"failure_auroc_{tag}"]["CTM"] - un[f"failure_auroc_{tag}"]["Energy"])
        un[f"gap_{tag}"] = gap
        un[f"gap_{tag}_decomposition"] = {"pi_times_1_minus_pi": pi * (1 - pi),
                                          "failure_auroc_gap_CTM_minus_Energy": un[f"failure_auroc_{tag}"]["CTM"] - un[f"failure_auroc_{tag}"]["Energy"],
                                          "product": dec, "residual": gap - dec}
        assert abs(gap - dec) <= IDENTITY_TOL, (tag, gap - dec)
    assert round(un["gap_raw"], 5) == frozen["gap_raw"] and round(un["gap_balanced"], 5) == frozen["gap_balanced"]
    un["material_unrounded"] = bool(abs(un["gap_balanced"]) >= MATERIALITY_AUGRC)
    un["material_frozen"] = frozen["material"]
    idx = {"id_idx": id_idx.astype(np.int64), "ood_idx": ood_idx.astype(np.int64)}
    return frozen, un, idx


def state_dict_digest(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k]
        a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def id_test_offset(cf, len_full: int) -> tuple[int, dict]:
    """Canonical position offset of the config's iid test set inside the
    dataset's test split (mirrors FDShiftsDataLoader.setup)."""
    pre = int(len_full * 0.1) if str(getattr(cf.test, "iid_set_split", "")) == "tenPercent" else 0
    dev = DEVRIES_VAL_SLICE if str(cf.trainer.val_split.name if hasattr(cf.trainer.val_split, "name") else cf.trainer.val_split) == "devries" else 0
    return pre + dev, {"tenPercent_pre_slice": pre, "devries_val_slice": dev, "test_split_length": int(len_full)}


# ---------------------------------------------------------------------------
# One checkpoint (container-only imports inside).
# ---------------------------------------------------------------------------

def extract_one(cell: dict, root: Path, out_dir: Path, use_cuda: bool) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.loaders.dataset_collection import get_dataset
    from fd_shifts.utils import exp_utils
    from src import utils
    from src.trained_module import TrainedModule
    from x6_spectral.measure_checkpoint import load_model
    from pilot0.extract_pool_coords import SRC_KEY, build_ood_plan, forward_loader
    from pilot0.extract_roster_b_newshifts import NEW_SETS, new_set_loader
    from pilot0.geometry import fit_feature_model, geometry_record, papyan_metrics
    from pilot0.ood_coords import estimate_ood_coords
    from pilot0.repair_stats import N_MIN, assign_components, compact_p10
    from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

    t0 = time.time()
    model_path = next((f"{p}{cell['model_path']}" for p in PREFIXES if (root / p / cell["model_path"]).is_dir()), None)
    assert model_path, f"not on disk: {cell['model_path']}"
    slug = cell["model_path"].replace("/", "__")
    source = SRC_KEY[cell["model_path"].split("_paper_sweep/")[0]]
    cf, module, study_name = load_model(model_path, use_cuda)
    # checkpoint identity (same resolution rule as load_model)
    ckpt_path = exp_utils._get_path_to_best_ckpt(cf.exp.dir, "last", cf.test.selection_mode)
    ck = torch.load(ckpt_path, map_location="cpu")
    ckpt_id = {"path": str(ckpt_path), "file_sha256": hashlib.sha256(Path(ckpt_path).read_bytes()).hexdigest(),
               "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
               "state_dict_digest_loaded_module": state_dict_digest(module.state_dict()),
               "state_dict_digest_checkpoint": state_dict_digest(ck["state_dict"]) if "state_dict" in ck else None,
               "n_tensors_loaded_module": len(module.state_dict())}
    del ck
    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False, rank_feat=False, ash_method=None, use_cuda=use_cuda)
    _, w, b = utils.get_model_and_last_layer(module, study_name)
    n_classes = int(cf.data.num_classes)
    w_np = w.detach().cpu().numpy().astype(np.float64)[:n_classes]
    b_np = b.detach().cpu().numpy().astype(np.float64)[:n_classes]
    iid_token, plan, _ = build_ood_plan(cf, source)
    test_loaders = datamodule.test_dataloader()

    # training membership under the loaded config; deterministic view on that membership
    ds_det = get_dataset(name=datamodule.dataset_name, root=datamodule.data_dir, train=True, download=True,
                         target_transform=datamodule.target_transforms.get("train"),
                         transform=datamodule.augmentations["test"], kwargs=datamodule.dataset_kwargs)
    sampler = getattr(datamodule, "train_sampler", None)
    if sampler is not None and hasattr(sampler, "indices"):
        tr_idx = np.asarray(list(sampler.indices), dtype=np.int64)
        ds_view = Subset(ds_det, tr_idx.tolist())
        membership = {"rule": f"configured sampler {type(sampler).__name__}", "n": int(len(tr_idx)), "indices_sha256": digest(tr_idx)}
    else:
        tr_idx = np.arange(len(ds_det), dtype=np.int64)
        ds_view = ds_det
        membership = {"rule": "full training split (train_sampler is None under this val_split)", "n": int(len(ds_det)), "indices_sha256": digest(tr_idx)}
    membership["val_split"] = str(getattr(cf.trainer.val_split, "name", cf.trainer.val_split))
    logger.info(f"{slug}: forward train (deterministic view, {membership['rule']})")
    ev = forward_loader(model, DataLoader(ds_view, batch_size=datamodule.batch_size, shuffle=False,
                                          num_workers=datamodule.num_workers, pin_memory=datamodule.pin_memory))
    h_tr = ev["encoded"].cpu().numpy().astype(np.float32)
    y_tr = ev["labels"].cpu().numpy().astype(np.int64)
    membership["label_sequence_sha256"] = digest(y_tr)
    fm = fit_feature_model(h_tr, y_tr, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_tr.astype(np.float64), y_tr, n_classes)
    train_mean = fm.global_mean
    del h_tr, y_tr, ev

    def scores_for(h):
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"], "MLS": hs_["MLS"], "CTM": ctm(h64, proto_unc),
                "Maha": maha(h64), "fDBD": fdbd(h64, g, w_np, train_mean), "_logits": g}

    rec = {"schema_fourshift": 1, "schema_fourshift_v2": SCHEMA_V2, **cell, "resolved_model_path": model_path, "slug": slug,
           "source": source, "study": study_name, "n_classes": n_classes, "dim": int(fm.global_mean.shape[0]), "view": "deterministic",
           "geometry": geometry_record(w_np, b_np, fm), "papyan": papyan_metrics(w_np, fm), "ood": {},
           "v2": {"checkpoint": ckpt_id, "train_membership": membership, "identity_tolerance": IDENTITY_TOL,
                  "balance_seed": BALANCE_SEED, "ood_unrounded": {}}}
    arrays = {"w": w_np, "b": b_np, "proto_unc": proto_unc, "global_mean": fm.global_mean,
              "class_means_centered": fm.class_means, "sigma_w": fm.sigma_w.astype(np.float64), "train_indices": tr_idx}

    # ID test: canonical identities
    iid_idx = int(iid_token.split("_")[1])
    ds_full = get_dataset(name=datamodule.dataset_name, root=datamodule.data_dir, train=False, download=True,
                          target_transform=None, transform=None, kwargs=datamodule.dataset_kwargs)
    len_full = len(ds_full)
    offset, slices = id_test_offset(cf, len_full)
    logger.info(f"{slug}: forward iid test ({iid_token}, offset {offset})")
    ev = forward_loader(model, test_loaders[iid_idx])
    h_id = ev["encoded"].cpu().numpy().astype(np.float32)
    y_id = ev["labels"].cpu().numpy().astype(np.int64)
    assert len(h_id) == len_full - offset, (len(h_id), len_full, offset)
    full_targets = np.asarray(getattr(ds_full, "targets", getattr(ds_full, "labels", None)), dtype=np.int64)
    canon_ok = bool(full_targets is not None and len(full_targets) == len_full and np.array_equal(full_targets[offset:], y_id))
    sc_id = scores_for(h_id)
    logits_id = sc_id.pop("_logits")
    res_id = (logits_id.argmax(1) != y_id).astype(float)
    rec["iid_test"] = dict(estimate_ood_coords(h_id, fm), n=int(len(h_id)), id_error_rate=float(res_id.mean()),
                           label_counts=np.bincount(y_id, minlength=n_classes).tolist())
    rec["v2"]["id_test"] = {"canonical_ids": f"test-split position offset {offset} + row", **slices, "n": int(len(h_id)),
                            "label_sequence_sha256": digest(y_id), "order_matches_test_split": canon_ok,
                            "id_error_rate_unrounded": float(res_id.mean())}
    arrays.update({"id__labels": y_id, "id__correct": (1 - res_id).astype(np.int8), "id__logits": logits_id.astype(np.float64),
                   "id__ids": (offset + np.arange(len(h_id))).astype(np.int64)})
    for s in SCORE_NAMES:
        arrays[f"id__score__{s}"] = np.asarray(sc_id[s], np.float64)
    del ev, h_id

    resize_img = (64, 64) if str(cf.data.dataset) == "tiny-imagenet-200" else (32, 32)
    for si, cname in enumerate(NEW_SETS, start=1):
        try:
            logger.info(f"{slug}: forward {cname}")
            loader = new_set_loader(cname, datamodule, resize_img)
            ev = forward_loader(model, loader)
            h_o = ev["encoded"].cpu().numpy().astype(np.float32)
            y_o = ev["labels"].cpu().numpy().astype(np.int64)
            sc_o = scores_for(h_o); logits_o = sc_o.pop("_logits")
            frozen, un, idx = outcomes_v2(sc_id, res_id, sc_o)
            # Gaussian inputs in float64
            H = h_o.astype(np.float64); n = len(H); mean = H.mean(0); Hc = H - mean
            cov_glob = Hc.T @ Hc / n
            hc = H - fm.global_mean; mu_hat = fm.class_means / fm.radii[:, None]
            labels = assign_components(hc, mu_hat); raw = np.bincount(labels, minlength=len(mu_hat))
            keep = np.where(raw >= N_MIN)[0]; comp = np.where(np.isin(labels, keep), labels, -1)
            ids = sorted(set(comp.tolist())); means = np.stack([H[comp == k].mean(0) for k in ids])
            counts = np.array([(comp == k).sum() for k in ids], dtype=float)
            R = H - means[[ids.index(k) for k in comp]]; cov_res = R.T @ R / n
            gscal = {"n": int(n), "component_ids": ids, "weights": (counts / n).tolist()}
            rec["ood"][cname] = dict(estimate_ood_coords(h_o, fm), p10=compact_p10(h_o, fm, w_np, set_index=si), gaussian=gscal, **frozen)
            rec["v2"]["ood_unrounded"][cname] = dict(un, dataset=type(loader.dataset).__name__, n=int(n),
                                                     canonical_ids="torchvision test-split position", label_sequence_sha256=digest(y_o))
            arrays[f"set__{cname}__ood_mean"] = mean; arrays[f"set__{cname}__cov_glob"] = cov_glob
            arrays[f"set__{cname}__comp_means"] = means; arrays[f"set__{cname}__cov_res"] = cov_res
            arrays[f"set__{cname}__component_labels"] = comp.astype(np.int64)
            arrays[f"set__{cname}__labels"] = y_o; arrays[f"set__{cname}__logits"] = logits_o.astype(np.float64)
            arrays[f"set__{cname}__balance_id_idx"] = idx["id_idx"]; arrays[f"set__{cname}__balance_ood_idx"] = idx["ood_idx"]
            for s in SCORE_NAMES:
                arrays[f"set__{cname}__score__{s}"] = np.asarray(sc_o[s], np.float64)
            del ev, h_o
        except Exception as err:  # noqa: BLE001 - per-set isolation
            logger.error(f"{slug}: {cname} FAILED: {err}")
            rec["ood"][cname] = {"error": str(err), "traceback": traceback.format_exc()}
    rec["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{slug}.npz", **arrays)
    rec["v2"]["npz_sha256"] = hashlib.sha256((out_dir / f"{slug}.npz").read_bytes()).hexdigest()
    rec["v2"]["npz_keys"] = sorted(arrays)
    (out_dir / f"{slug}.json").write_text(json.dumps(rec, indent=1, default=float))
    f = out_dir / f"FAILED_{slug}.json"
    if f.exists():
        f.unlink()
    logger.info(f"{slug}: wrote {len(rec['ood'])} sets ({rec['runtime_sec']}s)")
    del model, module, datamodule
    if use_cuda:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Local self-test (numpy only): identity, balancing draws, frozen agreement.
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(7)
    n_id, n_ood = 9000, 10000
    res_id = (rng.random(n_id) < 0.2).astype(float)
    sc_id = {s: rng.normal(1.0, 1.0, n_id) for s in SCORE_NAMES}
    sc_ood = {s: rng.normal(0.0, 1.0, n_ood) for s in SCORE_NAMES}
    sc_id["MLS"] = np.round(sc_id["MLS"], 1); sc_ood["MLS"] = np.round(sc_ood["MLS"], 1)      # ties
    frozen, un, idx = outcomes_v2(sc_id, res_id, sc_ood)
    assert idx["id_idx"].shape == (9000,) and idx["ood_idx"].shape == (9000,) and len(set(idx["ood_idx"])) == 9000
    id2, ood2, _ = balancing_indices(n_id, n_ood)
    assert np.array_equal(idx["ood_idx"], ood2), "balancing draw not reproducible"
    assert max(abs(v) for d in (un["identity_residual_raw"], un["identity_residual_balanced"]) for v in d.values()) <= IDENTITY_TOL
    assert abs(un["gap_balanced_decomposition"]["residual"]) <= IDENTITY_TOL
    assert round(un["gap_balanced"], 5) == frozen["gap_balanced"] and round(un["auroc_id_vs_ood"]["Energy"], 4) == frozen["auroc_id_vs_ood_Energy"]
    # frozen rounding loses information the unrounded record keeps
    assert un["gap_balanced"] != frozen["gap_balanced"] or abs(un["gap_balanced"] * 1e5 - round(un["gap_balanced"] * 1e5)) < 1e-9
    assert digest(np.arange(3)) != digest(np.arange(3, dtype=np.float64))
    print(f"[fourshift-v2] self-test PASS: identity residual max "
          f"{max(abs(v) for v in un['identity_residual_balanced'].values()):.1e}, balancing draws reproduced, "
          f"frozen rounded fields reproduced (gap_balanced {un['gap_balanced']:+.7f} -> {frozen['gap_balanced']:+.5f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", choices=list(MANIFESTS), default="rn18")
    ap.add_argument("--shard", type=str, default="1/1")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    man = json.loads(Path(MANIFESTS[args.panel]).read_text())
    cells = [c for c in man["cells"] if c.get("mechanically_eligible", True)]
    k, n = (int(x) for x in args.shard.split("/"))
    cells = cells[k - 1::n]
    out_dir = Path(f"rn18_handoff_replication/outputs/fourshift_v2_{args.panel}")
    todo = [c for c in cells if not (out_dir / f"{c['model_path'].replace('/', '__')}.json").exists()]
    print(f"[fourshift-v2/{args.panel}] shard {args.shard}: {len(cells)} eligible cells, {len(todo)} to run", flush=True)
    if args.list:
        for c in todo[:12]:
            print("  ", c["model_path"])
        return
    failures = 0
    for i, c in enumerate(todo, 1):
        print(f"[fourshift-v2] {i}/{len(todo)}: {c['model_path']}", flush=True)
        try:
            extract_one(c, root, out_dir, args.use_cuda)
        except Exception:  # noqa: BLE001
            failures += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{c['model_path'].replace('/', '__')}.json").write_text(
                json.dumps({"model_path": c["model_path"], "error": traceback.format_exc()}, indent=1))
            print(f"[fourshift-v2] FAILED {c['model_path']}", flush=True)
    print(f"[fourshift-v2] done: {len(todo) - failures} ok, {failures} failed", flush=True)


if __name__ == "__main__":
    main()
