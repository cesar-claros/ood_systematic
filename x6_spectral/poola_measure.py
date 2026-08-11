"""Pool A pristine-tier measurement (X6 campaign; predictions before outcomes).

Runs the frozen r8 protocol on the X8 Pool A cached features
({encoder}_{dataset}_{split}.npz with `features`/`labels`): per cell
(encoder, source, probe-train size, seed), fit the pool's linear probe,
define the deployed ProjectionFiltering for this pool by the frozen recipe
(global and per-class PCA on probe-train correct-only features, per-class
centering, variance threshold 0.90, deployed component-count rule,
zero-correct fallback), run Tier-A diagnostics (rule r2), and run the r8
deployed-stack trial (out-of-sample ID reference: refit statistics on one
half of the validation carve-out, ID blocks from the other). Evaluation
sets mirror the gate-1 per-source OOD lists (cross-source test features
for the near sets plus the six far-OOD sets).

Feature-space only: no model forwards; minutes per cell on CPU. Writes
<out_dir>/poola/{encoder}__{source}__n{n}__s{seed}.json. Predictions lock
when these JSONs are committed to the record; only then may
poola_outcomes.py be run (it computes the pool's projection-variant AUGRC
tables for the first time).

Usage (HPC, from code/):
    python x6_spectral/poola_measure.py --features-dir $EXPERIMENT_ROOT_DIR/pool_a/features
    python x6_spectral/poola_measure.py --synthetic     # local self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "x8_pool_a"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import projection_filtering_analysis as pfa
from pool_a_csfs import train_probe
from spectra_campaign_harness import (deployed_pf_rank, deployed_trial,
                                      make_backprojector, measure, tier_a)


def jsonable(obj):
    """Recursively convert numpy objects for json.dump (as in stage 1)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def prune_census(diag: dict) -> dict:
    """Drop the full eigenvalue vector from the JSON record."""
    out = dict(diag)
    census = dict(out["census"])
    census.pop("eigs", None)
    out["census"] = census
    return out

ENCODERS = ["dinov2_vitb14", "clip_vitb16", "clip_vitl14"]
ID_SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
N_PER_CLASS = [25, 100, 0]  # 0 = full probe-train
SEEDS = [0, 1, 2]
VAL_MAX = 5000
BATCH_PER_DRAW = 128
N_DRAWS = 5
VARIANCE_EXPLAINED = 0.90
K_CLASS = 12
#: gate-1 OOD label -> Pool A npz dataset name (cross-source near sets use
#: the other ID sources' test features, exactly mirroring the fd-shifts
#: new-class studies; far sets are Pool A's own extractions).
NPZ_NAME = {"lsun resize": "lsun_resize", "lsun cropped": "lsun_cropped",
            "isun": "isun", "textures": "textures", "places365": "places365",
            "svhn": "svhn", "cifar10": "cifar10", "cifar100": "cifar100",
            "tinyimagenet": "tinyimagenet"}


def load_npz(features_dir: Path, encoder: str, dataset: str,
             split: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = features_dir / f"{encoder}_{dataset}_{split}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    labels = data["labels"] if "labels" in data else np.zeros(
        len(data["features"]), dtype=np.int64)
    return data["features"].astype(np.float64), labels.astype(np.int64)


def fold_head(probe: dict) -> tuple[np.ndarray, np.ndarray]:
    """Fold the probe's standardization into an affine head on raw features."""
    w = probe["W"].cpu().numpy().astype(np.float64)
    b = probe["b"].cpu().numpy().astype(np.float64)
    mu = probe["mu"].cpu().numpy().astype(np.float64)
    sd = probe["sd"].cpu().numpy().astype(np.float64)
    w_eff = w / sd[None, :]
    b_eff = b - (w * (mu / sd)[None, :]).sum(1)
    return w_eff, b_eff


def svd_pf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Deployed-recipe PCA: mean, components, component count at 0.90."""
    mean = x.mean(0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    ratio = s ** 2 / max((s ** 2).sum(), 1e-30)
    return mean, vt, deployed_pf_rank(ratio, VARIANCE_EXPLAINED)


def refit_maha(z: np.ndarray, labels: np.ndarray, n_classes: int,
               fallback_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([z[labels == c].mean(0) if (labels == c).any()
                      else fallback_mean for c in range(n_classes)])
    centered = np.concatenate([z[labels == c] - means[c]
                               for c in range(n_classes)
                               if (labels == c).any()])
    cov = centered.T @ centered / max(len(centered), 1)
    return means, np.linalg.pinv(cov, hermitian=True, rcond=1e-6)


def measure_cell(features_dir: Path, out_dir: Path, encoder: str,
                 source: str, n_pc: int, seed: int) -> str:
    slug = f"{encoder}__{source}__n{n_pc}__s{seed}"
    out_path = out_dir / f"{slug}.json"
    if out_path.exists():
        return "skip"
    loaded = load_npz(features_dir, encoder, source, "train")
    if loaded is None:
        return "missing-features"
    h_train, y_train = loaded
    n_classes = int(y_train.max()) + 1
    rng = np.random.default_rng(1000 * seed + 7)

    perm = rng.permutation(len(h_train))
    n_val = min(VAL_MAX, len(h_train) // 4)
    val_idx, rest = perm[:n_val], perm[n_val:]
    if n_pc > 0:
        sub_idx = np.concatenate([
            rng.choice(rest[y_train[rest] == c],
                       size=min(n_pc, (y_train[rest] == c).sum()),
                       replace=False) for c in range(n_classes)])
    else:
        sub_idx = rest
    h_sub, y_sub = h_train[sub_idx], y_train[sub_idx]
    h_val, y_val = h_train[val_idx], y_train[val_idx]

    t0 = time.time()
    probe = train_probe(torch.from_numpy(h_sub).float(),
                        torch.from_numpy(y_sub), n_classes, seed=seed)
    w_eff, b_eff = fold_head(probe)
    val_acc = float(((h_val @ w_eff.T + b_eff).argmax(1) == y_val).mean())

    preds_sub = (h_sub @ w_eff.T + b_eff).argmax(1)
    correct = preds_sub == y_sub
    h_cor, y_cor = h_sub[correct], y_sub[correct]
    if len(h_cor) < n_classes + 2:
        h_cor, y_cor = h_sub, y_sub

    g_mean, g_comps, n_g = svd_pf(h_cor)
    bp_global = make_backprojector(g_mean, g_comps, n_g)
    class_bp = []
    for c in range(n_classes):
        block = h_cor[y_cor == c]
        if len(block) < 3:
            block = h_sub[y_sub == c]
        if len(block) < 3:
            class_bp.append((g_mean, g_comps, n_g))
            continue
        class_bp.append(svd_pf(block))

    diag_correct = measure(h_cor, y_cor, w_eff, n_classes,
                           k_class=min(K_CLASS, max(2, len(h_cor)
                                                    // (4 * n_classes))))
    tier_a_out = tier_a(diag_correct, id_val_accuracy=val_acc)
    diag_all = measure(h_sub, y_sub, w_eff, n_classes,
                       k_class=min(K_CLASS, max(2, len(h_sub)
                                                // (4 * n_classes))))

    split_at = len(h_val) // 2
    ref_feats, ref_y = h_val[:split_at], y_val[:split_at]
    fit_feats, fit_y = h_val[split_at:], y_val[split_at:]
    correct_ref = (ref_feats @ w_eff.T + b_eff).argmax(1) == ref_y
    n_blocks = max(1, min(N_DRAWS, len(ref_feats) // BATCH_PER_DRAW))
    id_blocks = [ref_feats[d * BATCH_PER_DRAW:(d + 1) * BATCH_PER_DRAW]
                 for d in range(n_blocks)]
    id_fail = [~correct_ref[d * BATCH_PER_DRAW:(d + 1) * BATCH_PER_DRAW]
               for d in range(n_blocks)]
    z_g_fit = bp_global(fit_feats)
    preds_fit = (fit_feats @ w_eff.T + b_eff).argmax(1)
    z_cp_fit = np.empty_like(fit_feats)
    for c, (mean_c, comps_c, n_c) in enumerate(class_bp):
        mask = preds_fit == c
        if mask.any():
            z_cp_fit[mask] = make_backprojector(mean_c, comps_c,
                                                n_c)(fit_feats[mask])
    maha_sets = {"raw": refit_maha(fit_feats, fit_y, n_classes, g_mean),
                 "global": refit_maha(z_g_fit, fit_y, n_classes, g_mean),
                 "cp": refit_maha(z_cp_fit, fit_y, n_classes, g_mean)}

    record = {
        "cell": {"encoder": encoder, "source": source, "n_per_class": n_pc,
                 "seed": seed},
        "n_classes": n_classes, "dim": int(h_train.shape[1]),
        "n_probe_train": int(len(h_sub)),
        "n_probe_correct": int(correct.sum()),
        "probe_train_acc": float(correct.mean()),
        "probe_val_acc": val_acc, "n_global": int(n_g),
        "class_ranks": [n for _, _, n in class_bp],
        "arms": {"correct_only": prune_census(diag_correct),
                 "all": prune_census(diag_all)},
        "tier_a": tier_a_out,
        "r8_split": {"n_ref": int(split_at),
                     "n_fit": int(len(h_val) - split_at)},
        "datasets": {},
    }

    for ood in pfa.OOD_DATASETS[source]:
        loaded = load_npz(features_dir, encoder, NPZ_NAME[ood], "test")
        if loaded is None:
            record["datasets"][ood] = {"skipped": "features missing"}
            continue
        feats = loaded[0][:BATCH_PER_DRAW * N_DRAWS]
        if len(feats) < BATCH_PER_DRAW // 2:
            record["datasets"][ood] = {"skipped": f"only {len(feats)}"}
            continue
        n_draws = max(1, len(feats) // BATCH_PER_DRAW)
        draws = []
        for d in range(n_draws):
            block = feats[d * BATCH_PER_DRAW:(d + 1) * BATCH_PER_DRAW]
            draws.append(deployed_trial(id_blocks[d % n_blocks],
                                        id_fail[d % n_blocks], block,
                                        w_eff, b_eff, bp_global, class_bp,
                                        maha_sets))
        record["datasets"][ood] = {"summary": {
            "n_draws": n_draws,
            "trial_deployed_mean": {k: float(np.mean([dr[k] for dr in draws]))
                                    for k in draws[0]},
        }, "draws": draws}
    record["runtime_sec"] = round(time.time() - t0, 1)
    with open(out_path, "w") as fh:
        json.dump(jsonable(record), fh, indent=1)
    return "measured"


def make_synthetic(tmp: Path) -> None:
    """Tiny feature files exercising every code path (local self-test)."""
    rng = np.random.default_rng(0)
    dim, n_cls = 64, 6
    base = np.linalg.qr(rng.standard_normal((dim, n_cls + 4)))[0]
    mu = (np.eye(n_cls) - 1 / n_cls) @ base[:, :n_cls].T * 5.0
    for name in ID_SOURCES + list({NPZ_NAME[o] for src in ID_SOURCES
                                   for o in pfa.OOD_DATASETS[src]}):
        for split in ("train", "test"):
            n = 3000 if split == "train" else 800
            if name in ID_SOURCES:
                y = rng.integers(0, n_cls, n)
                h = mu[y] + rng.standard_normal((n, dim))
            else:
                y = np.zeros(n, dtype=np.int64)
                h = 3.0 * base[:, n_cls] + rng.standard_normal((n, dim))
            np.savez(tmp / f"synthetic_{name}_{split}.npz",
                     features=h.astype(np.float32), labels=y)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X6 Pool A pristine-tier measurement")
    parser.add_argument("--features-dir", type=str, default="pool_a_features")
    parser.add_argument("--out_dir", type=str, default="x6_spectral/outputs")
    parser.add_argument("--encoder", type=str, default=None,
                        choices=ENCODERS)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="poola_syn_"))
        make_synthetic(tmp)
        out = tmp / "out"
        out.mkdir()
        status = measure_cell(tmp, out, "synthetic", "cifar10", 25, 0)
        rec = json.load(open(out / "synthetic__cifar10__n25__s0.json"))
        print(f"synthetic cell: {status}; probe val acc "
              f"{rec['probe_val_acc']:.2f}; tier_a global = "
              f"{rec['tier_a']['global']['prediction']}; datasets measured "
              f"{sum('summary' in v for v in rec['datasets'].values())}; "
              f"sample trial keys {len(next(iter(rec['datasets'].values()))['summary']['trial_deployed_mean'])}")
        return

    features_dir = Path(args.features_dir)
    out_dir = Path(args.out_dir) / "poola"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoders = [args.encoder] if args.encoder else ENCODERS
    counts: dict[str, int] = {}
    for encoder in encoders:
        for source in ID_SOURCES:
            for n_pc in N_PER_CLASS:
                for seed in SEEDS:
                    status = measure_cell(features_dir, out_dir, encoder,
                                          source, n_pc, seed)
                    counts[status] = counts.get(status, 0) + 1
                    print(f"[{status:16s}] {encoder} {source} n{n_pc} "
                          f"s{seed}")
    print("done:", counts)


if __name__ == "__main__":
    main()
