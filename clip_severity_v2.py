"""ICML campaign CLIP severity regeneration (v2; HPC side; frozen
protocol section 8.2 + section 9 KID/FD severity-axis amendment).

Produces the severity table of record for the ICML campaign: KID
(kid_mmd2, primary axis d^K) and FD (frechet_clip_distance, robustness
axis d^F) for every (source, eval set) pair the campaign touches:
- the 32 REGISTERED pool pairs (4 sources x 8 sets; gate A checks that
  these reproduce the frozen clip_severity.csv values);
- the 16 NEW-SHIFT pairs (4 sources x mnist/fashionmnist/kmnist/stl10,
  roster B);
- the ROSTER-A pairs: OpenOOD cifar10/cifar100 (6 OOD sets each, from
  the benchmark imglists) and OpenOOD imagenet200 (the 5 stage-3 sets).

FROZEN SPEC (mirrors the deployed clip_proximity.py pipeline exactly):
- Backbone: open_clip ViT-B-32 with the EXPLICIT pretrained tag
  laion2b_s34b_b79k (no automatic tag selection); features
  l2-normalized (extract_image_features l2_normalize=True); D = 512.
- ID reference = the TRAIN split of each source (deployed convention);
  eval sets = test/val splits and the folder sets via
  src.clip_utils.make_loader; OpenOOD sets via their imglists with the
  CLIP preprocess.
- KID point estimate: kid_mmd(X_id, X_ood, n_subsets=50,
  subset_size=1000) = unbiased MMD^2 under the deployed kernel
  k(u, v) = (1 + u'v/D)^3 (gamma=1/D, coef0=1, degree=3), seed 0.
- KID uncertainty (gate B): INDEPENDENT seed groups: the same
  50x1000 estimator re-run at seeds 1..8; kid_seed_std = ddof-1 std of
  the 8 independent means (the within-run subset std underestimates
  the sampling error because subsets overlap; recorded separately).
- FD: fid_from_features (Frechet distance on the same features), plus
  the EQUAL-SAMPLE-SIZE resampling check: both sides subsampled to
  m = min(n_id, n_ood) at seeds 201..203 (FD is sample-size biased;
  the check bounds that bias per pair).
- Embedding cache: one npz per dataset key under --cache_dir; a
  dataset is embedded ONCE and every pair reads the cache. Provenance
  (versions, tag, kernel, seeds, per-cache sha256, table sha256) goes
  to clip_severity_v2_provenance.json.
Per GR-5 nothing here reads any detector outcome; severity is
outcome-free metadata.

Usage (HPC, inside the container, from code/):
    python clip_severity_v2.py --self-test
    python clip_severity_v2.py \
        --data_root_dir $DATASET_ROOT_DIR \
        --openood_root  $DATASET_ROOT_DIR/openood/data \
        [--stages pool,new,roster_a] [--batch_size 256]
Outputs: pilot0/clip_severity_v2.csv (+ _provenance.json)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))

MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "laion2b_s34b_b79k"  # explicit; never auto-selected
KID_SUBSETS, KID_SUBSET_SIZE = 50, 1000
KID_POINT_SEED = 0
KID_GROUP_SEEDS = tuple(range(1, 9))
FD_EQN_SEEDS = (201, 202, 203)
CACHE_DIR_DEFAULT = "pilot0/clip_v2_cache"
OUT_CSV = Path("pilot0/clip_severity_v2.csv")
OUT_PROV = Path("pilot0/clip_severity_v2_provenance.json")

# Registered pool pairs: sources and eval sets EXACTLY as in the frozen
# documentation/x6_spectral_scripts/clip_severity.csv (gate A joins on
# these names).
REGISTERED = {
    "cifar10": ("cifar100", "tinyimagenet", "isun", "lsun resize",
                "lsun cropped", "svhn", "places365", "textures"),
    "supercifar100": ("cifar10", "tinyimagenet", "isun", "lsun resize",
                      "lsun cropped", "svhn", "places365", "textures"),
    "cifar100": ("cifar10", "tinyimagenet", "isun", "lsun resize",
                 "lsun cropped", "svhn", "places365", "textures"),
    "tinyimagenet": ("cifar100", "cifar10", "isun", "lsun resize",
                     "lsun cropped", "places365", "textures", "svhn"),
}
# frozen-name -> (embedding cache key, make_loader spec, split)
POOL_EVAL_KEYS = {
    "cifar10": ("eval_cifar10_test", "cifar10", "test"),
    "cifar100": ("eval_cifar100_test", "cifar100", "test"),
    "tinyimagenet": ("eval_tinyimagenet_val", "tinyimagenet", "test"),
    "svhn": ("eval_svhn_test", "svhn", "test"),
    "isun": ("eval_isun", "isun", None),
    "lsun resize": ("eval_lsun_resize", "lsun_resize", None),
    "lsun cropped": ("eval_lsun_cropped", "lsun_cropped", None),
    "places365": ("eval_places365", "places365", None),
    "textures": ("eval_textures", "textures", None),
}
POOL_ID_KEYS = {s: (f"id_{s}_train", s, "train") for s in REGISTERED}
# roster-B set name -> (cache key, torchvision dataset, subdir)
NEW_SETS = {
    "mnist_new": ("eval_mnist_test", "MNIST", "mnist_new"),
    "fashionmnist_new": ("eval_fashionmnist_test", "FashionMNIST",
                         "fmnist_new"),
    "kmnist_new": ("eval_kmnist_test", "KMNIST", "kmnist_new"),
    "stl10_new": ("eval_stl10_test", "STL10", "stl10_new"),
}
OPENOOD_CIFAR_SUITES = {
    "cifar10": ("cifar100", "tin", "mnist", "svhn", "texture",
                "places365"),
    "cifar100": ("cifar10", "tin", "mnist", "svhn", "texture",
                 "places365"),
}


# ---------------------------------------------------------------------------
# Pair metrics. The HPC run uses the DEPLOYED implementations in
# src.clip_utils (kid_mmd, fid_from_features). The mirrors below copy
# that arithmetic exactly for the local self-test (src.clip_utils hard-
# imports open_clip); when open_clip IS importable the self-test also
# asserts mirror == deployed on shared inputs.
# ---------------------------------------------------------------------------

def _kid_mmd_mirror(X, Y, degree=3, gamma=None, coef0=1.0, n_subsets=50,
                    subset_size=1000, seed=0):
    rng = np.random.RandomState(seed)
    n_x, n_y = len(X), len(Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    def poly_kernel(A, B):
        return (gamma * (A @ B.T) + coef0) ** degree

    vals = []
    for _ in range(n_subsets):
        idx_x = rng.choice(n_x, size=min(subset_size, n_x), replace=False)
        idx_y = rng.choice(n_y, size=min(subset_size, n_y), replace=False)
        Xs, Ys = X[idx_x], Y[idx_y]
        Kxx = poly_kernel(Xs, Xs)
        Kyy = poly_kernel(Ys, Ys)
        Kxy = poly_kernel(Xs, Ys)
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        m = Kxx.shape[0]
        n = Kyy.shape[0]
        vals.append((Kxx.sum() / (m * (m - 1)))
                    + (Kyy.sum() / (n * (n - 1))) - 2.0 * Kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def _fid_mirror(X, Y):
    from scipy.linalg import sqrtm
    mu1, mu2 = X.mean(axis=0), Y.mean(axis=0)
    S1 = np.cov(X, rowvar=False)
    S2 = np.cov(Y, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(S1 @ S2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(S1 + S2 - 2 * covmean))


def _metric_fns(allow_mirror: bool):
    try:
        from src.clip_utils import fid_from_features, kid_mmd
        return kid_mmd, fid_from_features, "src.clip_utils (deployed)"
    except ImportError:
        if not allow_mirror:
            raise
        print("[clipv2] WARNING: src.clip_utils not importable here; "
              "self-test uses the arithmetic mirrors", flush=True)
        return _kid_mmd_mirror, _fid_mirror, "inline mirror (self-test)"


def pair_metrics(X_id: np.ndarray, X_ood: np.ndarray,
                 allow_mirror: bool = False) -> dict:
    kid_mmd, fid_from_features, impl = _metric_fns(allow_mirror)
    kid, kid_sub_std = kid_mmd(X_id, X_ood, n_subsets=KID_SUBSETS,
                               subset_size=KID_SUBSET_SIZE,
                               seed=KID_POINT_SEED)
    group_means = [kid_mmd(X_id, X_ood, n_subsets=KID_SUBSETS,
                           subset_size=KID_SUBSET_SIZE, seed=s)[0]
                   for s in KID_GROUP_SEEDS]
    fd = fid_from_features(X_id, X_ood)
    m = min(len(X_id), len(X_ood))
    fd_eqn = []
    for s in FD_EQN_SEEDS:
        rng = np.random.RandomState(s)
        fd_eqn.append(fid_from_features(
            X_id[rng.choice(len(X_id), size=m, replace=False)],
            X_ood[rng.choice(len(X_ood), size=m, replace=False)]))
    return {"n_id": int(len(X_id)), "n_ood": int(len(X_ood)),
            "kid_mmd2": float(kid),
            "kid_subset_std": float(kid_sub_std),
            "kid_seed_means": [float(v) for v in group_means],
            "kid_seed_std": float(np.std(group_means, ddof=1)),
            "frechet_clip_distance": float(fd),
            "fd_eqn_m": int(m),
            "fd_eqn_mean": float(np.mean(fd_eqn)),
            "fd_eqn_std": float(np.std(fd_eqn, ddof=1)),
            "metric_impl": impl}


# ---------------------------------------------------------------------------
# Embedding cache.
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, cache_dir: Path, batch_size: int, num_workers: int,
                 data_root_dir: str | None, openood_root: str | None):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root_dir = data_root_dir
        self.openood_root = Path(openood_root) if openood_root else None
        self._model = None
        self.cache_meta: dict[str, dict] = {}

    def _clip(self):
        if self._model is None:
            from src.clip_utils import load_clip
            model, preprocess, _tok, device, backend = load_clip(
                model_name=MODEL_NAME, pretrained=PRETRAINED_TAG)
            assert backend == "open_clip", backend
            self._model = (model, preprocess, device, backend)
            print(f"[clipv2] loaded {MODEL_NAME}/{PRETRAINED_TAG} "
                  f"on {device}", flush=True)
        return self._model

    def _extract(self, loader) -> np.ndarray:
        from src.clip_utils import extract_image_features
        model, _pre, device, backend = self._clip()
        X = extract_image_features(model, loader, device, backend,
                                   l2_normalize=True)
        return np.asarray(X, dtype=np.float32)

    def get(self, key: str, build_loader) -> np.ndarray:
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            X = np.load(path)["feats"]
        else:
            t0 = time.time()
            X = self._extract(build_loader())
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, feats=X)
            print(f"[clipv2] embedded {key}: {X.shape} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        assert X.ndim == 2 and X.shape[1] == 512, (key, X.shape)
        self.cache_meta[key] = {
            "n": int(len(X)), "dim": int(X.shape[1]),
            "sha256": hashlib.sha256(X.tobytes()).hexdigest()}
        return X

    # -- loader builders -----------------------------------------------------
    def pool_loader(self, spec: str, split: str | None):
        from src.clip_utils import make_loader
        _model, preprocess, _dev, _b = self._clip()
        kwargs = dict(preprocess=preprocess, batch_size=self.batch_size,
                      num_workers=self.num_workers)
        if self.data_root_dir:
            kwargs["data_root_dir"] = Path(self.data_root_dir)
        out = (make_loader(spec, split=split, **kwargs) if split
               else make_loader(spec, **kwargs))
        return out[0] if isinstance(out, tuple) else out

    def new_set_loader(self, tv_name: str, subdir: str):
        import torchvision
        from torch.utils.data import DataLoader
        _model, preprocess, _dev, _b = self._clip()
        root = (Path(self.data_root_dir or ".") / subdir)
        cls = getattr(torchvision.datasets, tv_name)
        if tv_name == "STL10":
            ds = cls(str(root), split="test", download=True,
                     transform=preprocess)
        else:
            ds = cls(str(root), train=False, download=True,
                     transform=preprocess)
        return DataLoader(ds, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def imglist_loader(self, imglist_rel: str, images_sub: str):
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        _model, preprocess, _dev, _b = self._clip()
        assert self.openood_root is not None, "--openood_root required"
        data_root = self.openood_root

        class ImglistDataset(Dataset):
            def __init__(self):
                self.items = []
                root = data_root / images_sub
                for line in ((data_root / imglist_rel)
                             .read_text().splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    rel, _label = line.rsplit(" ", 1)
                    self.items.append(root / rel)

            def __len__(self):
                return len(self.items)

            def __getitem__(self, i):
                img = Image.open(self.items[i]).convert("RGB")
                return preprocess(img), 0

        return DataLoader(ImglistDataset(), batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)


# ---------------------------------------------------------------------------
# Preflight: report EVERY missing input up front, before any embedding.
# ---------------------------------------------------------------------------

def preflight(args, stages: list[str]) -> None:
    from pilot0.extract_stage3_imagenet200 import (ID_LIST, OOD_LISTS,
                                                   TRAIN_LIST)
    missing: list[str] = []

    def check_imglist(root: Path, rel: str, images_sub: str) -> None:
        path = root / rel
        if not path.is_file():
            missing.append(f"imglist {rel}")
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            img = line.rsplit(" ", 1)[0]
            if not (root / images_sub / img).is_file():
                missing.append(f"images for {rel} "
                               f"(first ref: {images_sub}/{img})")
            return

    if "pool" in stages:
        default_root = Path("/work/cniel/sw/FD_Shifts/project/datasets")
        root = Path(args.data_root_dir) if args.data_root_dir \
            else default_root
        for sub in ("LSUN_resize", "LSUN", "iSUN", "dtd/images",
                    "places365"):
            if not (root / sub).is_dir():
                missing.append(f"pool folder set {root / sub}")
    if "roster_a" in stages:
        oo = Path(args.openood_root)
        for source, suite in OPENOOD_CIFAR_SUITES.items():
            il = f"benchmark_imglist/{source}"
            check_imglist(oo, f"{il}/train_{source}.txt",
                          "images_classic")
            check_imglist(oo, f"{il}/test_{source}.txt", "images_classic")
            for name in suite:
                check_imglist(oo, f"{il}/test_{name}.txt",
                              "images_classic")
        check_imglist(oo, TRAIN_LIST[0], TRAIN_LIST[1])
        check_imglist(oo, ID_LIST[0], ID_LIST[1])
        for rel, sub, _tier in OOD_LISTS.values():
            check_imglist(oo, rel, sub)
    if missing:
        print("[clipv2] PREFLIGHT FAILED; missing inputs:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        raise SystemExit(
            "[clipv2] download the missing data first (CIFAR-side: "
            "bash pilot0/icml_download_openood_cifar.sh; ImageNet-side: "
            "bash pilot0/stage3_download_openood.sh), then rerun; "
            "cached embeddings are kept and skipped.")
    print("[clipv2] preflight OK: every imglist, first image, and "
          "folder set present", flush=True)


# ---------------------------------------------------------------------------
# Pair enumeration and the run.
# ---------------------------------------------------------------------------

def run(args) -> None:
    from pilot0.extract_stage3_imagenet200 import (ID_LIST, OOD_LISTS,
                                                   TRAIN_LIST)
    stages = args.stages.split(",")
    preflight(args, stages)
    emb = Embedder(Path(args.cache_dir), args.batch_size,
                   args.num_workers, args.data_root_dir,
                   args.openood_root)
    rows: list[dict] = []

    def add_row(roster, source, eval_name, eval_key, X_id, X_ood):
        print(f"[clipv2] pair {source} -> {eval_name}", flush=True)
        rows.append(dict(roster=roster, source=source,
                         eval_dataset=eval_name, eval_key=eval_key,
                         **pair_metrics(X_id, X_ood)))

    if "pool" in stages or "new" in stages:
        for source, evals in REGISTERED.items():
            ikey, ispec, isplit = POOL_ID_KEYS[source]
            X_id = emb.get(ikey, lambda s=ispec, sp=isplit:
                           emb.pool_loader(s, sp))
            if "pool" in stages:
                for ev in evals:
                    ckey, spec, split = POOL_EVAL_KEYS[ev]
                    X_ood = emb.get(ckey, lambda s=spec, sp=split:
                                    emb.pool_loader(s, sp))
                    add_row("registered", source, ev, ev, X_id, X_ood)
            if "new" in stages:
                for set_name, (ckey, tv, sub) in NEW_SETS.items():
                    X_ood = emb.get(
                        ckey,
                        lambda t=tv, sd=sub: emb.new_set_loader(t, sd))
                    add_row("new_shifts", source, set_name, set_name,
                            X_id, X_ood)

    if "roster_a" in stages:
        for source, suite in OPENOOD_CIFAR_SUITES.items():
            il = f"benchmark_imglist/{source}"
            X_id = emb.get(
                f"oo_{source}_train",
                lambda r=f"{il}/train_{source}.txt":
                    emb.imglist_loader(r, "images_classic"))
            for name in suite:
                X_ood = emb.get(
                    f"oo_{source}_{name}",
                    lambda r=f"{il}/test_{name}.txt":
                        emb.imglist_loader(r, "images_classic"))
                add_row("roster_a", source, name, name, X_id, X_ood)
        X_id = emb.get("oo_imagenet200_train",
                       lambda: emb.imglist_loader(*TRAIN_LIST))
        emb.get("oo_imagenet200_test",
                lambda: emb.imglist_loader(*ID_LIST))
        for name, (rel, sub, _tier) in OOD_LISTS.items():
            X_ood = emb.get(f"oo_in200_{name}",
                            lambda r=rel, s=sub: emb.imglist_loader(r, s))
            add_row("roster_a", "imagenet200", name, name, X_id, X_ood)

    import pandas as pd
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    table_sha = hashlib.sha256(OUT_CSV.read_bytes()).hexdigest()

    import open_clip
    import torch
    prov = {"plan": "ICML retarget protocol section 9 (KID/FD severity "
                    "amendment): CLIP severity regeneration v2",
            "model": MODEL_NAME, "pretrained_tag": PRETRAINED_TAG,
            "tag_selection": "explicit (no automatic selection)",
            "feature_dim": 512, "l2_normalize": True,
            "id_reference": "train split of each source (deployed "
                            "clip_proximity.py convention)",
            "kernel": "k(u,v) = (1 + u'v/512)^3 (gamma=1/D, coef0=1, "
                      "degree=3), unbiased MMD^2",
            "kid_subsets": [KID_SUBSETS, KID_SUBSET_SIZE],
            "kid_point_seed": KID_POINT_SEED,
            "kid_group_seeds": list(KID_GROUP_SEEDS),
            "fd_eqn_seeds": list(FD_EQN_SEEDS),
            "versions": {"open_clip": open_clip.__version__,
                         "torch": torch.__version__,
                         "numpy": np.__version__},
            "stages": stages, "n_pairs": len(rows),
            "caches": emb.cache_meta, "table_sha256": table_sha}
    OUT_PROV.write_text(json.dumps(prov, indent=1))
    print(f"[clipv2] wrote {OUT_CSV} ({len(rows)} pairs); "
          f"sha256 {table_sha}", flush=True)


# ---------------------------------------------------------------------------
# Self-test (synthetic; no CLIP, no data).
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.RandomState(7)

    def sphere(n, shift=0.0):
        X = rng.randn(n, 512) + shift
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(
            np.float64)

    X_id = sphere(2500)
    same = sphere(2100)
    near = sphere(2100, shift=0.02)
    far = sphere(2100, shift=0.10)
    m_same = pair_metrics(X_id, same, allow_mirror=True)
    m_near = pair_metrics(X_id, near, allow_mirror=True)
    m_far = pair_metrics(X_id, far, allow_mirror=True)
    assert abs(m_same["kid_mmd2"]) < 5e-4, m_same["kid_mmd2"]
    assert m_far["kid_mmd2"] > m_near["kid_mmd2"] > 0, (
        m_near["kid_mmd2"], m_far["kid_mmd2"])
    assert m_far["frechet_clip_distance"] > m_near[
        "frechet_clip_distance"] > m_same["frechet_clip_distance"] >= 0
    assert m_far["kid_seed_std"] < 0.2 * m_far["kid_mmd2"], (
        "independent-seed spread should be small vs the far signal")
    assert len(m_far["kid_seed_means"]) == len(KID_GROUP_SEEDS)
    assert m_far["fd_eqn_m"] == 2100
    r1 = pair_metrics(X_id, far, allow_mirror=True)
    assert r1["kid_mmd2"] == m_far["kid_mmd2"], "not deterministic"
    try:
        from src.clip_utils import fid_from_features, kid_mmd
        k_dep = kid_mmd(X_id, far, n_subsets=5, subset_size=500, seed=3)
        k_mir = _kid_mmd_mirror(X_id, far, n_subsets=5, subset_size=500,
                                seed=3)
        assert np.allclose(k_dep, k_mir), (k_dep, k_mir)
        assert np.isclose(fid_from_features(X_id, far),
                          _fid_mirror(X_id, far))
        print("[clipv2] self-test: mirrors MATCH the deployed "
              "src.clip_utils implementations")
    except ImportError:
        print("[clipv2] self-test: src.clip_utils unavailable here; "
              "mirror-vs-deployed equality check runs on HPC")
    print(f"[clipv2] self-test PASS (impl: {m_far['metric_impl']}); "
          f"kid same/near/far = {m_same['kid_mmd2']:.2e} "
          f"{m_near['kid_mmd2']:.2e} {m_far['kid_mmd2']:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root_dir", type=str, default=None,
                    help="pool folder-set root (LSUN_resize, iSUN, dtd, "
                         "places365 live here); default = the "
                         "clip_utils HPC default")
    ap.add_argument("--openood_root", type=str, default=None,
                    help="OpenOOD data root (benchmark_imglist + "
                         "images_classic/images_largescale)")
    ap.add_argument("--cache_dir", type=str, default=CACHE_DIR_DEFAULT)
    ap.add_argument("--stages", type=str, default="pool,new,roster_a")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--preflight", action="store_true",
                    help="check inputs and exit (no CLIP load, no "
                         "embedding)")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if "roster_a" in args.stages:
        assert args.openood_root, "--openood_root required for roster_a"
    if args.preflight:
        preflight(args, args.stages.split(","))
        return
    run(args)


if __name__ == "__main__":
    main()
