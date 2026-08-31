"""ICML roster A extraction: OpenOOD CIFAR-10 / CIFAR-100 ResNet-18 base
CE checkpoints (HPC side; frozen protocol section 8.2 + KID/FD
amendment).

MANIFEST GATE: refuses to run unless pilot0/icml_roster_a_manifest.json
exists (the committed metadata-only enumeration). Per GR-5 the outputs
of this extractor stay UNREAD until the committed analysis suite runs.

FROZEN CONVENTIONS (mirroring the stage-2/3 expansion extractors):
- Model: torchvision resnet18(num_classes=C) with the OpenOOD 32x32
  stem (conv1 -> 3x3 stride 1 pad 1, maxpool -> Identity); the state
  dict must load with strict=True (any mismatch fails the checkpoint).
  Penultimate feature = avgpool flatten (512-d); logits recomputed as
  h @ W' + b from the stored fc (sliced-head convention).
- Preprocessing (OpenOOD cifar test convention, declared): Resize 32,
  CenterCrop 32, ToTensor, per-source normalization: CIFAR-10
  (0.4914, 0.4822, 0.4465)/(0.2470, 0.2435, 0.2616); CIFAR-100
  (0.5071, 0.4867, 0.4408)/(0.2675, 0.2565, 0.2761). The train split is
  forwarded with the same deterministic transform (expansion
  convention: the feature model fits unaugmented activations).
- Suites (OpenOOD v1.5 benchmark imglists): cifar10 -> ID
  test_cifar10; OOD cifar100, tin, mnist, svhn, texture, places365.
  cifar100 -> ID test_cifar100; OOD cifar10, tin, mnist, svhn, texture,
  places365. Missing imglists or images are reported per set, never
  silently skipped.
- Scores: the frozen feature-level mirrors (Energy/CTM claim-bearing;
  MSR/MLS/Maha/fDBD secondary); outcomes via the frozen set_outcomes
  (prevalence-balanced materiality, rng 20260827).
- Geometry: fit_feature_model on train; papyan_metrics (the paper's
  stated NC1) + geometry_record; per-set estimate_ood_coords.
- P10 inputs (endpoint E3): per OOD set the compact mixture block from
  the FROZEN repair_stats rules (compact_p10: nearest-prototype
  components, N_MIN = 25 merge-to-other, shared residual rho); no
  outcome enters its computation.
Resumable; per-checkpoint FAILED_ isolation.

Usage (HPC, inside the container, from code/):
    python icml_extract_roster_a_cifar.py --source cifar10 \
        --data_root $DATASET_ROOT_DIR/openood/data [--list]
    python icml_extract_roster_a_cifar.py --source cifar100 ...
Output: pilot0/icml_roster_a_coords/<slug>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))

from pilot0.extract_stage2_expansion import SCORE_NAMES, set_outcomes
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.repair_stats import compact_p10
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

MANIFEST = Path("pilot0/icml_roster_a_manifest.json")
OUT_DIR_DEFAULT = "pilot0/icml_roster_a_coords"
NORMS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))}
SUITES = {"cifar10": ("cifar100", "tin", "mnist", "svhn", "texture",
                      "places365"),
          "cifar100": ("cifar10", "tin", "mnist", "svhn", "texture",
                       "places365")}
N_CLASSES = {"cifar10": 10, "cifar100": 100}


def build_model(ckpt: Path, n_classes: int, use_cuda: bool):
    import torch
    from torchvision.models import resnet18

    net = resnet18(num_classes=n_classes)
    net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
    net.maxpool = torch.nn.Identity()
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = { (k[7:] if k.startswith("module.") else k): v
              for k, v in state.items() }
    net.load_state_dict(state, strict=True)
    w = net.fc.weight.detach().cpu().numpy().astype(np.float64)
    b = net.fc.bias.detach().cpu().numpy().astype(np.float64)
    net.fc = torch.nn.Identity()
    net.eval()
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    return net.to(device), w, b, device


def build_loader(data_root: Path, source: str, imglist_rel: str,
                 batch_size: int, num_workers: int):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    mean, std = NORMS[source]
    tf = transforms.Compose([
        transforms.Resize(32), transforms.CenterCrop(32),
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    class ImglistDataset(Dataset):
        def __init__(self):
            self.items = []
            root = data_root / "images_classic"
            for line in (data_root / imglist_rel).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(" ", 1)
                self.items.append((root / rel, int(label)))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            img = Image.open(path).convert("RGB")
            return tf(img), torch.tensor(label)

    ds = ImglistDataset()
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False)


def forward(model, loader, device):
    import torch
    from tqdm import tqdm
    hs, ys = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, position=0, leave=True):
            hs.append(model(x.to(device)).cpu())
            ys.append(y)
    return (torch.concat(hs).numpy().astype(np.float32),
            torch.concat(ys).numpy().astype(np.int64))


def extract_one(ckpt: Path, source: str, data_root: Path, out_dir: Path,
                use_cuda: bool, batch_size: int, num_workers: int) -> None:
    t0 = time.time()
    slug = f"{source}__{ckpt.parent.parent.name}__{ckpt.parent.name}"
    n_classes = N_CLASSES[source]
    model, w_np, b_np, device = build_model(ckpt, n_classes, use_cuda)
    il = f"benchmark_imglist/{source}"

    h_tr, y_tr = forward(model, build_loader(
        data_root, source, f"{il}/train_{source}.txt", batch_size,
        num_workers), device)
    fm = fit_feature_model(h_tr, y_tr, n_classes)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_tr.astype(np.float64), y_tr, n_classes)
    train_mean = fm.global_mean

    def scores_for(h: np.ndarray) -> dict:
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"],
                "MLS": hs_["MLS"], "CTM": ctm(h64, proto_unc),
                "Maha": maha(h64), "fDBD": fdbd(h64, g, w_np, train_mean),
                "_logits": g}

    record: dict = {"schema_icml_a": 1, "ckpt": str(ckpt), "slug": slug,
                    "source": source, "n_classes": n_classes,
                    "dim": int(h_tr.shape[1]), "n_train": int(len(h_tr)),
                    "geometry": geometry_record(w_np, b_np, fm),
                    "papyan": papyan_metrics(w_np, fm), "ood": {}}
    del h_tr, y_tr

    h_id, y_id = forward(model, build_loader(
        data_root, source, f"{il}/test_{source}.txt", batch_size,
        num_workers), device)
    sc_id = scores_for(h_id)
    res_id = (sc_id.pop("_logits").argmax(1) != y_id).astype(float)
    record["iid_test"] = dict(estimate_ood_coords(h_id, fm),
                              n=int(len(h_id)),
                              id_error_rate=float(res_id.mean()))
    del h_id

    for si, name in enumerate(SUITES[source], start=1):
        try:
            h_o, _ = forward(model, build_loader(
                data_root, source, f"{il}/test_{name}.txt", batch_size,
                num_workers), device)
            sc_o = scores_for(h_o)
            sc_o.pop("_logits")
            record["ood"][name] = dict(
                estimate_ood_coords(h_o, fm),
                p10=compact_p10(h_o, fm, w_np, set_index=si),
                **set_outcomes(sc_id, res_id, sc_o))
            del h_o
        except Exception as err:  # noqa: BLE001 - per-set isolation
            record["ood"][name] = {"error": str(err)}
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(record, indent=1, default=float))
    failed = out_dir / f"FAILED_{slug}.json"
    if failed.exists():
        failed.unlink()
    print(f"[roster-a] {slug}: {len(record['ood'])} sets, "
          f"{record['runtime_sec']}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True,
                    choices=["cifar10", "cifar100"])
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()
    assert MANIFEST.exists(), ("PROTOCOL GATE: commit and stage "
                               f"{MANIFEST} (icml_roster_a_enumerate.py) "
                               "before extraction")
    man = json.loads(MANIFEST.read_text())
    cks = [Path(p) for p in man["rosters"].get(args.source, [])]
    data_root = Path(args.data_root)
    il = f"benchmark_imglist/{args.source}"
    needed = ([f"{il}/train_{args.source}.txt",
               f"{il}/test_{args.source}.txt"]
              + [f"{il}/test_{n}.txt" for n in SUITES[args.source]])
    missing = []
    for r in needed:
        if not (data_root / r).is_file():
            missing.append(f"imglist {r}")
            continue
        for line in (data_root / r).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            img = line.rsplit(" ", 1)[0]
            if not (data_root / "images_classic" / img).is_file():
                missing.append(f"images for {r} (first ref: "
                               f"images_classic/{img}; run pilot0/"
                               f"icml_download_openood_cifar.sh)")
            break
    out_dir = Path(args.out_dir)
    todo = [c for c in cks if not (
        out_dir / (f"{args.source}__{c.parent.parent.name}__"
                   f"{c.parent.name}.json")).exists()]
    print(f"[roster-a] {args.source}: {len(cks)} enumerated, "
          f"{len(todo)} to run; missing imglists: {missing or 'none'}",
          flush=True)
    if args.list:
        for c in todo:
            print("  ", c)
        return
    assert not missing, f"missing imglists/data: {missing}"
    failures = 0
    for c in todo:
        try:
            extract_one(c, args.source, data_root, out_dir,
                        args.use_cuda, args.batch_size, args.num_workers)
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failures += 1
            slug = (f"{args.source}__{c.parent.parent.name}__"
                    f"{c.parent.name}")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{slug}.json").write_text(json.dumps(
                {"ckpt": str(c), "error": traceback.format_exc()},
                indent=1))
            print(f"[roster-a] FAILED {slug}", flush=True)
    print(f"[roster-a] done: {len(todo) - failures} ok, {failures} failed",
          flush=True)


if __name__ == "__main__":
    main()
