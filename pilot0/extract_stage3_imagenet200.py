"""Stage-3 OpenOOD ImageNet-200 pilot extraction (source-expansion
protocol stage 3; HPC side).

FROZEN SPECIFICATION (declared before any ImageNet-200 outcome is
computed; protocol frozen 2026-08-26, amendments 1-5)
=====================================================================

Scope (protocol stage 3): the THREE released OpenOOD v1.5 ImageNet-200
ResNet-18 cross-entropy runs, testing evaluator harmonization, score
orientation, formula saturation, coordinate support, detector-family
ranking direction, and material-gap prevalence. NO geometry regression at
n = 3. Transport label: joint source-objective-architecture (cross-entropy
and ResNet-18 are both absent from the training pool). Covariate-shifted
ID sets (ImageNet-C/R/v2) are OUT of pilot scope.

Integration rules (protocol section 2.3): OpenOOD supplies image lists,
splits, and datasets only; features and logits are exported here and
scored with THIS project's frozen implementations. Suites: ID =
test_imagenet200; near-OOD = ssb_hard, ninco; far-OOD = textures,
inaturalist, openimage_o (the v1.5 imagenet200 ood config). Preprocessing
replicates the OpenOOD imagenet200 dataset config exactly: Resize(256,
bilinear), CenterCrop(224), ImageNet normalization. DECLARED DEVIATION
from the fd-shifts convention: the train split is forwarded with the SAME
deterministic test transform (external checkpoints carry no fd-shifts
train-augmentation convention to preserve; the feature model is fit on
unaugmented activations).

Model loading: torchvision resnet18(num_classes=200) (OpenOOD's
ResNet18_224x224 subclasses the torchvision ResNet, so state-dict keys
match); penultimate feature = avgpool flatten (512-d); logits recomputed
as h @ W.T + b from the stored fc, the project's sliced-head convention.
Checkpoints are auto-discovered as best*.ckpt under --ckpt_root (expect 3
seed runs).

Outcomes: identical machinery to Stage 2 (amendment 5c/5d):
Energy/CTM claim-bearing, MSR/MLS/Maha/fDBD secondary; per set the raw and
prevalence-balanced failure AUGRC (subsample to min(n_id, n_ood), rng
20260827), signed Energy-CTM gaps, materiality on the balanced gap;
frozen geometry/coordinate estimators unchanged.

Usage (from code/, inside the campaign container):
    python pilot0/extract_stage3_imagenet200.py --list \
        --ckpt_root $DATASET_ROOT_DIR/openood/results \
        --data_root $DATASET_ROOT_DIR/openood/data
    python pilot0/extract_stage3_imagenet200.py --ckpt_root ... --data_root ...
Outputs: pilot0/stage3_imagenet200_coords/<run>.json
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

from pilot0.extract_stage2_expansion import set_outcomes
from pilot0.geometry import (fit_feature_model, geometry_record,
                             papyan_metrics)
from pilot0.ood_coords import estimate_ood_coords
from pilot0.scores import MahalanobisScorer, ctm, fdbd, head_scores

SCHEMA_S3 = 1
OUT_DIR_DEFAULT = "pilot0/stage3_imagenet200_coords"
N_CLASSES = 200
TRAIN_LIST = ("benchmark_imglist/imagenet200/train_imagenet200.txt",
              "images_largescale")
ID_LIST = ("benchmark_imglist/imagenet200/test_imagenet200.txt",
           "images_largescale")
OOD_LISTS = {
    "ssb_hard": ("benchmark_imglist/imagenet/test_ssb_hard.txt",
                 "images_largescale", "near"),
    "ninco": ("benchmark_imglist/imagenet/test_ninco.txt",
              "images_largescale", "near"),
    "textures": ("benchmark_imglist/imagenet/test_textures.txt",
                 "images_classic", "far"),
    "inaturalist": ("benchmark_imglist/imagenet/test_inaturalist.txt",
                    "images_largescale", "far"),
    "openimage_o": ("benchmark_imglist/imagenet/test_openimage_o.txt",
                    "images_largescale", "far"),
}


def build_loader(data_root: Path, imglist_rel: str, images_sub: str,
                 batch_size: int, num_workers: int):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256,
                          interpolation=transforms.InterpolationMode
                          .BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    class ImglistDataset(Dataset):
        def __init__(self):
            self.items = []
            root = data_root / images_sub
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


def forward(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from tqdm import tqdm

    hs, ys = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, position=0, leave=True):
            hs.append(model(x.to(device)).cpu())
            ys.append(y)
    return (torch.concat(hs).numpy().astype(np.float32),
            torch.concat(ys).numpy().astype(np.int64))


def extract_one(ckpt: Path, run_name: str, data_root: Path, out_dir: Path,
                use_cuda: bool, batch_size: int, num_workers: int,
                exclude_basenames: set | None = None) -> None:
    import torch
    from torchvision.models import resnet18

    t0 = time.time()
    device = torch.device("cuda" if use_cuda else "cpu")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model = resnet18(num_classes=N_CLASSES)
    model.load_state_dict(state)
    w_np = model.fc.weight.detach().numpy().astype(np.float64)
    b_np = model.fc.bias.detach().numpy().astype(np.float64)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    def loader_for(spec):
        return build_loader(data_root, spec[0], spec[1], batch_size,
                            num_workers)

    print(f"[{run_name}] forward train", flush=True)
    h_train, y_train = forward(model, loader_for(TRAIN_LIST), device)
    assert int(y_train.max()) < N_CLASSES
    fm = fit_feature_model(h_train, y_train, N_CLASSES)
    proto_unc = fm.class_means + fm.global_mean
    maha = MahalanobisScorer(h_train.astype(np.float64), y_train, N_CLASSES)

    def scores_for(h: np.ndarray) -> dict:
        h64 = h.astype(np.float64)
        g = h64 @ w_np.T + b_np
        hs_ = head_scores(g)
        return {"Energy": hs_["Energy"], "MSR": hs_["MSR"],
                "MLS": hs_["MLS"], "CTM": ctm(h64, proto_unc),
                "Maha": maha(h64),
                "fDBD": fdbd(h64, g, w_np, fm.global_mean), "_logits": g}

    record: dict = {
        "schema_stage3": SCHEMA_S3, "run": run_name, "ckpt": str(ckpt),
        "source": "imagenet200", "paradigm": "crossentropy",
        "backbone": "resnet18", "n_classes": N_CLASSES,
        "dim": int(h_train.shape[1]), "n_train": int(len(h_train)),
        "geometry": geometry_record(w_np, b_np, fm),
        "papyan": papyan_metrics(w_np, fm), "ood": {},
    }

    print(f"[{run_name}] forward id test", flush=True)
    id_loader = loader_for(ID_LIST)
    h_iid, y_iid = forward(model, id_loader, device)
    n_excluded = 0
    if exclude_basenames:
        import os as _os
        paths = [str(p_) for p_, _ in id_loader.dataset.items]
        assert len(paths) == len(h_iid)
        keep = np.array([_os.path.basename(p_) not in exclude_basenames
                         for p_ in paths])
        n_excluded = int((~keep).sum())
        h_iid, y_iid = h_iid[keep], y_iid[keep]
    record["n_idtest_excluded"] = n_excluded
    sc_id = scores_for(h_iid)
    res_id = (sc_id["_logits"].argmax(1) != y_iid).astype(float)
    record["iid_test"] = dict(estimate_ood_coords(h_iid, fm),
                              n=int(len(h_iid)),
                              id_error_rate=round(float(res_id.mean()), 4))

    n_material = 0
    for cname, spec in OOD_LISTS.items():
        try:
            print(f"[{run_name}] forward {cname}", flush=True)
            h_ood, _ = forward(model, loader_for(spec), device)
            sc_ood = scores_for(h_ood)
            entry = dict(estimate_ood_coords(h_ood, fm), kind=spec[2])
            entry.update(set_outcomes(sc_id, res_id, sc_ood))
            n_material += int(entry["material"])
            record["ood"][cname] = entry
            del h_ood, sc_ood
        except Exception as err:  # noqa: BLE001 - per-set isolation
            print(f"[{run_name}] {cname} FAILED: {err}", flush=True)
            record["ood"][cname] = {"error": str(err), "kind": spec[2]}

    ok = [c for c, v in record["ood"].items() if "error" not in v]
    record["n_sets_extracted"] = len(ok)
    record["suite_complete"] = len(ok) == len(OOD_LISTS)
    record["n_material_sets"] = n_material
    record["runtime_sec"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.json").write_text(json.dumps(record, indent=1))
    failed = out_dir / f"FAILED_{run_name}.json"
    if failed.exists():
        failed.unlink()
    print(f"[{run_name}] wrote stage-3 record ({len(ok)}/{len(OOD_LISTS)} "
          f"sets, {n_material} material, {record['runtime_sec']}s)",
          flush=True)


def discover_ckpts(ckpt_root: Path) -> list[tuple[Path, str]]:
    by_parent: dict[Path, Path] = {}
    for p in sorted(ckpt_root.rglob("best*.ckpt")):
        cur = by_parent.get(p.parent)
        if cur is None or p.name == "best.ckpt":
            by_parent[p.parent] = p
    return [(p, f"{p.parent.parent.name}__{p.parent.name}")
            for _, p in sorted(by_parent.items())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT)
    parser.add_argument("--exclude_idtest", type=str, default=None)
    args = parser.parse_args()
    ckpts = discover_ckpts(Path(args.ckpt_root))
    data_root = Path(args.data_root)
    print(f"[stage3] {len(ckpts)} checkpoints under {args.ckpt_root} "
          f"(expect 3)")
    missing = [rel for rel, _ in ([TRAIN_LIST, ID_LIST]
                                  + [v[:2] for v in OOD_LISTS.values()])
               if not (data_root / rel).is_file()]
    for rel in missing:
        print(f"[stage3] MISSING imglist: {rel}")
    for ckpt, name in ckpts:
        print(f"[stage3] {name}: {ckpt}")
    if args.list:
        return
    if missing:
        sys.exit("[stage3] imglists missing; download benchmark_imglist "
                 "first")
    exclude_basenames = None
    if args.exclude_idtest:
        import os as _os
        cert = json.loads(Path(args.exclude_idtest).read_text())
        exclude_basenames = {_os.path.basename(p_)
                             for d in cert.get("duplicates", [])
                             for p_ in d["id_test"]}
        print(f"[stage3] DEDUP MODE: excluding {len(exclude_basenames)} "
              f"id_test files")
    import torch
    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    out_dir = Path(args.out_dir)
    done = skipped = failed = 0
    for ckpt, name in ckpts:
        if (out_dir / f"{name}.json").exists():
            skipped += 1
            continue
        try:
            extract_one(ckpt, name, data_root, out_dir, use_cuda,
                        args.batch_size, args.num_workers,
                        exclude_basenames)
            done += 1
        except Exception:  # noqa: BLE001 - per-checkpoint isolation
            failed += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"FAILED_{name}.json").write_text(json.dumps(
                {"ckpt": str(ckpt),
                 "traceback": traceback.format_exc()}, indent=1))
            print(f"[stage3] FAILED {name} (recorded, continuing)")
    print(f"[stage3] finished: {done} new, {skipped} skipped, "
          f"{failed} failed")


if __name__ == "__main__":
    main()
