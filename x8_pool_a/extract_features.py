"""Pool A feature extraction (HPC side): frozen DINOv2 / CLIP encoders.

Caches L2-unnormalized penultimate features per (encoder, dataset, split) as
.npz under $EXPERIMENT_ROOT_DIR/pool_a/features/. Dataset resolution reuses
`src.clip_utils.make_loader`, i.e. the exact folders and torchvision datasets
the paper's CLIP-proximity pipeline uses ($DATASET_ROOT_DIR layout for the
ImageFolder OOD sets, ./data for the torchvision ID sets).

ID sources (cifar10, cifar100, supercifar100, tinyimagenet) are extracted for
both train and test splits (probes fit on train, CSFs/regret evaluate on
test); OOD sets are extracted once (their evaluation split).

Usage (one GPU, ~1-3 h total for all encoders x datasets):
    python extract_features.py --encoder dinov2_vitb14 --dataset all
    python extract_features.py --encoder clip_vitb16   --dataset all
    python extract_features.py --encoder clip_vitb16 --dataset cifar10 \
        --limit 512   # smoke test
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.clip_utils import make_loader  # noqa: E402

ID_DATASETS = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
OOD_DATASETS = ["isun", "lsun_cropped", "lsun_resize", "svhn", "places365",
                "textures"]
DATASETS = ID_DATASETS + OOD_DATASETS
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Pinned pre-SDPA ref (Aug 2023): runs on torch 1.13 (paper container) and
# torch 2.x alike; matches the hub cache baked into the container image.
DINOV2_HUB_REPO = "facebookresearch/dinov2:81b2b64"


def load_encoder(name: str, device: str):
    """Frozen encoder + its preprocessing transform.

    Returns:
        (model, preprocess): model maps a preprocessed batch to features;
        preprocess is a torchvision-style transform for PIL images.
    """
    if name.startswith("dinov2"):
        import torchvision.transforms as T
        try:
            # skip_validation: the fork check rejects commit SHAs outright
            model = torch.hub.load(DINOV2_HUB_REPO, name, skip_validation=True)
        except Exception as hub_err:
            # Fallback for environments with a modern timm (>=0.9). The paper
            # container instead bakes a build-verified hub cache (TORCH_HOME),
            # because fd-shifts pins timm==0.5.4 which predates DINOv2.
            timm_names = {"dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m"}
            try:
                import timm
                model = timm.create_model(timm_names[name], pretrained=True,
                                          num_classes=0, img_size=224)
                print(f"torch.hub dinov2 load failed ({hub_err}); "
                      f"using timm {timm_names[name]}")
            except Exception as timm_err:
                raise RuntimeError(
                    f"DINOv2 unavailable: torch.hub failed ({hub_err}) and "
                    f"timm fallback failed ({timm_err}). Use the rebuilt "
                    "paper container (baked hub cache) or a venv with "
                    "torch>=2 per the dispatch runbook.") from timm_err
        preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif name.startswith("clip"):
        import open_clip
        arch = {"clip_vitb16": "ViT-B-16", "clip_vitl14": "ViT-L-14"}[name]
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained="openai")
        model = model.visual
    else:
        raise ValueError(name)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device), preprocess


def build_loader(dataset: str, split: str, preprocess, batch_size: int,
                 num_workers: int, limit: int | None):
    """Site loader via clip_utils.make_loader, normalized to a bare loader."""
    out = make_loader(dataset, split=split, batch_size=batch_size,
                      num_workers=num_workers, preprocess=preprocess,
                      limit=limit)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def extract(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """One pass over the loader collecting features and labels together."""
    feats, labels = [], []
    for x, y in tqdm(loader, desc="extracting", leave=False):
        feats.append(model(x.to(device, non_blocking=True))
                     .float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def main() -> None:
    """Extract and cache features for the requested encoder and datasets."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--encoder", required=True,
                    choices=["dinov2_vitb14", "clip_vitb16", "clip_vitl14"])
    ap.add_argument("--dataset", required=True,
                    help=f"one of {DATASETS} or 'all'")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per dataset (smoke tests)")
    ap.add_argument("--out-dir", default=None,
                    help="default: $EXPERIMENT_ROOT_DIR/pool_a/features")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(
        args.out_dir
        or pathlib.Path(os.environ["EXPERIMENT_ROOT_DIR"]) / "pool_a" / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess = load_encoder(args.encoder, device)
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        splits = ["train", "test"] if ds in ID_DATASETS else ["test"]
        for split in splits:
            out_path = out_dir / f"{args.encoder}_{ds}_{split}.npz"
            if out_path.exists():
                print(f"skip (exists): {out_path}")
                continue
            loader = build_loader(ds, split, preprocess, args.batch_size,
                                  args.num_workers, args.limit)
            feats, labels = extract(model, loader, device)
            np.savez_compressed(out_path, features=feats, labels=labels)
            print(f"wrote {out_path}  features={feats.shape} "
                  f"labels={labels.shape}")


if __name__ == "__main__":
    main()
