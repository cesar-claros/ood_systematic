"""CLIP near/mid/far regimes for the ImageNet suite (Sec 3.4 protocol).

Applies the paper's protocol unchanged to ImageNet-1k as the ID set and the
eight x9 OOD sets: L2-normalized CLIP image features, the four proximity
metrics of `clip_proximity.proximity_metrics` (KID, FID, inverse text
alignment against prompt-ensembled class prototypes, image-centroid
distance), and k = 3 k-means on the standardized metrics via
`clip_clustering_all_backbones.cluster_one` (clusters ordered by FID).

ID reference = a class-balanced draw of --id-per-class images per class from
the x9 superset (the paper uses the full CIFAR/TinyImageNet train sets, 50k
to 100k images; 50 per class gives 50k here). The 'test' row is the full
ImageNet validation set. Class names are open_clip's IMAGENET_CLASSNAMES,
whose order matches the HF label ids.

Runs in the paper container (open_clip, pyarrow); one GPU, about 20 min per
encoder. Outputs under --out-dir:
  clip_proximity_imagenet_<encoder>.json  (same layout as clip_proximity.py)
  clip_distances_imagenet_<encoder>.csv   (same layout as clip_scores/)

  singularity exec --nv <paper.sif> python x9_imagenet/clip_regimes.py \
      --data-root $DATASET_ROOT_DIR --out-dir clip_scores_imagenet
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from open_clip import IMAGENET_CLASSNAMES

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))
from clip_clustering_all_backbones import cluster_one  # noqa: E402
from clip_proximity import proximity_metrics  # noqa: E402
from src.clip_utils import (  # noqa: E402
    class_centroids,
    load_clip,
    make_text_prototypes,
)
from x9_imagenet.data import make_loader  # noqa: E402

ENCODERS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
KNN_CHUNK = 8192


def reference_uids(superset_dir: pathlib.Path, per_class: int, seed: int
                   ) -> set[str]:
    """Class-balanced seeded draw of superset row uids ("<shard>:<row>")."""
    m = pd.read_parquet(superset_dir / "superset_manifest.parquet")
    m["out_row"] = m.groupby("out_shard").cumcount()
    m["uid"] = (m["out_shard"].map(lambda s: f"superset-{s:05d}.parquet")
                + ":" + m["out_row"].astype(str))
    rng = np.random.default_rng(seed)
    return set(m.groupby("label")["uid"]
               .apply(lambda u: list(rng.permutation(u.to_numpy())[:per_class]))
               .explode())


@torch.no_grad()
def embed(model, preprocess, shards: list[pathlib.Path], device: str,
          row_filter: set[str] | None, batch_size: int, num_workers: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalized CLIP image features and labels for a list of shards."""
    loader = make_loader(shards, preprocess, row_filter=row_filter,
                         batch_size=batch_size, num_workers=num_workers)
    feats, labels = [], []
    for imgs, y, _ in loader:
        f = model.encode_image(imgs.to(device, non_blocking=True)).float()
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def run_encoder(encoder: str, paths: dict, uids: set[str], out_dir: pathlib.Path,
                batch_size: int, num_workers: int) -> pd.DataFrame:
    """Proximity JSON and k = 3 regime CSV for one CLIP encoder."""
    model, preprocess, tokenizer, device, backend = load_clip(model_name=encoder)
    x_id, y_id = embed(model, preprocess, paths["superset"], device, uids,
                       batch_size, num_workers)
    logger.info(f"{encoder}: ID reference {x_id.shape}")
    t_proto = make_text_prototypes(model, tokenizer, device,
                                   list(IMAGENET_CLASSNAMES), backend=backend)
    c_img = class_centroids(x_id, y_id)

    results = {}
    for name, shards in [("imagenet", paths["val"]), *paths["ood"].items()]:
        x, _ = embed(model, preprocess, shards, device, None, batch_size,
                     num_workers)
        results[name] = proximity_metrics(x_id, x, t_proto, c_img,
                                          knn_chunk=KNN_CHUNK)
        logger.info(f"{encoder}: {name} ({len(x):,}) "
                    f"fid {results[name]['global']['fid']:.4f}")

    json_path = out_dir / f"clip_proximity_imagenet_{encoder}.json"
    json_path.write_text(json.dumps(results, indent=2))
    distances, metrics_show = cluster_one(str(json_path), "imagenet", n_clusters=3)
    distances[metrics_show].to_csv(out_dir / f"clip_distances_imagenet_{encoder}.csv")
    groups = distances["group_name"]
    groups = groups.iloc[:, 0] if isinstance(groups, pd.DataFrame) else groups
    for name, regime in groups.items():
        logger.info(f"{encoder}: {name} -> {regime}")
    return distances


def main() -> None:
    """Embed the suite with every encoder and write proximities and regimes."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="clip_scores_imagenet")
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--id-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    root = pathlib.Path(args.data_root)
    ood: dict[str, list[pathlib.Path]] = {}
    for f in sorted((root / "ood_parquet").glob("*-*.parquet")):
        ood.setdefault(f.name.rsplit("-", 1)[0], []).append(f)
    paths = {
        "superset": sorted((root / "imagenet1k_superset").glob("superset-*.parquet")),
        "val": sorted((root / "imagenet1k_raw" / "data").glob("validation-*.parquet")),
        "ood": ood,
    }
    if not (paths["superset"] and paths["val"] and len(ood) == 8):
        raise SystemExit(f"missing data under {root}: superset "
                         f"{len(paths['superset'])}, val {len(paths['val'])}, "
                         f"OOD sets {sorted(ood)}")
    uids = reference_uids(root / "imagenet1k_superset", args.id_per_class, args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for encoder in args.encoders:
        run_encoder(encoder, paths, uids, out_dir, args.batch_size,
                    args.num_workers)


if __name__ == "__main__":
    main()
