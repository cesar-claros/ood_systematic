from __future__ import annotations

import os, json, argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from src.clip_utils import (
    load_clip,
    make_loader,
    extract_image_features,
    make_text_prototypes,
    knn_distances,
    fid_from_features,
    kid_mmd,
    class_centroids,
    nearest_class_distance,
    text_alignment_scores
)

# ---------- Runner (example) ----------
def proximity_metrics(X_id: np.ndarray, X_ood: np.ndarray, Tproto: np.ndarray,
                      C_img: np.ndarray, knn_chunk: int | None = None) -> dict:
    """Global and class-aware CLIP proximity of one evaluation set to the ID set.

    Args:
        X_id: L2-normalized ID reference features [N_id, D].
        X_ood: L2-normalized features of the evaluation set [N, D].
        Tproto: ID class text prototypes [C, D].
        C_img: ID class image centroids [C, D].
        knn_chunk: If set, compute the k-NN distances over row chunks of
            X_ood of this size (identical values, bounded memory).

    Returns:
        The {"global": ..., "class_aware": ...} record stored per set.
    """
    if knn_chunk is None:
        knn = knn_distances(X_id, X_ood, k=5)
    else:
        knn = np.concatenate([knn_distances(X_id, X_ood[i:i + knn_chunk], k=5)
                              for i in range(0, len(X_ood), knn_chunk)])
    fid = fid_from_features(X_id, X_ood)
    kid_mean, kid_std = kid_mmd(X_id, X_ood, n_subsets=50, subset_size=1000)
    text_align = text_alignment_scores(X_ood, Tproto)
    img_centroid_dist = nearest_class_distance(X_ood, C_img)
    return {
        "global": {
            "knn_mean": float(knn.mean()),
            "knn_median": float(np.median(knn)),
            "fid": float(fid),
            "kid_mean": float(kid_mean),
            "kid_std": float(kid_std),
        },
        "class_aware": {
            "text_alignment_mean": float(text_align.mean()),
            "text_alignment_median": float(np.median(text_align)),
            "img_centroid_dist_mean": float(img_centroid_dist.mean()),
            "img_centroid_dist_median": float(np.median(img_centroid_dist)),
        },
    }


def run(iid_dataset:str, output_dir:str, model_name:str = "ViT-B-32", pretrained:str = None, batch_size:int = 256):
    ti_condition = 'tinyimagenet' if (iid_dataset=='cifar10' or iid_dataset=='cifar100' or iid_dataset=='supercifar100') else 'cifar10'
    c100_condition = 'cifar100' if (iid_dataset=='cifar10' or iid_dataset=='tinyimagenet') else 'cifar10'
    logger.info(f'Loading CLIP model: {model_name}...')
    model, preprocess, tokenizer, device, backend = load_clip(model_name=model_name, pretrained=pretrained)
    logger.info(f'Loading {iid_dataset} data set...')
    # 1) Load ID = SuperCIFAR (CIFAR100 train as ID features)
    # id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=batch_size) # Duplicate line removed
    logger.info(f'Computing features...')
    id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=batch_size)
    X_id, y = extract_image_features(model, id_loader, device, backend, targets_out=True)  # [N_id, D]
    # id_test_loader = make_loader(iid_dataset, split="test", preprocess=preprocess, batch_size=batch_size)
    # X_id_test, y_test = extract_image_features(model, id_test_loader, device, backend, targets_out=True)  # [N_id, D]
    # Optional: superclass text prototypes
    Tproto = make_text_prototypes(model, tokenizer, device, classes, backend=backend)

    # Optional: superclass image centroids (needs y_super indices)
    # y_super = load_cifar100_coarse_indices(split="train")  # replace with your array of length N_id
    # C_img = None
    # if y_super is not None:
    C_img = class_centroids(X_id, y)

    # 2) Example OOD loaders (replace paths/keys as needed)
    ood_specs = {
        f'{iid_dataset}':(f'{iid_dataset}','test'),
        c100_condition: (c100_condition, "test"),
        ti_condition: (ti_condition, "test"),
        "isun": ("isun", None),
        "lsun_resize": ("lsun_resize", None),
        "lsun_cropped": ("lsun_cropped", None),
        "places365": ("places365", None),
        "svhn": ("svhn", "test"),
        "textures": ("textures", None),
    }

    results = {}
    logger.info(f'Loading OOD dataset: {ood_specs.keys()}...')
    for name, (spec, split) in tqdm(ood_specs.items()):
        logger.info(f'Loading {name} data set...')
        if split == "folder":
            loader = make_loader(spec, split="test", preprocess=preprocess, batch_size=batch_size)
        else:
            loader = make_loader(spec, split=split, preprocess=preprocess, batch_size=batch_size)
        logger.info(f'Computing features...')
        X_ood = extract_image_features(model, loader, device, backend)
        logger.info(f'Computing distances...')
        results[name] = proximity_metrics(X_id, X_ood, Tproto, C_img)
        # print(f"[{name}] {json.dumps(results[name], indent=2)}")
        logger.info(f"[{name}] Results computed.")

    # Optional: save results
    os.makedirs(output_dir, exist_ok=True)
    model_tag = model_name.replace("/", "-")
    path = os.path.join(output_dir, f"clip_proximity_{iid_dataset}_{model_tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved results to: {path}")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    parser.add_argument('--iid_dataset',
                        type=str,
                        required=True,
                        help="IID data set name",
                        choices=['cifar10','cifar100','supercifar100','tinyimagenet'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='.',
                        help="Directory to save the results")
    parser.add_argument('--model_name',
                        type=str,
                        default='ViT-B-32',
                        help="CLIP model name (e.g., ViT-B-32, ViT-B-16, ViT-L-14)")
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help="Pretrained weights tag (default: auto-resolved per model)")
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help="Batch size for feature extraction (default: 256, try 512-1024 on GPU)")
    args = parser.parse_args()
    run(args.iid_dataset, args.output_dir,
        model_name=args.model_name, pretrained=args.pretrained,
        batch_size=args.batch_size)
