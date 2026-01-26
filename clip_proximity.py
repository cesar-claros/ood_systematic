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
def run(iid_dataset:str, output_dir:str):
    ti_condition = 'tinyimagenet' if (iid_dataset=='cifar10' or iid_dataset=='cifar100' or iid_dataset=='supercifar100') else 'cifar10'
    c100_condition = 'cifar100' if (iid_dataset=='cifar10' or iid_dataset=='tinyimagenet') else 'cifar10'
    logger.info('Loading CLIP model...')
    model, preprocess, tokenizer, device, backend = load_clip(model_name="ViT-B-32")
    logger.info(f'Loading {iid_dataset} data set...')
    # 1) Load ID = SuperCIFAR (CIFAR100 train as ID features)
    # id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=256) # Duplicate line removed
    logger.info(f'Computing features...')
    id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=256)
    X_id, y = extract_image_features(model, id_loader, device, backend, targets_out=True)  # [N_id, D]
    # id_test_loader = make_loader(iid_dataset, split="test", preprocess=preprocess, batch_size=256)
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
            loader = make_loader(spec, split="test", preprocess=preprocess, batch_size=256)
        else:
            loader = make_loader(spec, split=split, preprocess=preprocess, batch_size=256)
        logger.info(f'Computing features...')
        X_ood = extract_image_features(model, loader, device, backend)
        logger.info(f'Computing distances...')
        # ---- Global proximity ----
        knn = knn_distances(X_id, X_ood, k=5)
        fid = fid_from_features(X_id, X_ood)
        kid_mean, kid_std = kid_mmd(X_id, X_ood, n_subsets=50, subset_size=1000)
        # mu, S_inv = fit_mahalanobis(X_id, shrinkage=0.05)
        # maha = mahalanobis_score(mu, S_inv, X_ood)

        # cover95 = coverage_at_r(X_id, X_ood, r_quantile=0.95, k=5)

        # ---- Superclass-aware (hierarchical) ----
        text_align = text_alignment_scores(X_ood, Tproto)
        # img_centroid_dist = None
        # if C_img is not None:
        img_centroid_dist = nearest_class_distance(X_ood, C_img)

        results[name] = {
            "global": {
                "knn_mean": float(knn.mean()),
                "knn_median": float(np.median(knn)),
                # "mahalanobis_mean": float(maha.mean()),
                # "mahalanobis_median": float(np.median(maha)),
                "fid": float(fid),
                "kid_mean": float(kid_mean),
                "kid_std": float(kid_std),
                # "coverage@r(id-95pct,k=5)": float(cover95),
            },
            "class_aware": {
                "text_alignment_mean": float(text_align.mean()),
                "text_alignment_median": float(np.median(text_align)),
                "img_centroid_dist_mean": (float(img_centroid_dist.mean()) if img_centroid_dist is not None else None),
                "img_centroid_dist_median": (float(np.median(img_centroid_dist)) if img_centroid_dist is not None else None),
            }
        }
        # print(f"[{name}] {json.dumps(results[name], indent=2)}")
        logger.info(f"[{name}] Results computed.")

    # Optional: save results
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"clip_proximity_{iid_dataset}.json")
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
    args = parser.parse_args()
    iid_dataset = args.iid_dataset
    output_dir = args.output_dir
    run(iid_dataset, output_dir)
