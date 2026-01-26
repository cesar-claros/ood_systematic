import os, argparse
import numpy as np
import joblib
from tqdm import tqdm
from loguru import logger
from src.rc_stats import RiskCoverageStats
from src.clip_utils import (
    load_clip,
    make_loader,
    extract_image_features,
    make_text_prototypes,
    make_text_confidence_prototypes,
    text_alignment_scores,
    text_confidence_alignment_scores,
    PROMPTS,
    K,
    LIKERT_scale,
    PROMPTS_confidence
)

# ---------- Runner (example) ----------
def run(iid_dataset:str):
    ti_condition = 'tinyimagenet' if (iid_dataset=='cifar10' or iid_dataset=='cifar100' or iid_dataset=='supercifar100') else 'cifar10'
    c100_condition = 'cifar100' if (iid_dataset=='cifar10' or iid_dataset=='tinyimagenet') else 'cifar10'
    logger.info('Loading CLIP model...')
    model, preprocess, tokenizer, device, backend = load_clip(model_name="ViT-B-32")
    logger.info(f'Loading {iid_dataset} data set...')
    # 1) Load ID = SuperCIFAR (CIFAR100 train as ID features)
    logger.info(f'Computing features...')
    id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=256)
    X_id = extract_image_features(model, id_loader, device, backend)  # [N_id, D]
    id_test_loader = make_loader(iid_dataset, split="test", preprocess=preprocess, batch_size=256)
    X_id_test, y_test = extract_image_features(model, id_test_loader, device, backend, targets_out=True)  # [N_id, D]

    # Optional: superclass text prototypes
    Tproto = make_text_prototypes(model, tokenizer, device, classes, backend=backend)
    Tproto_confidence = make_text_confidence_prototypes(model, tokenizer, device, classes, backend=backend)
    Tproto_confidence_flat = Tproto_confidence.view(len(classes)*K,-1).numpy()
    # Optional: superclass image centroids (needs y_super indices)
    # y_super = load_cifar100_coarse_indices(split="train")  # replace with your array of length N_id
    # C_img = None
    # if y_super is not None:
    #     C_img = superclass_centroids(X_id, y_super, n_super=20)
    # Zero-shot prediction
    scores_id, prediction_id = text_alignment_scores(X_id_test, Tproto)
    # Keep only correctly predicted instances for OOD evaluation
    X_id_correct_test = X_id_test[y_test==prediction_id]
    y_id_correct_test = np.ones(X_id_correct_test.shape[0],dtype=int)
    # 2) Example OOD loaders (replace paths/keys as needed)
    ood_specs = {
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
        y_ood = np.zeros(X_ood.shape[0],dtype=int)
        X_ood_augmented = np.concatenate([X_id_correct_test,X_ood], axis=0)
        y_augmented = np.concatenate([y_id_correct_test, y_ood], axis=0)
        logger.info(f'Computing distances...')
        
        scores, class_idx, uncertainty_idx = text_confidence_alignment_scores(X_ood_augmented, Tproto, Tproto_confidence_flat)
        stats = RiskCoverageStats(confids=scores, residuals=1-y_augmented)

        results[name] = {
            "stats": stats,
            'class_idx':class_idx,
            'uncertainty_idx':uncertainty_idx,
        }
        print(f"[{name}] AUGRC={stats.augrc}")
    results['global']={
        'classes':classes,
        'prompts_confidence':PROMPTS_confidence,
    }
    # Optional: save results
    path = f"clip_uncertainty_{iid_dataset}.joblib"
    # with open(path, "w") as f:
    joblib.dump(results, path)
    print(f"Saved: {path}")

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    parser.add_argument('--iid_dataset', 
                        type=str, 
                        required=True, 
                        help="IID data set name", 
                        choices=['cifar10','cifar100','supercifar100','tinyimagenet'])
    args = parser.parse_args()
    iid_dataset = args.iid_dataset
    run(iid_dataset)
