#%%
import os, math, json, random, sys, warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision as tv
import torchvision.transforms as T
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar
import tinyimagenet
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from loguru import logger
import pickle
from PIL import Image, ImageFile
from tqdm import tqdm
import argparse
from src.rc_stats import RiskCoverageStats
import joblib
import open_clip
#%%
class SuperCIFAR100(tv.datasets.VisionDataset):
    """Super`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This holds out subclasses

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (Callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (Callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        kwargs: Optional[Callable] = None,
    ) -> None:
        super(SuperCIFAR100, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.coarse_targets = []
        self.fine_targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.coarse_targets.extend(entry["coarse_labels"])
                    self.fine_targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        holdout_fine_classes = [
            "seal",
            "flatfish",
            "tulip",
            "bowl",
            "pear",
            "lamp",
            "couch",
            "beetle",
            "lion",
            "skyscraper",
            "sea",
            "cattle",
            "raccoon",
            "lobster",
            "girl",
            "lizard",
            "hamster",
            "oak_tree",
            "motorcycle",
            "tractor",
        ]

        holdout_fine_idx = [self.class_to_idx[cl] for cl in holdout_fine_classes]
        train_data_ix = [
            ix
            for ix, (fine_label, coarse_label) in enumerate(
                zip(self.fine_targets, self.coarse_targets)
            )
            if (fine_label not in holdout_fine_idx and coarse_label != 19)
        ]
        holdout_data_ix = [
            ix
            for ix, (fine_label, coarse_label) in enumerate(
                zip(self.fine_targets, self.coarse_targets)
            )
            if (fine_label in holdout_fine_idx and coarse_label != 19)
        ]

        if self.train:
            self.targets = list(np.array(self.coarse_targets)[train_data_ix])
            self.data = self.data[train_data_ix]
        else:
            self.targets = list(np.array(self.coarse_targets)[holdout_data_ix])
            self.data = self.data[holdout_data_ix]

        num_classes = len(np.unique(self.targets))
        logger.info(
            "SuperCIFAR check num_classes in data {}. Training {}".format(
                num_classes, self.train
            )
        )
        self.classes = self.coarse_classes

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
            self.coarse_classes = data["coarse_label_names"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            logger.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")

#%%
# ---------- CLIP loader (open_clip preferred; fallback to openai/clip) ----------
def load_clip(model_name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = None        
    if pretrained is None:
        # strong general model; change if you like "laion2b_s34b_b79k" or "openai"
        pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device, "open_clip"


#%%
# ---------- Datasets ----------
# For datasets not in torchvision (e.g., Tiny-ImageNet, iSUN, Places), use ImageFolder.
def make_loader(
    name_or_path: str,
    split: str = "test",
    batch_size: int = 256,
    num_workers: int = 4,
    preprocess=None,
    limit: Optional[int] = None,
    data_root_dir:str = Path('/work/cniel/sw/FD_Shifts/project/datasets'),
):
    if os.path.isdir(name_or_path):
        # Generic folder: each subdir is a class (ignored for global metrics)
        ds = tv.datasets.ImageFolder(root=name_or_path, transform=preprocess)
    else:
        name = name_or_path.lower()
        if name == "cifar10":
            ds = tv.datasets.CIFAR10(root="./data", train=(split=="train"), download=True, transform=preprocess)
            classes = ds.classes
        elif name == "cifar100":
            ds = tv.datasets.CIFAR100(root="./data", train=(split=="train"), download=True, transform=preprocess)
            classes = ds.classes
        elif name == "supercifar100":
            ds = SuperCIFAR100(root="./data", train=(split=="train"), download=True, transform=preprocess)
            classes = ds.classes
            classes.remove('vehicles_2')
        elif name == "tinyimagenet":
            ds = tinyimagenet.TinyImageNet(root="./data", split=("train" if split=="train" else "val"), transform=preprocess)
            classes = [words[0] for n,words in ds.idx_to_words.items()]
        elif name == "svhn":
            ds = tv.datasets.SVHN(root="./data", split=("train" if split=="train" else "test"), download=True, transform=preprocess)
        elif name == "lsun_resize":
            try:
                ds = tv.datasets.ImageFolder(data_root_dir.joinpath('LSUN_resize'), transform=preprocess)
            except:
                # expects LSUN download; otherwise use ImageFolder path
                raise ValueError("Use a folder path for lsun_resize (preprocessed) or prepare lsun_resize accordingly.")
        elif name == "lsun_cropped":
            try:
                ds = tv.datasets.ImageFolder(data_root_dir.joinpath('LSUN'), transform=preprocess)
            except:
                # expects LSUN download; otherwise use ImageFolder path
                raise ValueError("Use a folder path for lsun_cropped (preprocessed) or prepare lsun_cropped accordingly.")
        elif name == "isun":
            try:
                ds = tv.datasets.ImageFolder(data_root_dir.joinpath('iSUN'), transform=preprocess)
            except:
                # expects LSUN download; otherwise use ImageFolder path
                raise ValueError("Use a folder path for isun (preprocessed) or prepare isun accordingly.")
        elif name == "textures":
            try:
                ds = tv.datasets.ImageFolder(data_root_dir.joinpath('dtd','images'), transform=preprocess)
            except:
                # expects LSUN download; otherwise use ImageFolder path
                raise ValueError("Use a folder path for textures (preprocessed) or prepare textures accordingly.")
        elif name == "places365":
            try:
                ds = tv.datasets.ImageFolder(data_root_dir.joinpath('places365'), transform=preprocess)
            except:
                # expects LSUN download; otherwise use ImageFolder path
                raise ValueError("Use a folder path for places365 (preprocessed) or prepare places365 accordingly.")
        elif name == "dtd":
            try:
                ds = tv.datasets.DTD(root="./data", split=("train" if split=="train" else "test"), download=True, transform=preprocess)
            except:
                raise ValueError("DTD not available in your torchvision. Provide a folder path instead.")
        else:
            raise ValueError(f"Unknown dataset key: {name_or_path}. Use a folder path or supported key.")
    if limit is not None and limit < len(ds):
        idx = np.random.RandomState(0).choice(len(ds), size=limit, replace=False)
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    if ('cifar10' in name or 'tinyimagenet' in name) and split=='train':
        classes = [c.replace('_',' ') for c in classes]
        return loader, classes
    else:
        return loader

#%%
# ---------- Feature extraction ----------
@torch.no_grad()
def extract_image_features(model, loader, device, backend="open_clip", l2_normalize=True, targets_out=False) -> np.ndarray:
    feats = []
    labels = []
    for imgs, targets in tqdm(loader):
        imgs = imgs.to(device, non_blocking=True)
        if backend == "open_clip":
            feats_i = model.encode_image(imgs)
        feats_i = feats_i.float()
        if l2_normalize:
            feats_i = nn.functional.normalize(feats_i, dim=-1)
        feats.append(feats_i.cpu().numpy())
        labels.append(targets.cpu().numpy())
    if targets_out:    
        return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)
    else:
        return np.concatenate(feats, axis=0)
#%%
# ---------- Text prototypes for CIFAR-100 superclasses ----------
# SUPERCLASSES = [
#     "aquatic mammals","fish","flowers","food containers","fruit and vegetables",
#     "household electrical devices","household furniture","insects","large carnivores",
#     "large man-made outdoor things","large natural outdoor scenes",
#     "large omnivores and herbivores","medium-sized mammals","non-insect invertebrates",
#     "people","reptiles","small mammals","trees","vehicles 1","vehicles 2"
# ]
PROMPTS = [
    "A photo of {}",
    "A low-resolution photo of {}",
    "A zoomed-in photo of {}",
    "A centered photo of {}",
]
K = 10
LIKERT_scale = ['. Certainty: almost certainly not.',
                '. Certainty: very unlikely.',
                '. Certainty: unlikely.',
                '. Certainty: somewhat unlikely.',
                '. Certainty: about as likely as not.',
                '. Certainty: somewhat likely.',
                '. Certainty: likely.',
                '. Certainty: very likely.',
                '. Certainty: almost certain.',
                '. Certainty: certain.']
#%%
PROMPTS_confidence = [f'{LIKERT_scale[k]}' for k in range(K)]
# PROMPTS_confidence = [f'. Confidence {k}/{K}.' for k in range(1,K+1)]

#%%
@torch.no_grad()
def make_text_prototypes(model, tokenizer, device, classes, backend="open_clip", l2_normalize=True):
    txt_embs = []
    for c in classes:
        texts = [p.format(c) for p in PROMPTS]
        tokens = tokenizer(texts)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens)
        tokens = tokens.to(device)
        emb = model.encode_text(tokens)
        emb = emb.float()
        emb = emb.mean(dim=0, keepdim=True)  # prompt ensembling
        if l2_normalize:
            emb = nn.functional.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu().numpy())
    return np.concatenate(txt_embs, axis=0)  # [20, D]

#%%
@torch.no_grad()
def make_text_confidence_prototypes(model, tokenizer, device, classes, backend="open_clip", l2_normalize=True):
    txt_embs = []
    L = len(PROMPTS)
    # k_range = range(1,K+1)
    for c in classes:
        texts = [p.format(c)+f'{k}' for k in PROMPTS_confidence for p in PROMPTS]
        tokens = tokenizer(texts)
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens)
        tokens = tokens.to(device)
        emb = model.encode_text(tokens)
        emb = emb.view(L,K,emb.shape[-1])
        emb = emb.float()
        emb = emb.mean(dim=0, keepdim=True)  # prompt ensembling
        if l2_normalize:
            emb = nn.functional.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu())
    return torch.concat(txt_embs, axis=0)  # [C, K, D]
#%%
# ---------- Proximity metrics ----------
def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # assumes both already L2-normalized
    return a @ b.T
#%%
def knn_distances(X: np.ndarray, Y: np.ndarray, k: int = 5) -> np.ndarray:
    # Euclidean in normalized space ~ sqrt(2(1-cos)), but we compute directly for clarity
    # Return mean distance to k-NN in X for each y in Y
    # For large X, consider faiss; here we use numpy (fits up to ~100k reasonably)
    # Compute cosine sims to use angular distance
    sims = pairwise_cosine(Y, X)  # [len(Y), len(X)]
    # top-k similarities
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    topk = np.take_along_axis(sims, idx, axis=1)
    # convert cosine to angular/Euclid on unit sphere
    # distance = sqrt(2 * (1 - cos))
    dists = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - topk)))
    return dists.mean(axis=1)
#%%
def fit_mahalanobis(X: np.ndarray, shrinkage: float = 0.01):
    # Fit global Gaussian N(mu, Sigma); return inv-cov for Mahalanobis
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # shrinkage covariance: (1 - a) * S + a * I * trace(S)/d
    S = Xc.T @ Xc / (len(X) - 1)
    d = S.shape[0]
    trace = np.trace(S) / d
    Sigma = (1 - shrinkage) * S + shrinkage * trace * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma + 1e-6 * np.eye(d))
    return mu, Sigma_inv

def mahalanobis_score(mu: np.ndarray, Sigma_inv: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Zc = Z - mu
    # Mahalanobis distance squared
    m2 = np.einsum("nd,dd,nd->n", Zc, Sigma_inv, Zc)
    return m2  # lower = closer
#%%
def fid_from_features(X: np.ndarray, Y: np.ndarray) -> float:
    # FID = ||mu_x - mu_y||^2 + Tr(Sigma_x + Sigma_y - 2*sqrt(Sigma_x Sigma_y))
    from scipy.linalg import sqrtm
    mu1, mu2 = X.mean(axis=0), Y.mean(axis=0)
    S1 = np.cov(X, rowvar=False)
    S2 = np.cov(Y, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(S1 @ S2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(S1 + S2 - 2 * covmean)
    return float(fid)
#%%
def kid_mmd(X: np.ndarray, Y: np.ndarray, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0,
            n_subsets: int = 100, subset_size: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """
    KID: polynomial kernel MMD^2 with unbiased estimator and subset averaging.
    Returns mean and std over subsets.
    """
    rng = np.random.RandomState(seed)
    n_x, n_y = len(X), len(Y)
    if gamma is None:
        # heuristic: 1 / dim
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
        # unbiased MMD^2
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        m = Kxx.shape[0]
        n = Kyy.shape[0]
        mmd2 = (Kxx.sum() / (m*(m-1))) + (Kyy.sum() / (n*(n-1))) - 2.0 * (Kxy.mean())
        vals.append(mmd2)
    return float(np.mean(vals)), float(np.std(vals))
#%%
def coverage_at_r(X_id: np.ndarray, Y: np.ndarray, r_quantile: float = 0.95, k: int = 5) -> float:
    # radius r = r_quantile of ID k-NN distances (to ID)
    id_knn = knn_distances(X_id, X_id, k=k)
    r = np.quantile(id_knn, r_quantile)
    y_knn = knn_distances(X_id, Y, k=k)
    return float((y_knn <= r).mean())

# ---------- Superclass-aware metrics ----------
def superclass_centroids(X_id: np.ndarray, y_super: np.ndarray, n_super: int = 20) -> np.ndarray:
    cents = []
    for s in range(n_super):
        mask = (y_super == s)
        cents.append(X_id[mask].mean(axis=0, keepdims=True))
    cents = np.concatenate(cents, axis=0)
    # Normalize since image features were normalized
    cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)
    return cents

def nearest_superclass_distance(X: np.ndarray, cents: np.ndarray) -> np.ndarray:
    # cosine→angular distance
    sims = X @ cents.T
    best = sims.max(axis=1)
    d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - best)))
    return d  # lower = closer to some superclass
#%%
def text_alignment_scores(X: np.ndarray, Tproto: np.ndarray) -> np.ndarray:
    # cosine similarity to best text prototype
    sims = X @ Tproto.T
    return sims.max(axis=1), sims.argmax(axis=1)  # higher = better alignment

#%%
def text_confidence_alignment_scores(X: np.ndarray, Tproto: np.ndarray, Tproto_confidence: np.ndarray) -> np.ndarray:
    # cosine similarity to best text prototype
    sims = X @ Tproto.T
    class_idx = sims.argmax(axis=1)
    sims_confidence = X @ Tproto_confidence.T
    sims_confidence_reshaped = sims_confidence.reshape(X.shape[0],Tproto.shape[0],K)
    class_confidence_list = [x[class_idx[k],:] for k,x in enumerate(sims_confidence_reshaped)]
    class_confidence = np.vstack(class_confidence_list)
    return class_confidence.max(axis=1), class_idx, class_confidence.argmax(axis=1)  # higher = better alignment
#%%
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
#%%
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    parser.add_argument('--iid_dataset', type=str, required=True, help="IID data set name", choices=['cifar10','cifar100','supercifar100','tinyimagenet'])
    args = parser.parse_args()
    iid_dataset = args.iid_dataset
    run(iid_dataset)


