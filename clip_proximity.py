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

# %%
# class TinyImagenet(tinyimagenet.TinyImageNet):
#     def __init__(
#         self,
#         root: str,
#         split: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         ) -> None:
#         super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)
    
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target
    
#     def __len__(self):
#         return len(self.imgs)

#%%
# ---------- CLIP loader (open_clip preferred; fallback to openai/clip) ----------
def load_clip(model_name: str = "ViT-B-32", pretrained: Optional[str] = None, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = None
    try:
        import open_clip
        if pretrained is None:
            # strong general model; change if you like "laion2b_s34b_b79k" or "openai"
            pretrained = "laion2b_s34b_b79k"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, preprocess, tokenizer, device, "open_clip"
    except Exception as e:
        warnings.warn(f"open_clip unavailable ({e}); falling back to openai/clip.")
        import clip
        model, preprocess = clip.load(model_name, device=device)
        def tokenizer(texts: List[str]):
            return clip.tokenize(texts)
        model.eval()
        return model, preprocess, tokenizer, device, "clip"

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
        else:
            feats_i = model.encode_image(imgs)  # openai/clip compat
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
    "a photo of {}",
    "a low-resolution photo of {}",
    "a zoomed-in photo of {}",
    "a centered photo of {}",
]

@torch.no_grad()
def make_text_prototypes(model, tokenizer, device, classes, backend="open_clip", l2_normalize=True):
    txt_embs = []
    for c in classes:
        texts = [p.format(c) for p in PROMPTS]
        if backend == "open_clip":
            import open_clip
            tokens = tokenizer(texts)
            if not torch.is_tensor(tokens):
                tokens = torch.tensor(tokens)
            tokens = tokens.to(device)
            emb = model.encode_text(tokens)
        else:
            import clip
            tokens = tokenizer(texts).to(device)
            emb = model.encode_text(tokens)
        emb = emb.float()
        emb = emb.mean(dim=0, keepdim=True)  # prompt ensembling
        if l2_normalize:
            emb = nn.functional.normalize(emb, dim=-1)
        txt_embs.append(emb.cpu().numpy())
    return np.concatenate(txt_embs, axis=0)  # [20, D]
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
def class_centroids(X_id: np.ndarray, y_id: np.ndarray, ) -> np.ndarray:
    cents = []
    unique_y = np.unique(y_id)
    for s in unique_y:
        mask = (y_id == s)
        cents.append(X_id[mask].mean(axis=0, keepdims=True))
    cents = np.concatenate(cents, axis=0)
    # Normalize since image features were normalized
    cents = cents / np.linalg.norm(cents, axis=1, keepdims=True)
    return cents

def nearest_class_distance(X: np.ndarray, cents: np.ndarray) -> np.ndarray:
    # cosine→angular distance
    sims = X @ cents.T
    best = sims.max(axis=1)
    d = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - best)))
    return d  # lower = closer to some class
#%%
def text_alignment_scores(X: np.ndarray, Tproto: np.ndarray) -> np.ndarray:
    # cosine similarity to best text prototype
    sims = X @ Tproto.T
    return sims.max(axis=1)  # higher = better alignment
#%%
# ---------- CIFAR-100 coarse labels helper (SuperCIFAR) ----------
# Torchvision's CIFAR100 returns fine labels. Provide your own coarse labels as array if needed.
# Here we include a compact mapping (index per sample) if you have it; otherwise set to None and skip centroids.
def load_cifar100_coarse_indices(split="train"):
    # If you already have (fine->coarse) mapping per sample, replace this with your mapping loader.
    # Placeholder returns None; many users maintain their own file.
    return None
#%%
# ---------- Runner (example) ----------
def run(iid_dataset:str):
    ti_condition = 'tinyimagenet' if (iid_dataset=='cifar10' or iid_dataset=='cifar100' or iid_dataset=='supercifar100') else 'cifar10'
    c100_condition = 'cifar100' if (iid_dataset=='cifar10' or iid_dataset=='tinyimagenet') else 'cifar10'
    logger.info('Loading CLIP model...')
    model, preprocess, tokenizer, device, backend = load_clip(model_name="ViT-B-32")
    logger.info(f'Loading {iid_dataset} data set...')
    # 1) Load ID = SuperCIFAR (CIFAR100 train as ID features)
    id_loader, classes = make_loader(iid_dataset, split="train", preprocess=preprocess, batch_size=256)
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
        print(f"[{name}] {json.dumps(results[name], indent=2)}")

    # Optional: save results
    path = f"clip_proximity_{iid_dataset}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")
#%%
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    parser.add_argument('--iid_dataset', type=str, required=True, help="IID data set name", choices=['cifar10','cifar100','supercifar100','tinyimagenet'])
    args = parser.parse_args()
    iid_dataset = args.iid_dataset
    run(iid_dataset)


#%%

# %%
