#%%
import os
import joblib
from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Tuple, List, Union, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _sigmoid_calibration
from sklearn.metrics import log_loss
import pandas as pd
from src.calibration_metric import ece
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("true", "1", "t", "yes", "y"):
        return True
    if v in ("false", "0", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}. Use True/False.")

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "supercifar100", "tinyimagenet"],
        help="Dataset choice."
    )
    p.add_argument(
        "--vit", 
        action="store_true", 
        help="Set this flag if using ViT model results"
        )
    p.add_argument(
        "--scores-dir", 
        type=str, 
        default="scores_final", 
        help="Directory to save final scores"
        )
    return p.parse_args()

def resolve_dataset_name(dataset: str, vit: bool) -> str:
    if dataset == "cifar100" and vit:
        return "cifar100_modelvit_bbvit_lr0.01"
    if dataset == "supercifar100" and vit:
        return "super_cifar100"
    if dataset == "supercifar100" and not vit:
        return "supercifar"
    if dataset == "tinyimagenet":
        return "tiny-imagenet-200"
    return dataset

def read_confids_vit(dataset,set_name):
    logger.info(f"Reading ViT results for dataset: {dataset}, set: {set_name}")
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/vit/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir() if f'{dataset}_' in j.parts[-1]]
    # print(len(folders_in_exp_dir))
    if 'super' in  dataset:
        lst_idx = [2,3,7,6,8]
    else:
        lst_idx = [1,2,6,5,7]
    res = []
    
    logger.debug(f"Found {len(folders_in_exp_dir)} experiment folders.")
    
    for k in range(len(folders_in_exp_dir)):
        model_description = folders_in_exp_dir[k].parts[-1].split('_')
        files_in_folder = [j for j in folders_in_exp_dir[k].joinpath('analysis').glob(f'*confids*{set_name}.csv')]
        # print(files_in_folder)
        for i in range(len(files_in_folder)):
            exp_description = files_in_folder[i].parts[-1].split('_')
            stats_df = pd.read_csv(files_in_folder[i], index_col=0)
            stats_df[['model','network','drop_out','run','reward']] = [model_description[j] for j in lst_idx]
            stats_df[['RankWeight','RankFeat','ASH']] = [ *exp_description[1:3], exp_description[3]]
            stats_df['filepath'] = files_in_folder[i]
            res.append(stats_df)
            
    if not res:
        logger.warning(f"No results found for dataset: {dataset}, set: {set_name}")
        return pd.DataFrame()
        
    results = pd.concat(res)
    # print(results['run'])
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    logger.success(f"Successfully read {len(results)} rows for {dataset} - {set_name}")
    return results

def read_confids(dataset,set_name):
    logger.info(f"Reading results for dataset: {dataset}, set: {set_name}")
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/{dataset}_paper_sweep/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir()]
    res = []
    
    logger.debug(f"Found {len(folders_in_exp_dir)} experiment folders.")

    for k in range(len(folders_in_exp_dir)):
        model_description = folders_in_exp_dir[k].parts[-1].split('_')
        files_in_folder = [j for j in folders_in_exp_dir[k].joinpath('analysis').glob(f'*confids*{set_name}.csv')]
        for i in range(len(files_in_folder)):
            exp_description = files_in_folder[i].parts[-1].split('_')
            stats_df = pd.read_csv(files_in_folder[i], index_col=0)
            stats_df[['model','network','drop_out','run','reward']] = model_description
            stats_df[['RankWeight','RankFeat','ASH']] = [ *exp_description[1:3], exp_description[3]]
            stats_df['filepath'] = files_in_folder[i]
            res.append(stats_df)
            
    if not res:
        logger.warning(f"No results found for dataset: {dataset}, set: {set_name}")
        return pd.DataFrame()

    results = pd.concat(res)
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    logger.success(f"Successfully read {len(results)} rows for {dataset} - {set_name}")
    return results
#%%
# ==========================================
# 1. Unified Calibrator Class
# ==========================================
def finite_xy(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    m = np.isfinite(scores) & np.isfinite(labels)
    return scores[m], labels[m]

def _to_1d_scores(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X
    if X.ndim == 2 and X.shape[1] == 1:
        return X[:, 0]
    raise ValueError(f"Expected X to be shape (n,) or (n,1); got {X.shape}.")

class SklearnSigmoidCalibrator:
    """
    Wrapper for scikit-learn's internal _sigmoid_calibration.
    Fits A and B in: p = 1 / (1 + exp(A * score + B))
    Includes Platt's label smoothing natively.
    """
    def __init__(self):
        self.A_ = None
        self.B_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        scores = X.flatten()
        labels = y.flatten()
        
        # Scikit-learn's internal function handles the label smoothing and optimization
        self.A_, self.B_ = _sigmoid_calibration(scores, labels, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.A_ is None or self.B_ is None:
            raise RuntimeError("Calibrator is not fitted yet.")
            
        scores = X.flatten()
        z = self.A_ * scores + self.B_
        
        # Clip to prevent numerical overflow in exp()
        z = np.clip(z, -700, 700)
        return 1.0 / (1.0 + np.exp(z))

    def save_params(self, filepath: str) -> None:
        """Save fitted parameters A_ and B_ to disk."""
        if self.A_ is None or self.B_ is None:
            raise RuntimeError("Cannot save params: calibrator is not fitted yet.")
        payload = {"A_": float(self.A_), "B_": float(self.B_)}
        joblib.dump(payload, filepath)

    def load_params(self, filepath: str) -> "SklearnSigmoidCalibrator":
        """Load parameters A_ and B_ from disk into this instance."""
        payload = joblib.load(filepath)
        if "A_" not in payload or "B_" not in payload:
            raise ValueError(f"Invalid parameter file: missing A_ and/or B_. Got keys={list(payload.keys())}")
        self.A_ = float(payload["A_"])
        self.B_ = float(payload["B_"])
        return self

class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip')
    def fit(self, X, y, sample_weight=None):
        self.iso.fit(X, y, sample_weight=sample_weight)
        return self
    def predict_proba(self, X):
        return self.iso.predict(X)
    def save_params(self, filepath: str) -> None:
        """Save the fitted isotonic model to disk."""
        if not hasattr(self.iso, "X_thresholds_") or not hasattr(self.iso, "y_thresholds_"):
            raise RuntimeError("Cannot save params: calibrator is not fitted yet.")
        joblib.dump(self.iso, filepath)

    def load_params(self, filepath: str) -> "IsotonicCalibrator":
        """Load a fitted isotonic model from disk into this instance."""
        iso = joblib.load(filepath)
        if not isinstance(iso, IsotonicRegression):
            raise ValueError(f"Invalid file contents: expected IsotonicRegression, got {type(iso)}")
        self.iso = iso
        return self

class BetaCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs')
        self.min_s, self.max_s = 0.0, 1.0
    def _transform(self, X):
        scaled = (X - self.min_s) / (self.max_s - self.min_s + 1e-12)
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return np.column_stack([np.log(scaled), -np.log(1 - scaled)])
    def fit(self, X, y, sample_weight=None):
        self.min_s, self.max_s = np.min(X), np.max(X)
        self.lr.fit(self._transform(X), y, sample_weight=sample_weight)
        return self
    def predict_proba(self, X):
        return self.lr.predict_proba(self._transform(X))[:, 1]
    def save_params(self, filepath: str) -> None:
        """Save fitted parameters (min/max scaling + logistic regression) to disk."""
        if not hasattr(self.lr, "coef_") or not hasattr(self.lr, "intercept_"):
            raise RuntimeError("Cannot save params: calibrator is not fitted yet.")
        payload = {
            "min_s": float(self.min_s),
            "max_s": float(self.max_s),
            "lr": self.lr,  # joblib can serialize sklearn estimators safely
        }
        joblib.dump(payload, filepath)

    def load_params(self, filepath: str) -> "BetaCalibrator":
        """Load fitted parameters from disk into this instance."""
        payload = joblib.load(filepath)
        for k in ("min_s", "max_s", "lr"):
            if k not in payload:
                raise ValueError(f"Invalid parameter file: missing {k}. Got keys={list(payload.keys())}")

        lr = payload["lr"]
        if not isinstance(lr, LogisticRegression):
            raise ValueError(f"Invalid file contents: expected LogisticRegression, got {type(lr)}")
        if not hasattr(lr, "coef_") or not hasattr(lr, "intercept_"):
            raise ValueError("Loaded LogisticRegression does not appear to be fitted.")

        self.min_s = float(payload["min_s"])
        self.max_s = float(payload["max_s"])
        self.lr = lr
        return self

class Calibrator:
    def __init__(self, method='sigmoid', cv=None, shuffle=True, random_state=42):
        self.method = method
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.models_ = []
        
    def _get_base_model(self):
        if self.method == 'sigmoid': return SklearnSigmoidCalibrator()
        if self.method == 'isotonic': return IsotonicCalibrator()
        if self.method == 'beta': return BetaCalibrator()

    def fit(self, X, y, sample_weight=None):
        scores = _to_1d_scores(X).astype(float)
        y = np.asarray(y)
        self.models_ = []
        self.probs_cv = []
        self.labels_cv = []
        if self.cv is None or self.cv <= 1:
            model = self._get_base_model().fit(scores, y, sample_weight=sample_weight)
            self.models_.append(model)
            return self
            
        cv_splitter = StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)
        for train_idx, test_idx in cv_splitter.split(scores, y):
            s_tr, y_tr = scores[train_idx], y[train_idx]
            s_test, y_test = scores[test_idx], y[test_idx]
            sw_tr = sample_weight[train_idx] if sample_weight is not None else None
            self.models_.append(self._get_base_model().fit(s_tr, y_tr, sample_weight=sw_tr))
            self.probs_cv.append(self.models_[-1].predict_proba(s_test))
            self.labels_cv.append(y_test)
        return self

    def predict_proba(self, X):
        scores = _to_1d_scores(X).astype(float)
        fold_preds = np.column_stack([model.predict_proba(scores) for model in self.models_])
        return fold_preds.mean(axis=1)
    
    def get_cv_probs_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get concatenated out-of-fold calibrated probabilities and corresponding labels."""
        if not self.probs_cv or not self.labels_cv:
            raise RuntimeError("No CV predictions available. Ensure fit() was called with cv > 1.")
        return np.concatenate(self.probs_cv), np.concatenate(self.labels_cv)

    def save_params(self, filepath: str) -> None:
        """Save calibrator configuration + fitted fold models to disk."""
        if not self.models_:
            raise RuntimeError("Cannot save params: calibrator is not fitted yet.")

        # Rely on each base model's own save/load schema via joblib pickling.
        # (These are your wrappers; they contain only small sklearn objects/params.)
        payload = {
            "method": self.method,
            "cv": self.cv,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "models_": self.models_,
        }
        joblib.dump(payload, filepath)

    def load_params(self, filepath: str) -> "Calibrator":
        """Load calibrator configuration + fitted fold models from disk into this instance."""
        payload = joblib.load(filepath)

        required = ("method", "cv", "shuffle", "random_state", "models_")
        for k in required:
            if k not in payload:
                raise ValueError(f"Invalid parameter file: missing {k}. Got keys={list(payload.keys())}")

        method = payload["method"]
        if method not in ("sigmoid", "isotonic", "beta"):
            raise ValueError(f"Invalid parameter file: unknown method={method!r}")

        models = payload["models_"]
        if not isinstance(models, list) or len(models) == 0:
            raise ValueError("Invalid parameter file: models_ must be a non-empty list.")

        # Basic type validation to catch wrong files early
        expected_cls = {
            "sigmoid": SklearnSigmoidCalibrator,
            "isotonic": IsotonicCalibrator,
            "beta": BetaCalibrator,
        }[method]
        for i, m in enumerate(models):
            if not isinstance(m, expected_cls):
                raise ValueError(
                    f"Invalid parameter file: models_[{i}] expected {expected_cls.__name__}, got {type(m)}"
                )

        self.method = method
        self.cv = payload["cv"]
        self.shuffle = bool(payload["shuffle"])
        self.random_state = payload["random_state"]
        self.models_ = models
        return self

# ==========================================
# 2. Core Mathematical Functions
# ==========================================

def extract_k_and_pood(mixed_probs_list: List[np.ndarray], mixed_labels_list: List[np.ndarray], bound_type='l1'):
    """Extracts base metrics K and p_I per model for plotting continuous alpha curves."""
    k_list, pi_list = [], []
    for probs, labels in zip(mixed_probs_list, mixed_labels_list):
        probs, labels = np.asarray(probs), np.asarray(labels)
        mask_id = (labels == 1)
        mask_ood = (labels == 0)
        if bound_type == 'l1':
            p_c = np.mean(probs[mask_id]) if np.sum(mask_id) > 0 else np.nan
            p_i = np.mean(probs[mask_ood]) if np.sum(mask_ood) > 0 else np.nan
            k_list.append(1.0 - p_c)
            pi_list.append(p_i)
        elif bound_type == 'l2':
            k_sq = np.mean((1.0 - probs[mask_id])**2) if np.sum(mask_id) > 0 else np.nan
            p_ood_sq = np.mean((probs[mask_ood])**2) if np.sum(mask_ood) > 0 else np.nan
            k_list.append(k_sq)
            pi_list.append(p_ood_sq)
    return np.array(k_list), np.array(pi_list)

def calc_l1_bounds_from_mixed(mixed_probs_list: List[np.ndarray], mixed_labels_list: List[np.ndarray], fixed_alpha: float = None) -> np.ndarray:
    bounds = []
    for probs, labels in zip(mixed_probs_list, mixed_labels_list):
        probs, labels = np.asarray(probs), np.asarray(labels)
        mask_id = (labels == 1)
        mask_ood = (labels == 0)
        n_id, n_ood = np.sum(mask_id), np.sum(mask_ood)
        
        if n_id == 0 or n_ood == 0:
            bounds.append(np.nan)
            continue
            
        alpha = fixed_alpha if fixed_alpha is not None else (n_ood / n_id)
        p_c = np.mean(probs[mask_id])
        p_i = np.mean(probs[mask_ood])
        
        bound = (1 / (1 + alpha)) * (1.0 - p_c) + (alpha / (1 + alpha)) * p_i
        bounds.append(bound)
    return np.array(bounds)

def calc_l2_bounds_from_mixed(mixed_probs_list: List[np.ndarray], mixed_labels_list: List[np.ndarray], fixed_alpha: float = None) -> np.ndarray:
    bounds = []
    for probs, labels in zip(mixed_probs_list, mixed_labels_list):
        probs, labels = np.asarray(probs), np.asarray(labels)
        mask_id = (labels == 1)
        mask_ood = (labels == 0)
        n_id, n_ood = np.sum(mask_id), np.sum(mask_ood)
        
        if n_id == 0 or n_ood == 0:
            bounds.append(np.nan)
            continue
            
        alpha = fixed_alpha if fixed_alpha is not None else (n_ood / n_id)
        k_sq = np.mean((1.0 - probs[mask_id])**2)
        p_ood_sq = np.mean((probs[mask_ood])**2)
        
        bound = np.sqrt((1 / (1 + alpha)) * k_sq + (alpha / (1 + alpha)) * p_ood_sq)
        bounds.append(bound)
    return np.array(bounds)

def get_mean_ci(data_1d, conf=0.95):
    """Calculates mean and 95% CI half-width, ignoring NaNs."""
    clean_data = data_1d[~np.isnan(data_1d)]
    n = len(clean_data)
    if n < 2: return np.nan, np.nan
    mean = np.mean(clean_data)
    se = stats.sem(clean_data)
    ci_h = se * stats.t.ppf((1 + conf) / 2., n-1)
    return mean, ci_h
# ==========================================
# 3. The Four Evaluation Routes
# ==========================================
def plot_route_1_severity_stats(data_dict: Dict, fixed_alpha: float = None):
    """Plots Route 1: Shift Severity (Mean L1 Bound ± 95% CI) with optional fixed alpha.
      Also performs paired t-tests between best and second-best methods per dataset.
    Args:
        data_dict: Nested dict of shape {method: {dataset: {'probs': [...], 'labels': [...]}}} (method correspond to different calibrators) 
        fixed_alpha: If provided, uses this fixed alpha for all bounds instead of empirical alpha. Useful for isolating method performance from dataset-specific contamination ratios.
    """
    methods = list(data_dict.keys())
    datasets = list(data_dict[methods[0]].keys())
    
    plt.figure(figsize=(9, 6))
    raw_bounds_tracker = {d: [] for d in datasets}

    for method in methods:
        means, cis = [], []
        for ds in datasets:
            probs = data_dict[method][ds]['probs']
            labels = data_dict[method][ds]['labels']
            
            bounds = calc_l1_bounds_from_mixed(probs, labels, fixed_alpha=fixed_alpha)
            m, ci = get_mean_ci(bounds)
            means.append(m)
            cis.append(ci)
            raw_bounds_tracker[ds].append((bounds, method))
            
        plt.errorbar(datasets, means, yerr=cis, marker='o', capsize=5, label=method, linewidth=2)
    
    for i, ds in enumerate(datasets):
        # Drop NaNs for statistical testing
        sorted_methods = sorted(raw_bounds_tracker[ds], key=lambda x: np.nanmean(x[0]))
        best_bounds = sorted_methods[0][0][~np.isnan(sorted_methods[0][0])]
        second_best_bounds = sorted_methods[1][0][~np.isnan(sorted_methods[1][0])]
        
        if len(best_bounds) > 1 and len(second_best_bounds) > 1:
            _, p_val = stats.ttest_rel(best_bounds, second_best_bounds)
            if p_val < 0.05:
                plt.text(i, np.nanmean(best_bounds) - 0.02, "*", fontsize=16, ha='center', color='black')

    alpha_str = f"fixed_alpha={fixed_alpha}" if fixed_alpha else "empirical alpha"
    plt.title(f"Route 1: Shift Severity (Mean L1 Bound ± 95% CI, {alpha_str})\n* denotes p < 0.05 vs 2nd best")
    plt.ylabel("L1 ECE Upper Bound")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_route_2_alpha_stats(data_dict: Dict, target_dataset: str, max_alpha: float = 5.0):
    alphas = np.linspace(0, max_alpha, 50)
    plt.figure(figsize=(9, 6))
    
    for method in data_dict.keys():
        probs = data_dict[method][target_dataset]['probs']
        labels = data_dict[method][target_dataset]['labels']
        
        # Extract base metrics per model to construct the curve
        k_arr, pi_arr = extract_k_and_pi(probs, labels)
        
        mean_curve, lower_bound, upper_bound = [], [], []
        aub_per_model = np.zeros(len(k_arr))
        
        for a in alphas:
            bounds_5 = (1 / (1 + a)) * k_arr + (a / (1 + a)) * pi_arr
            m, ci = get_mean_ci(bounds_5)
            mean_curve.append(m)
            lower_bound.append(m - ci)
            upper_bound.append(m + ci)
            aub_per_model += bounds_5 * (max_alpha / 50)
            
        plt.plot(alphas, mean_curve, label=f"{method} (Mean AUB: {np.mean(aub_per_model):.2f})", linewidth=2)
        plt.fill_between(alphas, lower_bound, upper_bound, alpha=0.2)

    plt.title(f"Route 2: Alpha-Robustness on {target_dataset}")
    plt.xlabel("Contamination Ratio (alpha = OOD/ID)")
    plt.ylabel("L1 ECE Upper Bound")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_route_3_pareto_stats(data_dict: Dict, target_dataset: str):
    plt.figure(figsize=(8, 8))
    
    for method in data_dict.keys():
        probs = data_dict[method][target_dataset]['probs']
        labels = data_dict[method][target_dataset]['labels']
        
        k_arr, pi_arr = extract_k_and_pi(probs, labels)
        
        k_mean, k_ci = get_mean_ci(k_arr)
        p_mean, p_ci = get_mean_ci(pi_arr)
        
        line = plt.errorbar(k_mean, p_mean, xerr=k_ci, yerr=p_ci, fmt='o', capsize=4, label=method, markersize=8)
        plt.annotate(method, (k_mean, p_mean), xytext=(8, 8), textcoords='offset points', color=line[0].get_color())

    plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='K = p_OOD')
    plt.title(f"Route 3: Pareto Trade-off with 95% CI Crosshairs ({target_dataset})")
    plt.xlabel("Mean ID Baseline Uncertainty (K)")
    plt.ylabel("Mean Average OOD Confidence (p_OOD)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_route_4_variance_stats(data_dict: Dict, method: str, fixed_alpha: float = None):
    datasets = list(data_dict[method].keys())
    l1_means, l1_cis, l2_means, l2_cis = [], [], [], []
    
    for ds in datasets:
        probs = data_dict[method][ds]['probs']
        labels = data_dict[method][ds]['labels']
        
        b_l1 = calc_l1_bounds_from_mixed(probs, labels, fixed_alpha=fixed_alpha)
        b_l2 = calc_l2_bounds_from_mixed(probs, labels, fixed_alpha=fixed_alpha)
        
        m1, c1 = get_mean_ci(b_l1)
        m2, c2 = get_mean_ci(b_l2)
        
        l1_means.append(m1); l1_cis.append(c1)
        l2_means.append(m2); l2_cis.append(c2)

    x = np.arange(len(datasets))
    width = 0.35
    
    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, l1_means, width, yerr=l1_cis, capsize=5, label='L1 Bound (Linear)', color='skyblue')
    plt.bar(x + width/2, l2_means, width, yerr=l2_cis, capsize=5, label='L2 Bound (RMSCE)', color='salmon')
    
    alpha_str = f"fixed_alpha={fixed_alpha}" if fixed_alpha else "empirical alpha"
    plt.title(f"Route 4: Variance Gap for {method}\n(Mean ± 95% CI, {alpha_str})")
    plt.ylabel("Upper Bound Value")
    plt.xticks(x, datasets)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()