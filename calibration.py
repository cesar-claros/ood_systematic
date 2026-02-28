#%%
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import calibration_metric
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import _sigmoid_calibration  # private API
from src.utils_calibration import *
# import optbinning
#%%

# def _to_1d_scores(X: np.ndarray) -> np.ndarray:
#     """
#     Convert X to a 1D array of scores.
#     - If X is (n,), use as-is.
#     - If X is (n,1), squeeze.
#     - Otherwise error (because we don't know which column is your score).
#     """
#     X = np.asarray(X)
#     if X.ndim == 1:
#         return X
#     if X.ndim == 2 and X.shape[1] == 1:
#         return X[:, 0]
#     raise ValueError(
#         f"Expected X to be shape (n,) or (n,1) for a single score per sample; got {X.shape}."
#     )


# def _sigmoid_predict(scores_1d: np.ndarray, A: float, B: float) -> np.ndarray:
#     # Platt sigmoid: p = 1 / (1 + exp(A * f + B))
#     z = A * scores_1d + B
#     # numerical stability: avoid overflow in exp for extreme z
#     z = np.clip(z, -700, 700)
#     return 1.0 / (1.0 + np.exp(z))


# @dataclass(frozen=True)
# class SigmoidCVEnsemble:
#     """Holds fold-wise sigmoid calibration parameters and can ensemble them."""
#     fold_params: List[Tuple[float, float]]  # [(A1,B1), (A2,B2), ...]
#     cv: StratifiedKFold

#     def predict_proba_all(self, X) -> np.ndarray:
#         """
#         Returns calibrated probabilities from each fold calibrator.
#         Shape: (n_samples, n_folds)
#         """
#         scores = _to_1d_scores(X).astype(float)
#         probs = np.empty((scores.shape[0], len(self.fold_params)), dtype=float)
#         for j, (A, B) in enumerate(self.fold_params):
#             probs[:, j] = _sigmoid_predict(scores, A, B)
#         return probs

#     def predict_proba_mean(self, X) -> np.ndarray:
#         """Mean-ensemble probability across all fold calibrators. Shape: (n_samples,)"""
#         return self.predict_proba_all(X).mean(axis=1)


# def fit_sigmoid_calibration_cv_ensemble(
#     X,
#     y,
#     *,
#     n_splits: int = 5,
#     shuffle: bool = True,
#     random_state: int = 42,
#     sample_weight: Optional[np.ndarray] = None,
#     return_oof: bool = True,
# ) -> Tuple[SigmoidCVEnsemble, Optional[np.ndarray]]:
#     """
#     Fit fold-wise sigmoid calibrators using stratified CV and return an ensemble object.

#     Parameters
#     ----------
#     X : array-like, shape (n,) or (n,1)
#         Uncalibrated real-valued scores (logit/margin/energy/etc.).
#     y : array-like, shape (n,)
#         Binary labels.
#     sample_weight : array-like, shape (n,), optional
#         Sample weights for calibration fitting.
#     return_oof : bool
#         If True, returns out-of-fold calibrated probs.

#     Returns
#     -------
#     ensemble : SigmoidCVEnsemble
#         Contains fold-wise (A,B) parameters and methods to ensemble predictions.
#     oof_probs : np.ndarray or None
#         Out-of-fold calibrated probabilities (only if return_oof=True).
#     """
#     scores = _to_1d_scores(X).astype(float)
#     y = np.asarray(y)

#     if y.ndim != 1:
#         raise ValueError("y must be 1D.")
#     classes = np.unique(y)
#     if classes.size != 2:
#         raise ValueError(f"y must be binary; got classes={classes}.")

#     if scores.shape[0] != y.shape[0]:
#         raise ValueError("X and y must have the same number of samples.")

#     if sample_weight is not None:
#         sample_weight = np.asarray(sample_weight)
#         if sample_weight.shape[0] != y.shape[0]:
#             raise ValueError("sample_weight must have shape (n_samples,)")

#     cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

#     fold_params: List[Tuple[float, float]] = []
#     oof_probs = np.empty(scores.shape[0], dtype=float) if return_oof else None

#     for train_idx, test_idx in cv.split(scores, y):
#         s_tr, y_tr = scores[train_idx], y[train_idx]
#         s_te = scores[test_idx]

#         if sample_weight is None:
#             A, B = _sigmoid_calibration(s_tr, y_tr)
#         else:
#             A, B = _sigmoid_calibration(s_tr, y_tr, sample_weight=sample_weight[train_idx])

#         fold_params.append((float(A), float(B)))

#         if return_oof:
#             oof_probs[test_idx] = _sigmoid_predict(s_te, A, B)

#     ensemble = SigmoidCVEnsemble(fold_params=fold_params, cv=cv)
#     return ensemble, oof_probs
# %%
# X: your uncalibrated scores (1D) OR (n,2) probs with positive class in column 1
# y: binary labels
dataset = 'cifar10'
run = 1
model = 'confidnet'
val = pd.read_csv(f'/work/cniel/sw/FD_Shifts/project/experiments/{dataset}_paper_sweep/{model}_bbvgg13_do0_run{run}_rew2.2/analysis/confids_RW0_RF0_ASHNone_iid_val.csv', index_col=0)
test = pd.read_csv(f'/work/cniel/sw/FD_Shifts/project/experiments/{dataset}_paper_sweep/{model}_bbvgg13_do0_run{run}_rew2.2/analysis/confids_RW0_RF0_ASHNone_iid_test.csv', index_col=0)
ood_c100 = pd.read_csv(f'/work/cniel/sw/FD_Shifts/project/experiments/{dataset}_paper_sweep/{model}_bbvgg13_do0_run{run}_rew2.2/analysis/confids_RW0_RF0_ASHNone_ood_sncs_c100.csv', index_col=0)
#%%
import matplotlib.pyplot as plt
conf_fnc = 'Residual'
plt.hist(val[conf_fnc][val['residuals']==0].values, color='blue', bins=50)
plt.hist(val[conf_fnc][val['residuals']==1].values, color='orange', bins=50, alpha=0.5)
print(val[conf_fnc].describe())
#%%
plt.hist(ood_c100[conf_fnc][ood_c100['residuals']==0].values, color='blue', bins=50)
plt.hist(ood_c100[conf_fnc][ood_c100['residuals']==1].values, color='orange', bins=50, alpha=0.5)
print(ood_c100[conf_fnc].describe())
#%%
cal_method = 'isotonic'
s_val = val[conf_fnc].values
y_val = 1-val['residuals'].values.astype(int)  # convert residual
s_test = test[conf_fnc].values
y_test = 1-test['residuals'].values.astype(int)
s_ood = ood_c100[conf_fnc].values
y_ood = 1-ood_c100['residuals'].values.astype(int)
calibrator = Calibrator(method=cal_method.lower(), cv=5, random_state=42)
calibrator.fit(s_val, y_val)
p_val = calibrator.predict_proba(s_val)
p_test = calibrator.predict_proba(s_test)
p_ood = calibrator.predict_proba(s_ood)
#%%
optb_kwargs={"solver": 'mip', 
            "divergence": 'neg_brier', 
            "max_n_prebins": 50,
            "min_prebin_size": 5e-3,
            "min_bin_size": 5e-3,
            "max_bin_size": 0.8,
            # "min_n_bins": 4,
            # "monotonic_trend": "auto",
            # "max_pvalue": 0.05,
            "verbose": True,
            }
test_stat = 'binom_fast'
tce_binner = calibration_metric.TCEBinner(siglevel=0.05, optb_kwargs=optb_kwargs)
tce_binner.fit(p_val, y_val)
tce_binner.tce(p_val, y_val, test=test_stat)
tce_binner.tce(p_test, y_test, test=test_stat)
#%%
plt.hist(p_val[y_val==0], color='blue', bins=50)
plt.hist(p_val[y_val==1], color='orange', bins=50, alpha=0.5)
#%%
plt.hist(p_test[y_test==0], color='blue', bins=50)
plt.hist(p_test[y_test==1], color='orange', bins=50, alpha=0.5)
#%%
plt.hist(p_ood[y_ood==0], color='blue', bins=50)
plt.hist(p_ood[y_ood==1], color='orange', bins=50, alpha=0.5)
#%%

#%%
X = val[conf_fnc].values
y = 1-val['residuals'].values.astype(int)
ensemble, oof_p = fit_sigmoid_calibration_cv_ensemble(X, y, n_splits=5)

# Fold-wise parameters:
print(ensemble.fold_params)  # [(A1,B1), (A2,B2), ...]
#%%
X_new = val[conf_fnc].values
# For new scores:
p_all  = ensemble.predict_proba_all(X_new)   # (n_samples, n_folds)
p_mean = ensemble.predict_proba_mean(X_new)  # (n_samples,)
#%%
plt.hist(p_mean[val['residuals']==0], color='blue', bins=50)
plt.hist(p_mean[val['residuals']==1], color='orange', bins=50, alpha=0.5)
# print(ood_c100[conf_fnc].describe())
#%%
calibration_metric.ece(p_mean, 1-val['residuals'], n_bin=20, mode='l1', savepath=False)
#%%
calibration_metric.tce_ttest(p_mean, 1-val['residuals'], 
                             strategy="optbinning", 
                             optb_kwargs={"solver": 'mip', 
                                          "divergence": 'log_score', 
                                          "max_n_prebins": 20,
                                        #   "min_n_bins":6,
                                        #   "min_bin_size": 0.05,
                                          "max_bin_size": 0.6,
                                        #   "gamma": 1e3,
                                          "verbose": True, 
                                          "monotonic_trend": "auto",
                                          "max_pvalue": 0.05,
                                          })
#%%
calibration_metric.tce(p_mean, 1-val['residuals'], 
                             strategy="optbinning", 
                             optb_kwargs={"solver": 'mip', 
                                          "divergence": 'neg_brier', 
                                          "max_n_prebins": 20,
                                        #   "min_n_bins":6,
                                          "max_bin_size": 0.6,
                                          "monotonic_trend": "auto",
                                          "max_pvalue": 0.05,
                                          })
#%%

#%%
import matplotlib.pyplot as plt
# %%
plt.scatter(X_new,1-test['residuals'])
# %%
# %%
calibration_metric.ece(p_mean, 1-ood_c100['residuals'], n_bin=10, mode='l1', savepath=False)
#%%
calibration_metric.ace(p_mean, 1-ood_c100['residuals'], n_bin=10, mode='l1', savepath=False)
# %%
calibration_metric.tce(p_mean, 1-test['residuals'], strategy="quantile", n_bin=30)
# %%
# %%
