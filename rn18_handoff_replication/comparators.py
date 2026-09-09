"""Nine frozen zero-shot comparators (plan v3, carried v2 section 11.1),
fit ONLY on the 20 historical VGG dropout-off backbones under the
primary measurement contract; target predictions are fixed before any
ResNet gap is inspected.

1 always Energy; 2 always CTM; 3 VGG KID isotonic per source; 4 VGG FD
isotonic per source; 5 VGG source-shift mean; 6 VGG geometry-severity
ridge (intercept, 3 source indicators, d^K, within-source geometry
percentile, interaction); 7 VGG matched-scalar ridge, the REFERENCE
(the P00-H scalar inputs: log C, log D, log NC1, s_dict, theta_deg,
logit scale, radius CV, log gamma, a, log rho; intercept; 3 source
indicators); 8 VGG no-target-batch ridge (7 without gamma, a, rho);
9 VGG source-majority material winner with the frozen fallbacks.
Ridge: lambda in {1e-4,...,10} by leave-one-VGG-checkpoint-out folds
(VGG seed reuse across sources is not audited; disclosed), loss =
equal-source average of per-source MSE, ties within 1e-12 -> larger
lambda; standardization fit inside each training fold; zero-variance
features -> 0; only continuous coefficients penalized.

Predictions are continuous gaps in the target metric's units; the
decision rule (CTM if > 1e-12, Energy if < -1e-12, tie otherwise) is
applied by the analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from heldout_theory_validation import severity_only

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
SCALARS = ("logC", "logD", "logNC1", "s_dict", "theta_deg", "logit", "eta", "log_gamma", "a", "log_rho")
NO_BATCH = ("logC", "logD", "logNC1", "s_dict", "theta_deg", "logit", "eta")
NAMES = ("always_energy", "always_ctm", "vgg_kid_isotonic", "vgg_fd_isotonic",
         "vgg_source_shift_mean", "vgg_geometry_severity_ridge",
         "vgg_matched_scalar_ridge", "vgg_no_target_batch_ridge", "vgg_source_majority")
REFERENCE = "vgg_matched_scalar_ridge"


def _design(df: pd.DataFrame, cont: tuple, extra_inter: bool, ind_sources: tuple):
    X = df[list(cont)].to_numpy(float)
    if extra_inter:
        X = np.column_stack([X, df["dK"].to_numpy(float) * df["g_pct"].to_numpy(float)])
    S = (np.column_stack([(df.source == s).to_numpy(float) for s in ind_sources])
         if ind_sources else np.zeros((len(df), 0)))
    return X, S


def _ridge_fit(Xc, S, y, lam):
    mu, sd = Xc.mean(0), Xc.std(0)
    sd = np.where(sd > 0, sd, np.inf)                      # zero-variance -> 0 after scaling
    Z = (Xc - mu) / sd
    A = np.column_stack([np.ones(len(Z)), S, Z])
    pen = np.diag(np.r_[np.zeros(1 + S.shape[1]), np.full(Z.shape[1], lam)])
    beta = np.linalg.solve(A.T @ A + pen, A.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta}


def _ridge_predict(fit, Xc, S):
    Z = (Xc - fit["mu"]) / fit["sd"]
    return np.column_stack([np.ones(len(Z)), S, Z]) @ fit["beta"]


def _ridge_cv(df, y, cont, inter, ind_sources):
    X, S = _design(df, cont, inter, ind_sources)
    cells = df.cell.to_numpy()
    losses = {}
    for lam in LAMBDAS:
        pred = np.full(len(df), np.nan)
        for c in np.unique(cells):
            te = cells == c
            fit = _ridge_fit(X[~te], S[~te], y[~te], lam)
            pred[te] = _ridge_predict(fit, X[te], S[te])
        per_src = [np.mean((pred[df.source == s] - y[df.source == s]) ** 2) for s in SOURCES if (df.source == s).any()]
        losses[lam] = float(np.mean(per_src))
    best = min(losses.values())
    lam = max(l for l, v in losses.items() if v - best <= 1e-12)
    return _ridge_fit(X, S, y, lam), lam, losses


class Comparators:
    def __init__(self, train: pd.DataFrame, target_col: str):
        self.target_col = target_col
        y = train[target_col].to_numpy(float)
        self.train = train
        # indicator columns only for sources PRESENT in training (reference = first present)
        self.ind_sources = tuple(s for s in SOURCES if (train.source == s).any())[1:]
        self.iso_tables = {}
        self.mean_table = train.groupby(["source", "ood_set"])[target_col].mean().to_dict()
        self.fits = {}
        for name, cont, inter in (("vgg_geometry_severity_ridge", ("dK", "g_pct"), True),
                                  ("vgg_matched_scalar_ridge", SCALARS, False),
                                  ("vgg_no_target_batch_ridge", NO_BATCH, False)):
            fit, lam, losses = _ridge_cv(train, y, cont, inter, self.ind_sources)
            self.fits[name] = (fit, cont, inter, lam, losses)
        mat = train[np.abs(train["dG"]) >= 0.01]
        self.majority = {}
        glob = np.sign(train[train[target_col] != 0][target_col]).sum()
        for s in SOURCES:
            m = mat[mat.source == s]
            v = np.sign(m[target_col]).sum() if len(m) else 0.0
            if v == 0:
                nt = train[(train.source == s) & (train[target_col] != 0)]
                v = np.sign(nt[target_col]).sum() if len(nt) else 0.0
            if v == 0:
                v = glob
            self.majority[s] = 1.0 if v > 0 else -1.0 if v < 0 else -1.0   # final tie -> Energy

    def predict(self, name: str, df: pd.DataFrame) -> np.ndarray:
        if name == "always_energy":
            return np.full(len(df), -1.0)
        if name == "always_ctm":
            return np.full(len(df), 1.0)
        if name in ("vgg_kid_isotonic", "vgg_fd_isotonic"):
            col = "dK" if name == "vgg_kid_isotonic" else "dF"
            out = np.full(len(df), np.nan)
            for s in df.source.unique():
                tr = self.train[self.train.source == s].rename(columns={col: "d", self.target_col: "gap"})
                te = df[df.source == s].rename(columns={col: "d"})
                out[(df.source == s).to_numpy()] = severity_only(tr, te)
            return out
        if name == "vgg_source_shift_mean":
            return np.array([self.mean_table.get((s, e), np.nan) for s, e in zip(df.source, df.ood_set)])
        if name == "vgg_source_majority":
            return np.array([self.majority[s] for s in df.source])
        fit, cont, inter, _, _ = self.fits[name]
        X, S = _design(df, cont, inter, self.ind_sources)
        return _ridge_predict(fit, X, S)

    def summary(self) -> dict:
        return {n: {"lambda": self.fits[n][3], "cv_losses": self.fits[n][4]} for n in self.fits} | \
               {"source_majority": self.majority, "folds": "leave-one-VGG-checkpoint-out (seed reuse across sources not audited)"}
