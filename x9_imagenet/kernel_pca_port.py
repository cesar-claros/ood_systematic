"""KPCA RecError (global mode), ported line-for-line from
`src/csfs/kernel_pca.py` without the fd_shifts/module coupling.

The math is identical: L2-normalized features, RBF kernel, Nystrom landmark
set selected by ASCENDING energy score, whitened landmark eigenbasis,
K_c = K_nm.mean(0) centering, explained-variance component cut, negative
reconstruction-residual as confidence. Hyperparameters follow the paper's
Bayesian-optimization protocol on validation failure-AUGRC (same bounds,
same random_state). The constructor takes (w, b, num_classes) directly
instead of an fd_shifts module.
"""
from __future__ import annotations

import torch
from loguru import logger


def cov(tensor, centered: bool = False, rowvar: bool = True,
        bias: bool = False):
    """Verbatim from src/csfs/_utils.py."""
    tensor = tensor.clone()
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    if not centered:
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """scores_funcs.energy convention: T*logsumexp(logits/T), higher = ID."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


class KernelPCAPort:
    def __init__(self, w: torch.Tensor, b: torch.Tensor, num_classes: int):
        self.w, self.b = w, b
        self.num_classes = num_classes
        self.X_ref = self.K_c = self.normalization = self.u_q = None
        self.explained_variance = None
        self.kernel = None
        self.gamma = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def _feat_normalization(self, X):
        X = X / (torch.norm(X, p=2, dim=-1, keepdim=True) + 1e-12)
        return X.contiguous()

    def _rbf_kernel(self, X, Y, gamma=None):
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        X_norm = (X ** 2).sum(dim=1).view(-1, 1)
        Y_norm = (Y ** 2).sum(dim=1).view(1, -1)
        K = X_norm + Y_norm - 2 * X @ Y.t()
        return torch.exp(-gamma * K)

    def _kernel(self, X, Y):
        if self.kernel == "rbf":
            return self._rbf_kernel(X, Y, gamma=self.gamma)
        raise NotImplementedError("Only 'rbf' kernel implemented.")

    def _get_eigendecomposition(self, M):
        try:
            eigvals, eigvecs = torch.linalg.eigh(M)
        except Exception:  # noqa: BLE001
            n, m = M.shape
            M_p = M + 1e-6 * torch.eye(min(m, n), device=M.device)
            try:
                eigvals, eigvecs = torch.linalg.eigh(M_p)
            except Exception:  # noqa: BLE001
                logger.warning("KernelPCA: GPU eigh failed twice; "
                               "falling back to CPU fp64 LAPACK.")
                M_cpu = M.detach().double().cpu()
                M_cpu = (M_cpu + M_cpu.T) / 2
                eigvals, eigvecs = torch.linalg.eigh(M_cpu)
                eigvals = eigvals.to(dtype=M.dtype, device=M.device)
                eigvecs = eigvecs.to(dtype=M.dtype, device=M.device)
        idx = torch.argsort(eigvals, descending=True)
        return eigvals[idx], eigvecs[:, idx]

    def KPCA(self, X, ref_indices, verbose=False):
        X = self._feat_normalization(X).to(self.device)
        X_ref = X[ref_indices]
        K_mm = self._kernel(X_ref, X_ref)
        evals, evecs = self._get_eigendecomposition(K_mm)
        evals = torch.maximum(evals, torch.tensor(1e-12, device=evals.device))
        normalization = (evecs / torch.sqrt(evals)) @ evecs.T
        K_nm = self._kernel(X, X_ref) @ normalization.T
        K_c = K_nm.mean(dim=0)
        sigma = cov(K_nm - K_c, centered=True, rowvar=False)
        evals_full, evecs_full = self._get_eigendecomposition(sigma)
        accum = evals_full.cumsum(0)
        ratio = accum / accum[-1]
        n_components = int((ratio <= self.explained_variance).sum().item()) + 1
        u_q = evecs_full[:, :n_components]
        if verbose:
            logger.info(f"n_components={n_components}, "
                        f"variance ratio={ratio[n_components - 1]:.4f}")
        # Unlike the original (which round-trips through save/load), params
        # stay on self.device: <=100 MB at 5000 landmarks, and it avoids a
        # device mismatch against CUDA eval tensors in the BO loop.
        return X_ref, K_c, normalization, u_q

    def compute_params(self, activations_train, temperature=None,
                       n_landmarks=None, explained_variance=None,
                       gamma=None, kernel=None, verbose=False):
        activations_train = activations_train.clone()
        self.explained_variance = (0.95 if explained_variance is None
                                   else explained_variance)
        self.gamma = 0.2 if gamma is None else gamma
        self.temperature = 1.0 if temperature is None else temperature
        self.n_landmarks = (int(activations_train.shape[0] / self.num_classes)
                            if n_landmarks is None else int(n_landmarks))
        self.kernel = "rbf" if kernel is None else kernel
        logits_train = activations_train @ self.w.T + self.b
        en = energy_score(logits_train, temperature=self.temperature)
        ref_idx = en.argsort()[: self.n_landmarks]
        self.X_ref, self.K_c, self.normalization, self.u_q = self.KPCA(
            activations_train, ref_indices=ref_idx, verbose=verbose)

    def get_scores(self, activations_eval):
        X = self._feat_normalization(activations_eval.clone()).to(
            self.X_ref.device)
        K_nm = self._kernel(X, self.X_ref) @ self.normalization.T
        K_nm_c = K_nm - self.K_c
        back = (self.u_q @ self.u_q.T @ K_nm_c.T).T
        return -1 * torch.linalg.norm(back - K_nm_c, ord=2, dim=1)

    def tune_hyperparameters(self, activations_train, activations_val,
                             residuals_val, rc_metrics,
                             temperature=None,
                             var_bounds=(0.85, 0.99),
                             gamma_bounds=(0.10, 1.0),
                             landmarks_bounds=(2000, 5000),
                             n_iters=80, n_init=20):
        """rc_metrics: callable (confids_np, residuals_np) -> (augrc, aurc),
        matching pool_a_analysis.rc_metrics."""
        from bayes_opt import BayesianOptimization
        activations_val = activations_val.clone()

        def _get_metric(explained_variance, gamma, n_landmarks):
            self.compute_params(activations_train, temperature=temperature,
                                n_landmarks=int(n_landmarks),
                                explained_variance=explained_variance,
                                gamma=gamma)
            scores_val = self.get_scores(activations_val)
            a, _ = rc_metrics(scores_val.cpu().numpy(), residuals_val)
            return -a

        bo = BayesianOptimization(
            f=_get_metric,
            pbounds={"explained_variance": var_bounds,
                     "gamma": gamma_bounds,
                     "n_landmarks": landmarks_bounds},
            verbose=0, random_state=1)
        bo.maximize(init_points=n_init, n_iter=n_iters)
        best = bo.max["params"]
        self.compute_params(activations_train, temperature=temperature,
                            n_landmarks=int(best["n_landmarks"]),
                            explained_variance=best["explained_variance"],
                            gamma=best["gamma"], verbose=True)
        logger.info(f"KPCA best params: {best}, "
                    f"best AUGRC: {-bo.max['target']:.4f}")
        return best
