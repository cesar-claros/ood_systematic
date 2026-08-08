"""NCI: OOD detection through the lens of Neural Collapse.

Liu & Qin, "Detecting Out-of-Distribution Through the Lens of Neural
Collapse" (CVPR 2025; arXiv 2311.01479). Score of a sample with feature h,
predicted class c, ID train-feature mean u, and last-layer weight rows w:

    NCI(x) = <w_c, h - u> / ||h - u||_2 + alpha * ||h||_1

The first term measures alignment of the centered feature with the predicted
class's weight vector (ID features cluster near weight directions under
collapse); the L1-norm term filters low-norm OOD features. Higher = more
ID-like, matching the pipeline convention. `alpha` is selected on the
validation split by minimizing failure-detection AUGRC, mirroring the
pipeline's fit-on-validation convention for CSF hyperparameters.
"""

import os

import torch
from fd_shifts import logger

from src import utils
from src.rc_stats import RiskCoverageStats

ArrayType = torch.Tensor
_EPS = 1e-10
_DEFAULT_ALPHAS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


class NCI:
    """Neural-collapse-inspired detector (weight alignment + L1 norm filter)."""

    def __init__(self, module, study_name: str, cf):
        self.cf = cf
        self.num_classes = cf.data.num_classes
        if study_name == 'vit':
            w, _ = utils.get_model_and_last_layer(module, study_name,
                                                  return_model=False)
        else:
            _, w, _ = utils.get_model_and_last_layer(module, study_name)
        # Deep Gamblers heads carry an extra reservation row; keep class rows.
        self.w = w.detach().cpu().float()[:self.num_classes]
        self.train_mean = None
        self.alpha = None

    def _score(self, activations_eval: ArrayType, logits_eval: ArrayType,
               alpha: float) -> ArrayType:
        centered = activations_eval - self.train_mean
        pred = logits_eval[:, :self.num_classes].argmax(dim=1)
        align = (self.w[pred] * centered).sum(dim=1)
        align = align / (centered.norm(dim=1) + _EPS)
        return align + alpha * activations_eval.abs().sum(dim=1)

    def compute_NCI_params(self, activations_train: ArrayType,
                           activations_val: ArrayType | None = None,
                           logits_val: ArrayType | None = None,
                           correct_val: ArrayType | None = None,
                           alphas: tuple = _DEFAULT_ALPHAS,
                           method: str = "bo",
                           n_init: int = 20, n_iters: int = 80) -> None:
        """alpha selection on validation failure-AUGRC. method='bo' (the
        harmonized protocol, 2026-08-08): Bayesian optimization over the
        grid's span (0, 3e-2), alpha=0 (pure alignment) reachable, same BO
        convention as the pipeline's other tuned CSFs (20+80,
        random_state=1). method='grid': the original 7-point search, which
        produced the first E-F benchmark scores."""
        logger.info(f"NCI: Fitting parameters (alpha via {method})...")
        self.train_mean = activations_train.mean(dim=0)
        if activations_val is None or logits_val is None or correct_val is None:
            self.alpha = alphas[len(alphas) // 2]
            logger.info(f"NCI: No validation data given; alpha={self.alpha}")
            return
        residuals = (1 - correct_val).float()

        def val_augrc(alpha: float) -> float:
            confids = self._score(activations_val, logits_val, alpha)
            return RiskCoverageStats(confids=confids,
                                     residuals=residuals).augrc

        if method == "bo":
            from bayes_opt import BayesianOptimization
            bo = BayesianOptimization(
                f=lambda alpha: -val_augrc(alpha),
                pbounds={"alpha": (0.0, max(alphas))},
                verbose=0, random_state=1)
            bo.maximize(init_points=n_init, n_iter=n_iters)
            self.alpha = float(bo.max["params"]["alpha"])
            best_augrc = -bo.max["target"]
        elif method == "grid":
            best_alpha, best_augrc = alphas[0], float("inf")
            for alpha in alphas:
                augrc = val_augrc(alpha)
                if augrc < best_augrc:
                    best_alpha, best_augrc = alpha, augrc
            self.alpha = best_alpha
        else:
            raise ValueError(f"unknown alpha selection method: {method}")
        logger.info(f"NCI: Selected alpha={self.alpha:.6g} "
                    f"(val failure-AUGRC={best_augrc:.4f})")

    def save_params(self, path: str | None = None,
                    filename: str = 'NCI_params') -> None:
        assert self.train_mean is not None, 'NCI parameters not computed...'
        params = {'train_mean': self.train_mean, 'alpha': self.alpha,
                  'w': self.w}
        if path is None:
            os.makedirs(f'{self.cf.exp.dir}/params', exist_ok=True)
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'NCI: Saving parameters in {path}')
        torch.save(params, path)

    def load_params(self, path: str | None = None,
                    filename: str = 'NCI_params') -> None:
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'NCI: Loading parameters from {path}')
        params = torch.load(path)
        self.train_mean = params['train_mean']
        self.alpha = params['alpha']
        self.w = params['w']

    def get_scores(self, activations_eval: ArrayType,
                   logits_eval: ArrayType) -> ArrayType:
        logger.info('NCI: Computing scores...')
        return self._score(activations_eval, logits_eval, self.alpha)
