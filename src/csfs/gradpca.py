"""GradPCA head variants and the matched class-means activation PCA.

Implements the shipped-configuration GradPCA detector (Seleznova et al.,
ICLR 2026; official repo mselezniova/GradPCA, audited at commit 8f3261b)
and its exact activation-space partner, per the X6 E-series theorems in
`documentation/X6_gradpca_theorems.md`:

  - ``head_sum``    Head-parameter gradients of the summed logits
                    A(f) = sum_c f_c. Closed form: g(x) = (1_C h(x)^T, 1_C);
                    no autograd. By Theorem E1 the scores equal
                    ``act_cmeans`` exactly; this variant materializes the
                    literal gradient-space construction as the deployed
                    cross-check of that theorem.
  - ``head_max``    Head-parameter gradients of the maximum logit
                    (paper-text variant; NOT in the official code). Closed
                    form: g(x) = (e_{c-hat(x)} h(x)^T, e_{c-hat(x)}), the
                    class-gated activation lifting of Theorem E2. Routing
                    c-hat comes from the model's own head (w, b).
  - ``act_cmeans``  Class-means activation PCA: the identical spectral
                    construction applied to h(x) directly (the E1 partner).

Shared construction (GradPCA class-mean path, all variants): per-class mean
features from ALL train samples of each true class (no correct-only
filtering, matching the official implementation), centered by the
unweighted mean of class means, dual C x C Gram in float64, eigendecompose,
retain the smallest k reaching ``trace_threshold`` cumulative eigenvalue
mass (official default 0.99), lift into feature space, and score by the
squared retained-energy fraction

    s(x) = ||U_k^T (g(x) - g_bar)||^2 / ||g(x) - g_bar||^2 ,

higher = more ID. Class means use exact per-(true, predicted) accumulation
(``head_max``) or affinity of the feature map (``head_sum``/``act_cmeans``,
where the map is affine so the class mean of features is the feature of the
class mean); no N x P matrix is ever materialized. A test point whose
feature equals the global mean scores 0/0 = NaN by convention (surfaced,
never silently clipped; stats() nan_to_nums with a warning).
"""

import os
import torch
from fd_shifts import logger
from typing import Optional

ArrayType = torch.Tensor

#: Official GradPCA cumulative-trace threshold (methods/config.py, eps).
DEFAULT_TRACE_THRESHOLD = 0.99
#: Scoring chunk size: bounds the (chunk, C*d + C) feature block in memory.
DEFAULT_SCORE_CHUNK = 128

VARIANTS = ("head_sum", "head_max", "act_cmeans")


class GradPCA:
    """GradPCA head-gradient detector / class-means activation PCA.

    ``variant`` selects the feature map (see module docstring). The module
    is only touched for ``head_max`` (routing needs the head weights);
    ``head_sum`` and ``act_cmeans`` never access model parameters.
    """

    def __init__(self, module, study_name: str, cf, variant: str = "head_sum",
                 trace_threshold: float = DEFAULT_TRACE_THRESHOLD,
                 score_chunk: int = DEFAULT_SCORE_CHUNK):
        assert variant in VARIANTS, f"Unknown GradPCA variant {variant!r}; expected one of {VARIANTS}"
        self.cf = cf
        self.study_name = study_name
        self.variant = variant
        self.trace_threshold = trace_threshold
        self.score_chunk = score_chunk
        self.num_classes = cf.data.num_classes
        if variant == "head_max":
            from src import utils  # lazy: keeps the module importable without the full stack
            self.w, self.b = utils.get_model_and_last_layer(module, study_name, return_model=False)
            self.w = self.w.detach().to(torch.float64).cpu()
            self.b = self.b.detach().to(torch.float64).cpu()
        else:
            self.w = self.b = None
        # Fitted state
        self.mean = None          # (P,) float64 global mean of class-mean features
        self.components = None    # (P, k) float64 orthonormal retained directions
        self.n_components = None  # k

    # ------------------------------------------------------------------
    # Feature maps (all float64, CPU).
    # ------------------------------------------------------------------
    def _predictions(self, activations: ArrayType) -> ArrayType:
        """Routing for head_max: argmax of the model's own head logits.

        Logits are truncated to the first ``num_classes`` columns so a DG
        reservation logit can never win the argmax.
        """
        logits = activations.to(torch.float64) @ self.w.T + self.b
        return logits[:, : self.num_classes].argmax(dim=1)

    def _features(self, activations: ArrayType) -> ArrayType:
        """Map a batch of activations (n, d) to detector features (n, P)."""
        h = activations.to(torch.float64)
        n, d = h.shape
        C = self.num_classes
        if self.variant == "act_cmeans":
            return h
        if self.variant == "head_sum":
            # (1_C kron h, 1_C): every W-row block equals h, bias block = 1.
            return torch.cat([h.repeat(1, C), torch.ones(n, C, dtype=torch.float64)], dim=1)
        # head_max: block c-hat holds h, bias block = e_{c-hat}.
        preds = self._predictions(activations)
        w_block = torch.zeros(n, C, d, dtype=torch.float64)
        w_block[torch.arange(n), preds] = h
        bias_block = torch.zeros(n, C, dtype=torch.float64)
        bias_block[torch.arange(n), preds] = 1.0
        return torch.cat([w_block.reshape(n, C * d), bias_block], dim=1)

    def _class_mean_features(self, activations_train: ArrayType,
                             labels_train: ArrayType) -> ArrayType:
        """Exact (C, P) matrix of per-class mean features, no N x P tensor."""
        h = activations_train.to(torch.float64)
        y = labels_train.to(torch.long)
        d = h.shape[1]
        C = self.num_classes
        counts = torch.bincount(y, minlength=C)
        if (counts == 0).any():
            missing = (counts == 0).nonzero().flatten().tolist()
            raise ValueError(f"GradPCA ({self.variant}): classes {missing} have no train samples; "
                             f"the class-mean construction requires every class.")
        h_cmeans = torch.zeros(C, d, dtype=torch.float64)
        h_cmeans.index_add_(0, y, h)
        h_cmeans /= counts.to(torch.float64).unsqueeze(1)
        if self.variant == "act_cmeans":
            return h_cmeans
        if self.variant == "head_sum":
            # Affine feature map: class mean of features = feature of class mean.
            return torch.cat([h_cmeans.repeat(1, C), torch.ones(C, C, dtype=torch.float64)], dim=1)
        # head_max: accumulate per (true class, predicted class) cell.
        preds = self._predictions(activations_train)
        cell = y * C + preds
        cell_sums = torch.zeros(C * C, d, dtype=torch.float64)
        cell_sums.index_add_(0, cell, h)
        cell_counts = torch.bincount(cell, minlength=C * C).to(torch.float64)
        w_block = cell_sums.reshape(C, C, d) / counts.to(torch.float64).reshape(C, 1, 1)
        bias_block = cell_counts.reshape(C, C) / counts.to(torch.float64).unsqueeze(1)
        return torch.cat([w_block.reshape(C, C * d), bias_block], dim=1)

    # ------------------------------------------------------------------
    # Fit.
    # ------------------------------------------------------------------
    def compute_GradPCA_params(self, activations_train: ArrayType,
                               labels_train: ArrayType):
        logger.info(f"GradPCA ({self.variant}): Fitting parameters...")
        cmeans = self._class_mean_features(activations_train.clone(), labels_train)
        self.mean = cmeans.mean(dim=0)
        centered = cmeans - self.mean
        gram = centered @ centered.T  # dual C x C Gram, float64
        evals, evecs = torch.linalg.eigh(gram)
        evals, evecs = evals.flip(0).clamp(min=0.0), evecs.flip(1)
        cum = torch.cumsum(evals, dim=0) / evals.sum()
        k = int((cum >= self.trace_threshold).nonzero()[0].item()) + 1
        # Defensive cap: never lift a numerically-zero eigendirection.
        max_rank = int((evals > evals[0] * 1e-12).sum().item())
        k = min(k, max_rank)
        components = centered.T @ evecs[:, :k]
        self.components = components / components.norm(dim=0, keepdim=True)
        self.n_components = k
        logger.info(f"GradPCA ({self.variant}): retained k={k} of {self.num_classes} "
                    f"class-mean directions at trace threshold {self.trace_threshold}")

    # ------------------------------------------------------------------
    # Score.
    # ------------------------------------------------------------------
    def get_scores(self, activations_eval: ArrayType) -> ArrayType:
        assert self.components is not None, "GradPCA parameters have not been computed or loaded"
        logger.info(f"GradPCA ({self.variant}): Computing scores...")
        activations_eval = activations_eval.clone()
        scores = []
        for start in range(0, activations_eval.shape[0], self.score_chunk):
            z = self._features(activations_eval[start: start + self.score_chunk]) - self.mean
            num = (z @ self.components).pow(2).sum(dim=1)
            den = z.pow(2).sum(dim=1)
            scores.append(num / den)
        return torch.cat(scores)

    # ------------------------------------------------------------------
    # Persistence.
    # ------------------------------------------------------------------
    def save_params(self, path: Optional[str] = None, filename: str = "GradPCA_params"):
        assert self.components is not None, "GradPCA parameters have not been computed"
        params_dict = {
            "variant": self.variant,
            "trace_threshold": self.trace_threshold,
            "mean": self.mean,
            "components": self.components,
            "n_components": self.n_components,
        }
        if path is None:
            params_dir = f"{self.cf.exp.dir}/params"
            if not os.path.exists(params_dir):
                os.mkdir(params_dir)
            path = f"{params_dir}/{filename}.pt"
        else:
            path = f"{path}/{filename}.pt"
        logger.info(f"GradPCA ({self.variant}): Saving parameters in {path}")
        torch.save(params_dict, path)

    def load_params(self, path: Optional[str] = None, filename: str = "GradPCA_params"):
        if path is None:
            path = f"{self.cf.exp.dir}/params/{filename}.pt"
        else:
            path = f"{path}/{filename}.pt"
        assert os.path.exists(path), f"Specified path {path} does not exist..."
        logger.info(f"GradPCA ({self.variant}): Loading parameters from {path}")
        params_dict = torch.load(path)
        assert params_dict["variant"] == self.variant, (
            f"Parameter file holds variant {params_dict['variant']!r}, "
            f"but this instance is {self.variant!r}")
        self.trace_threshold = params_dict["trace_threshold"]
        self.mean = params_dict["mean"]
        self.components = params_dict["components"]
        self.n_components = params_dict["n_components"]
