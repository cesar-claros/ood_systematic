"""Template for a new Confidence Score Function (CSF).

Copy this file to `<your_csf_name>.py`, rename `TemplateCSF` to your class
name, and fill in the four numbered sections. Then add the class to
`src/csfs/__init__.py` (re-exports + __all__) and wire instantiations into
`src/utils_funcs.run_score_methods` / `load_score_methods`.

Conventions used by the existing CSFs in this directory (model your CSF on
whichever subset applies):

  __init__(self, cf, ...)
      Take the FD-Shifts config as the first positional argument. Some CSFs
      also accept (module, study_name, cf) when they need direct model
      access (e.g., NeCo, KernelPCA, NNGuide).

  compute_<Name>_params(self, activations_train, labels_train=None, ...)
      Fit any parameters needed before scoring. Called once per checkpoint
      during Stage 1 (`csf_fit.py`). For non-parametric CSFs, omit this
      method entirely (see `MahalanobisDistance` vs `GradNorm` for
      contrasting examples).

  save_params(self, path=None, filename=...)
      Serialize fitted state with torch.save. Defaults to
      `{cf.exp.dir}/params/{filename}.pt`. Skip for non-parametric CSFs.

  load_params(self, path=None, filename=...)
      Inverse of save_params. Stage 2 (`csf_eval.py`) calls this to
      restore state before scoring an unseen OOD split.

  get_scores(self, ...) -> ArrayType
      Return a 1-D tensor of confidence scores, one per row. Higher is
      more confident (i.e., more likely in-distribution). The scoring
      pipeline negates as needed for AURC/AUGRC computation.
"""

import os
import torch
from fd_shifts import logger
from typing import Optional

ArrayType = torch.Tensor


class TemplateCSF:
    """One-line description of what this CSF measures.

    Longer description: which input it consumes (logits, softmax,
    penultimate features), what statistic it computes, and any references.
    """

    def __init__(self, cf):
        # (1) Store config and initialize any fitted state to None.
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        # Example fitted-state placeholders:
        # self.precision = None
        # self.means = None

    # ------------------------------------------------------------------
    # (2) Fit. Omit this section entirely for non-parametric CSFs.
    # ------------------------------------------------------------------
    def compute_template_params(
        self,
        activations_train: ArrayType,
        labels_train: Optional[ArrayType] = None,
    ):
        logger.info("TemplateCSF: Fitting parameters...")
        activations_train = activations_train.clone()
        # Fit your statistic here. Examples:
        #   class-conditional means, precision matrices (Mahalanobis),
        #   PCA basis (NeCo, ProjectionFiltering),
        #   k-NN index over training features (NNGuide).
        raise NotImplementedError

    # ------------------------------------------------------------------
    # (3) Score. Required.
    # ------------------------------------------------------------------
    def get_scores(self, activations_eval: ArrayType) -> ArrayType:
        logger.info("TemplateCSF: Computing scores...")
        activations_eval = activations_eval.clone()
        # Return a 1-D tensor (N,) of confidence scores. Higher = more ID.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # (4) Persistence. Omit if the CSF has no fitted state.
    # ------------------------------------------------------------------
    def save_params(self, path: str | None = None, filename: str = "Template_params"):
        # Replace these assertions with your own fitted-state checks.
        # assert self.precision is not None, "Precision has not been computed..."
        params_dict = {
            # "precision": self.precision,
            # "means": self.means,
        }
        if path is None:
            params_dir = f"{self.cf.exp.dir}/params"
            if not os.path.exists(params_dir):
                os.mkdir(params_dir)
            path = f"{params_dir}/{filename}.pt"
        else:
            path = f"{path}/{filename}.pt"
        logger.info(f"TemplateCSF: Saving parameters in {path}")
        torch.save(params_dict, path)

    def load_params(self, path: str | None = None, filename: str = "Template_params"):
        if path is None:
            path = f"{self.cf.exp.dir}/params/{filename}.pt"
        else:
            path = f"{path}/{filename}.pt"
        assert os.path.exists(path), f"Specified path {path} does not exist..."
        logger.info(f"TemplateCSF: Loading parameters from {path}")
        params_dict = torch.load(path)
        # self.precision = params_dict["precision"]
        # self.means = params_dict["means"]
