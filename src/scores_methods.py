"""Backwards-compatibility shim for pickled params saved before the CSF split.

The CSF classes used to live in `src/scores_methods.py`. They were split into
`src/csfs/*.py`, `src/trained_module.py`, and `src/neural_collapse.py`, and
the original file was deleted. That broke `torch.load()` on any `.pt`
parameter file pickled before the split, because pickle stores class
references as `module_path.ClassName` — and `src.scores_methods.ProjectionFiltering`
no longer resolves.

This module re-exports every class and helper from their new locations under
the old `src.scores_methods.<name>` path, so `torch.load()` can still find
them when unpickling legacy parameter files. New code should NOT import from
here; it should use the canonical locations:

  - CSF classes:  src.csfs.<module_name>
  - TrainedModule: src.trained_module
  - NeuralCollapseMetrics: src.neural_collapse
  - cov / TorchStandardScaler / _l2normalize / power_iteration: src.csfs._utils
    (or, for the trained-module-internal helpers, src.trained_module)
"""

# CSF classes (now in src/csfs/).
from src.csfs.class_typical_matching import ClassTypicalMatching
from src.csfs.entropy import EntropyScores
from src.csfs.fdbd import fDBD
from src.csfs.geometric_complexity import GeometricComplexity
from src.csfs.gradnorm import GradNorm
from src.csfs.kernel_pca import KernelPCA
from src.csfs.kl_matching import KLMatching
from src.csfs.mahalanobis import MahalanobisDistance
from src.csfs.neco import NeCo
from src.csfs.nnguide import NNGuide
from src.csfs.pnml import pNML
from src.csfs.projection_filtering import ProjectionFiltering
from src.csfs.residual import ResidualScore
from src.csfs.temperature_scaling import TemperatureScaling
from src.csfs.vim import ViMScore

# Shared utilities that used to live as module-level definitions in
# scores_methods.py. Moved to src.csfs._utils after the split.
from src.csfs._utils import TorchStandardScaler, cov

# Non-CSF classes that were also in scores_methods.py.
from src.trained_module import TrainedModule, _l2normalize, power_iteration
from src.neural_collapse import NeuralCollapseMetrics

__all__ = [
    "ClassTypicalMatching",
    "EntropyScores",
    "fDBD",
    "GeometricComplexity",
    "GradNorm",
    "KernelPCA",
    "KLMatching",
    "MahalanobisDistance",
    "NeCo",
    "NeuralCollapseMetrics",
    "NNGuide",
    "pNML",
    "ProjectionFiltering",
    "ResidualScore",
    "TemperatureScaling",
    "TorchStandardScaler",
    "TrainedModule",
    "ViMScore",
    "_l2normalize",
    "cov",
    "power_iteration",
]
