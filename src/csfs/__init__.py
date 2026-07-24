"""Confidence Score Functions (CSFs).

Each file in this directory implements one CSF. The list below is the
authoritative inventory: importing `from src.csfs import MyCSF` works
once a new CSF is added to the re-exports below.

To add a new CSF:
    1. Copy `_template.py` to `<your_csf_name>.py` and fill it in.
    2. Add the import + symbol to the re-exports + __all__ below.
    3. Wire it into `src/csf_pipeline.py` (alongside the other CSF
       instantiations in `run_score_methods` / `load_score_methods`).
"""

from src.csfs.class_typical_matching import ClassTypicalMatching
from src.csfs.entropy import EntropyScores
from src.csfs.fdbd import fDBD
from src.csfs.geometric_complexity import GeometricComplexity
from src.csfs.gradnorm import GradNorm
from src.csfs.kernel_pca import KernelPCA
from src.csfs.kl_matching import KLMatching
from src.csfs.mahalanobis import MahalanobisDistance
from src.csfs.mahalanobis_pp import MahalanobisPP
from src.csfs.nci import NCI
from src.csfs.neco import NeCo
from src.csfs.nnguide import NNGuide
from src.csfs.pnml import pNML
from src.csfs.projection_filtering import ProjectionFiltering
from src.csfs.residual import ResidualScore
from src.csfs.temperature_scaling import TemperatureScaling
from src.csfs.vim import ViMScore

__all__ = [
    "ClassTypicalMatching",
    "EntropyScores",
    "fDBD",
    "GeometricComplexity",
    "GradNorm",
    "KernelPCA",
    "KLMatching",
    "MahalanobisDistance",
    "MahalanobisPP",
    "NCI",
    "NeCo",
    "NNGuide",
    "pNML",
    "ProjectionFiltering",
    "ResidualScore",
    "TemperatureScaling",
    "ViMScore",
]
