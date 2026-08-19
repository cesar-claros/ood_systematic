import os
import copy
import numpy as np
import torch
import faiss
from torch.autograd import Variable
from torch_pca import PCA
from fd_shifts import logger
from tqdm import tqdm
from torch.nn import functional as F
from typing import Any, Callable, TypeVar, List, Tuple, Optional
import pandas as pd
from src.rc_stats import RiskCoverageStats
from src import utils
from src import scores_funcs
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
#%%
ArrayType = torch.Tensor
T = TypeVar(
    "T", Callable[[ArrayType], ArrayType], Callable[[ArrayType, ArrayType], ArrayType]
)

from contextlib import contextmanager


@contextmanager
def quiet_logging(scope: str = "src"):
    """Silence per-iteration INFO chatter from `scope` modules.

    Used around Bayesian-optimization loops whose objective re-fits a CSF
    (~100 iterations x several log lines each). Exceptions still propagate
    and print normally, so errors remain visible.
    """
    logger.disable(scope)
    try:
        yield
    finally:
        logger.enable(scope)

def cov(tensor:ArrayType, centered:bool=False, rowvar:bool=True, bias:bool=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor.clone()
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    if not centered:
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()
#%%
class TorchStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, x, threshold=1.0):
        x = x.clone()
        self.mean_ = x.mean(0, keepdim=True)
        self.std_ = x.std(0, unbiased=False, keepdim=True)
        condition = self.std_< threshold
        if torch.any(condition):
            logger.info(f'Standard deviation of {condition.sum()} variables is less than {threshold} with average value={self.std_.mean():.4f}. Only centering is applied...')
            # self.mean_ = torch.zeros_like(self.mean_)
            self.std_ = torch.ones_like(self.std_)
    def transform(self, x, tol=1e-12):
        assert self.mean_ is not None, 'Mean has not been computed...'
        assert self.std_ is not None, 'Standard deviation has not been computed...' 
        x = x.clone()
        x -= self.mean_
        # x /= (self.std_ + tol)
        return x
    def inverse_transform(self, x, tol=1e-12):
        assert self.mean_ is not None, 'Mean has not been computed...'
        assert self.std_ is not None, 'Standard deviation has not been computed...' 
        x = x.clone()
        # x *= (self.std_ + tol)
        x += self.mean_
        return x

#%%
