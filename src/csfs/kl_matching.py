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

class KLMatching:

    def __init__(self,cf):
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        # self.precision = None
        # self.unique_labels = None
        self.means = None
    
    def pairwise_kl_divergence(self, p: ArrayType, q: ArrayType) -> ArrayType:
        """
        Compute the pairwise KL divergence between two sets of distributions.

        Args:
            p (torch.Tensor): A tensor of shape (N, D), where N is the batch size and D is the dimensionality of the distributions.
            q (torch.Tensor): A tensor of shape (M, D), where M is the batch size and D is the dimensionality of the distributions.

        Returns:
            torch.Tensor: A tensor of shape (N, M) containing the pairwise KL divergences.
        """

        N, D = p.shape
        M, _ = q.shape

        p = p.unsqueeze(1).expand(N, M, D)
        q = q.unsqueeze(0).expand(N, M, D)

        kl = torch.sum( torch.where(p != 0, p * torch.log(p / q), 0), dim=-1)

        return kl

    def min_pairwise_kl_divergence(self, p: ArrayType, q: ArrayType) -> ArrayType:
        """
        Find the argmin of the pairwise KL divergence between two sets of distributions.

        Args:
            p (torch.Tensor): A tensor of shape (N, D), where N is the batch size and D is the dimensionality of the distributions.
            q (torch.Tensor): A tensor of shape (M, D), where M is the batch size and D is the dimensionality of the distributions.

        Returns:
            torch.Tensor: A tensor of shape (N,) containing the indices of the minimum KL divergence for each distribution in p.
        """

        kl = self.pairwise_kl_divergence(p, q)
        return torch.min(kl, dim=1).values

    def compute_KLMatching_params(self, softmax_train: ArrayType,):
        logger.info("KLMatching: Fitting parameters...")
        softmax_train = softmax_train.clone()
        # n, n_classes = softmax_train.shape
        predicted_labels = softmax_train.max(dim=1).indices
        self.means = torch.vstack([
            softmax_train[ predicted_labels == i ].mean(dim=0) for i in range(self.num_classes) 
            ])
    
    def save_params(self, path:str|None=None, filename:str='KLMatching_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.means is not None, 'Class means constant have not been computed...'
        # assert self.unique_labels is not None, 'Unique labels have not been computed...'
        params_dict = {
                        # 'precision': self.precision,
                        'means': self.means,
                        # 'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'KLMatching: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='KLMatching_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'KLMatching: Loading parameters from {path}')
        params_dict = torch.load(path)
        # self.precision = params_dict['precision']
        self.means = params_dict['means']
        # self.unique_labels = params_dict['unique_labels']
    
    def get_scores( self, softmax_eval: ArrayType ) -> ArrayType:
        logger.info(f'KLMatching: Computing scores...')
        softmax_eval = softmax_eval.clone()
        scores_eval = -self.min_pairwise_kl_divergence(softmax_eval, self.means)
        return scores_eval

# %%
