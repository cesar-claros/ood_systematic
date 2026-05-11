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

from src.csfs._utils import cov

class ViMScore:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.ext_confid_name = self.cf.eval.ext_confid_name
        self.study_name = study_name
        self.query_confids = self.cf.eval.confidence_measures
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        
        self.u = -torch.linalg.pinv(self.w) @ self.b
        self.residual = None
        self.alpha = None
    
    def compute_ViM_params(self, activations_train: ArrayType,
                                D: int|None = None ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        logger.info('ViM Score: Fitting parameters...')
        activations_train = activations_train.clone()
        logit_train = activations_train @ self.w.T + self.b
        if D is None:
            if activations_train.shape[1] >= 2048:
                self.D = 1000
            elif activations_train.shape[1] >= 768:
                self.D = 512
            else:
                self.D = activations_train.shape[1] // 2
        else:
            self.D = D
        
        X_train = (activations_train - self.u)
        covariance = cov(X_train, centered=True, rowvar=False)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        sorted_indices = torch.argsort(eigenvalues.real, descending=True)
        self.residual = eigenvectors[:, sorted_indices[self.D:]].real
        virtual_logit_norm_train = torch.linalg.norm(X_train @ self.residual, dim=1)
        self.alpha = logit_train.max(dim=1).values.mean() / virtual_logit_norm_train.mean()
    
    def save_params(self, path:str|None=None, filename:str='ViM_params'):
        assert self.residual is not None, 'Residual matrix has not been computed'
        assert self.alpha is not None, 'alpha constant has not been computed'
        params_dict = {
                    'residual': self.residual,
                    'alpha': self.alpha,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'ViM Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='ViM_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'ViM Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.residual = params_dict['residual']
        self.alpha = params_dict['alpha']

    
    def get_scores( self, activations_eval: ArrayType ):
        # logger.info('ViM: Computing scores')
        logger.info(f'ViM Score: Computing scores...')
        activations_eval = activations_eval.clone()
        X_eval = (activations_eval - self.u)    
        logit_eval = activations_eval @ self.w.T + self.b

        virtual_logit_eval = torch.linalg.norm(X_eval @ self.residual, dim=1) * self.alpha
        energy_eval = torch.logsumexp(logit_eval, dim=1)
        scores_eval = -virtual_logit_eval + energy_eval
        
        return scores_eval

# %%
