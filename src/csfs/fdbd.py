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

class fDBD:
# Fast Decision Boundary based Out-of-Distribution Detector

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.normalizer = lambda x: x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-10)
        self.denominator_matrix = None
        self.mean_global_train = None
        # self.proportion = None

    def compute_fDBD_params(self, activations_train: ArrayType,):
        
        # self.proportion = proportion
        logger.info(f'fDBD Score: Fitting parameters...')
        activations_train = activations_train.clone()
        self.mean_global_train = activations_train.mean(dim=0)
        self.denominator_matrix = torch.zeros((self.num_classes, self.num_classes))
        for p in range(self.num_classes):
            w_p = self.w - self.w[p, :]
            denominator = torch.linalg.norm(w_p, dim=1)
            denominator[p] = 1
            self.denominator_matrix[p, :] = denominator
            
    def get_scores( self, activations_eval: ArrayType, logits_eval: ArrayType|None=None, normalizer='distance' ):
        # logger.info('ViM: Computing scores')
        logger.info(f'fDBD: Computing scores...')
        activations_eval = activations_eval.clone()
        if logits_eval is None:
            logits_eval = activations_eval @ self.w.T + self.b
        else:
            logits_eval = logits_eval.clone()
            logits_eval = logits_eval[:,:self.num_classes]
        mls, mls_idx = logits_eval.max(dim=1)
        logits_abs_sub = torch.abs(logits_eval - mls[:,None])
        if normalizer=='distance':
            scores_eval = 1/(self.num_classes-1) \
                                * torch.sum(logits_abs_sub / self.denominator_matrix[mls_idx], dim=1)\
                                * 1/(torch.norm(activations_eval - self.mean_global_train, dim=1))
        elif normalizer=='activation':
            scores_eval = 1/(self.num_classes-1) \
                                * torch.sum(logits_abs_sub / self.denominator_matrix[mls_idx], dim=1)\
                                * 1/(torch.norm(activations_eval, dim=1))
        else:
            raise NotImplementedError
        
        return scores_eval
    
    
    def save_params(self, path:str|None=None, filename:str='fDBD_params'):
        assert self.mean_global_train is not None, 'mean_global_train has not been computed'
        assert self.denominator_matrix is not None, 'denominator_matrix has not been computed'
        params_dict = {
                    'mean_global_train': self.mean_global_train,
                    'denominator_matrix': self.denominator_matrix,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'fDBD score: Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='fDBD_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'fDBD score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.mean_global_train = params_dict['mean_global_train']
        self.denominator_matrix = params_dict['denominator_matrix']


#%%
