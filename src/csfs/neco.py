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

from src.csfs._utils import TorchStandardScaler

class NeCo:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        self.pca_estimator = None
        self.tss = None

    def compute_NeCo_params(self, activations_train: ArrayType,
                                D: int|None = None ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        # Scaling
        logger.info('NeCo Score: Fitting parameters...')
        activations_train = activations_train.clone()
        dimensions = activations_train.shape[1]
        self.tss = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
        self.tss.fit(activations_train)
        self.activations_scaled = self.tss.transform(activations_train)
        # Principal Compoment Analysis 
        self.pca_estimator = PCA(n_components=None, svd_solver='full')
        self.pca_estimator.fit(self.activations_scaled)


    def get_scores( self, activations_eval: ArrayType, neco_dim:int|None=100):
        # logger.info('ViM: Computing scores')
        logger.info(f'NeCo Score: Computing scores...')
        activations_eval = activations_eval.clone()
        if self.cf.model.network.backbone is None:
            if 'ViT' in self.cf.model.network.name:
                X_eval = activations_eval
            else:
                X_eval = self.tss.transform(activations_eval)
        else:
            if 'ViT' in self.cf.model.network.backbone:
                X_eval = activations_eval
            else:
                X_eval = self.tss.transform(activations_eval)
        X_projected_full = self.pca_estimator.transform(X_eval)
        X_projected_reduced = X_projected_full[:,:neco_dim]
        logit_eval = activations_eval @ self.w.T + self.b
        logit_eval_max = logit_eval.max(dim=-1).values
        confs = []
        # activations_eval = activations_eval
        for i in tqdm(range(activations_eval.shape[0])):
            norm_full = torch.linalg.norm(X_eval[i, :])
            norm_reduced = torch.linalg.norm(X_projected_reduced[i, :])
            score = norm_reduced/norm_full
            confs.append(score)
        scores_eval = torch.Tensor(confs)
        if self.cf.model.network.backbone is None:
            if 'resnet' not in self.cf.model.network.name:
                scores_eval = scores_eval*logit_eval_max
        else:
            if 'resnet' not in self.cf.model.network.backbone:
                scores_eval = scores_eval*logit_eval_max
        
        return scores_eval

    def save_params(self, path:str|None=None, filename:str='NeCo_params'):
        assert self.tss is not None, 'Residual matrix has not been computed'
        assert self.pca_estimator is not None, 'alpha constant has not been computed'
        params_dict = {
                    'tss': self.tss,
                    'pca_estimator': self.pca_estimator,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'NeCo Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='NeCo_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'NeCo Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.tss = params_dict['tss']
        self.pca_estimator = params_dict['pca_estimator']


#%%
