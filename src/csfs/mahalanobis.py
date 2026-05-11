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

class MahalanobisDistance:

    def __init__(self,cf):
        self.cf = cf
        self.precision = None
        self.unique_labels = None
        self.means = None

    def compute_MahaDist_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType, ):
        logger.info("MahalanobisDistance: Fitting parameters...")
        activations_train = activations_train.clone()
        labels_train = labels_train.clone()
        self.unique_labels = torch.unique(labels_train)
        self.means = [
            activations_train[ labels_train == i ].mean(dim=0) for i in self.unique_labels 
            ]
        class_centered_features = torch.concat(
            [ activations_train[labels_train == i] - self.means[i] for i in self.unique_labels ],
            dim=0)
        covariance = cov(class_centered_features, centered=True, rowvar=False)
        try:
            self.precision = torch.linalg.pinv(covariance, hermitian=True, rtol=1e-6)
        except RuntimeError as e:
            if "The algorithm failed to converge" in str(e):
                print("Caught a convergence error with torch.linalg.eigh:", e)
                self.precision = torch.linalg.pinv(covariance.double() + 1e-6*torch.eye(covariance.shape[0]), hermitian=True, rtol=1e-6)
                self.precision = self.precision.float()
            else:
                raise e

        if torch.isnan(self.precision).any() or torch.isinf(self.precision).any():
            self.precision = torch.zeros_like(covariance)
    
    def save_params(self, path:str|None=None, filename:str='MahaDist_params'):
        assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.means is not None, 'Class means constant have not been computed...'
        assert self.unique_labels is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'precision': self.precision,
                        'means': self.means,
                        'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'MahalanobisDistance: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='MahaDist_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'MahalanobisDistance: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.precision = params_dict['precision']
        self.means = params_dict['means']
        self.unique_labels = params_dict['unique_labels']
    
    def get_scores( self, activations_eval: ArrayType, batch_size=1000 ):
        logger.info('MahalanobisDistance: Computing scores...')
        # activations_eval = activations_eval.clone()
        means_ = torch.stack(self.means, dim=0).T.unsqueeze(0).contiguous()
        class_centered_evaluations_ = (activations_eval.unsqueeze(2) - means_).contiguous()
        precision_ = self.precision.unsqueeze(0).contiguous()
        feat_data_loader = torch.utils.data.DataLoader(class_centered_evaluations_, batch_size=batch_size, shuffle=False)
        # scores_eval = (-1 * class_centered_evaluations_ * torch.matmul(precision_,class_centered_evaluations_)).sum(dim=1).amax(dim=1)
        # Check if CUDA (GPU support) is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Warning: CUDA not available, using CPU. This will be slower.")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision_ = precision_.to(device)
        score_list = []    
        for x in feat_data_loader:
            x = x.to(device)
            score_list.append( (-1 * x * torch.matmul(precision_,x)).sum(dim=1).amax(dim=1) )
        scores_eval = torch.cat(score_list, dim=0).cpu()
        return scores_eval
#%%
