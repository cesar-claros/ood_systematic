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

class EntropyScores:
    def __init__(self, cf, confid_func_name:str='renyi'):
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.confid_func_name = confid_func_name
        if confid_func_name == 'renyi':
            self.confid_func = scores_funcs.renyi_entropy
            # self.gamma_range =  np.arange(0.0, 1.0, 0.1)
        elif confid_func_name == 'generalized':
            self.confid_func = scores_funcs.generalized_entropy
            # self.gamma_range = np.arange(0.0, 1.0, 0.1)
        elif confid_func_name == 'tsallis':
            self.confid_func = scores_funcs.tsallis_entropy
            # self.gamma_range =  np.arange(0.0, 1.0, 0.1)
        else:
            raise NotImplementedError
        self.M = None
        self.gamma = None

    def compute_entropy_params(self, softmax:ArrayType, 
                                        residuals:ArrayType,
                                        gamma_bounds:tuple=(1e-6,0.999999),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Entropy Score ({self.confid_func_name}): Fitting parameters ...')
        softmax = softmax.clone()
        residuals = residuals.clone()
        # n, n_classes = softmax.shape
        M_bounds = (1, self.num_classes)
        def _get_metric(M, gamma):
            confidence = self.confid_func(softmax, gamma=gamma, M=int(M))
            stats_val_ = RiskCoverageStats(confids =  confidence, residuals = residuals)
            return -stats_val_.augrc
        
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'M': M_bounds, 
                                                'gamma': gamma_bounds,
                                                },
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization; suppress the objective's per-iteration
        # INFO chatter (errors still propagate and print).
        from src.csfs._utils import quiet_logging
        with quiet_logging():
            bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.M = int(best_params['M'])
        self.gamma = best_params['gamma']
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
    
    def get_scores(self, softmax_eval:ArrayType) -> ArrayType:
        logger.info(f'Entropy Score: Computing scores ...')
        softmax_eval = softmax_eval.clone()
        return self.confid_func(softmax_eval, gamma=self.gamma, M=self.M)

    
    def save_params(self, path:str|None=None, filename:str='EntropyScore_params'):
        assert self.M is not None, 'M has not been computed'
        assert self.gamma is not None, 'gamma has not been computed'
        params_dict = {
                    'M': self.M,
                    'gamma': self.gamma,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Entropy Score ({self.confid_func_name}): Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='EntropyScore_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Entropy Score ({self.confid_func_name}): Loading parameters from {path}')
        params_dict = torch.load(path)
        self.M = params_dict['M']
        self.gamma = params_dict['gamma']

#%%
