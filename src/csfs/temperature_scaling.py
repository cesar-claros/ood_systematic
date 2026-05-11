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

class TemperatureScaling:
    def __init__(self,cf):
        self.cf = cf
        self.temperature = None

    def compute_temperature(self, 
                    logits_val: ArrayType, 
                    labels_val: ArrayType):
        logits_val = logits_val.clone()
        labels_val = labels_val.clone()
        logger.info("Fitting temperature to logits in the validation set...")
        self.temperature = torch.ones(1).requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=2000)

        def _eval():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits_val / self.temperature, labels_val)
            loss.backward()
            return loss

        optimizer.step(_eval)  # type: ignore

        self.temperature = self.temperature.item()
        logger.info(f'Temperature={self.temperature:.3f}')

    def get_scaled_softmax(self, logits_eval: ArrayType) -> ArrayType:
        # import torch
        logger.info('Temperature scale: Computing scores...')
        logits_eval = logits_eval.clone()
        softmax_rescaled = torch.softmax(logits_eval / self.temperature, dim=1, dtype=torch.float64)
        return softmax_rescaled
    
    def save_params(self, path:str|None=None, filename:str='Temperature_params'):
        assert self.temperature is not None, 'Temperature has not been computed'
        params_dict = {
                    'temperature': self.temperature,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Saving temperature scaling parameter in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='Temperature_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        params_dict = torch.load(path)
        self.temperature = params_dict['temperature']
        logger.info(f'Loading temperature scale from {path}')
        logger.info(f'temperature={self.temperature:.3f}')

# %%
