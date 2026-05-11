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

class pNML:
# Predictive Normalized Maximum Likelihood

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
        self.p_parallel = None
        self.p_bot = None
        # self.probs = None
    
    def softmax(self, logits: ArrayType) -> ArrayType:
        """Compute softmax values for each sets of scores in x."""
        logits = logits.clone()
        logits_max = torch.max(logits, dim=1, keepdims=True).values
        e_x = torch.exp(logits - logits_max)
        return e_x / e_x.sum(dim=1, keepdims=True)
    
    def transform_activations(self, activations: ArrayType) -> ArrayType:
        activations = activations.clone()
        n, m = activations.shape
        norm = torch.linalg.norm(activations, dim=1, keepdims=True)
        activations = activations / norm
        ones = torch.ones((n, 1))
        return torch.hstack((ones,activations))

    def compute_pNML_params(self, activations_train: ArrayType) -> ArrayType:
        # Calc x_Bot
        logger.info('pNML Score: Fitting parameters...')
        activations_train = activations_train.clone()
        activations_aug_train = self.transform_activations(activations_train) # add bias term
        p = activations_aug_train.T @ activations_aug_train
        self.p_parallel = torch.linalg.pinv(p)
        self.p_bot = torch.eye(activations_aug_train.shape[1]) - self.p_parallel @ p

    def get_scores(self, activations_eval: ArrayType, output_predictions:bool=False) -> ArrayType:
        """
        Calculate the genie probability
        :param activations_eval: the dataset to evaluate: (n,m)
        :param probs: The model probability of the dataset: (n,)
        :param p_parallel: projection matrix, the parrallel component
        :param p_bot: projection matrix, the orthogonal component
        :return:
        """
        logger.info('pNML Score: Computing scores...')
        activations_eval = activations_eval.clone()
        bias_weight = torch.hstack( (self.b[:,None], self.w) )
        activations_aug_eval = self.transform_activations(activations_eval) # add bias term
        logits_eval = activations_aug_eval @ bias_weight.T
        probs = self.softmax(logits_eval)
        n, n_classes = probs.shape

        # Calc energy of each component
        # x_parallel_square = np.array([x @ self.p_parallel @ x.T for x in activations_aug])
        x_parallel_square = torch.sum((activations_aug_eval @ self.p_parallel) * activations_aug_eval, dim=1)
        # x_bot_square = np.array([x @ self.p_bot @ x.T for x in activations_aug])
        x_bot_square = torch.sum((activations_aug_eval @ self.p_bot) * activations_aug_eval, dim=1)
        # x_t_g = np.maximum(x_bot_square, x_parallel_square / (1 + x_parallel_square))
        x_t_g = torch.maximum(x_bot_square, x_parallel_square / (1 + x_parallel_square))
        # x_t_g = np.expand_dims(x_t_g, -1)
        x_t_g_repeated = x_t_g[:,None].repeat(1,self.num_classes)

        # Genie prediction
        genie_predictions = probs / (probs + (1 - probs) * (probs ** x_t_g_repeated))

        # Regret
        nfs = genie_predictions.sum(dim=1)
        regrets = np.log(nfs) / np.log(self.num_classes)

        # pNML probability assignment
        # pnml_prediction = genie_predictions / np.repeat(
        #     np.expand_dims(nfs, -1), n_classes, axis=1
        # )
        if output_predictions:
            pnml_prediction = genie_predictions / nfs[:,None].repeat(1,self.num_classes)
            return -regrets, pnml_prediction
        else:
            return -regrets
    
    def save_params(self, path:str|None=None, filename:str='pNML_params'):
        assert self.p_parallel is not None, 'p_parallel matrix has not been computed'
        assert self.p_bot is not None, 'p_bot matrix has not been computed'
        params_dict = {
                    'p_parallel': self.p_parallel,
                    'p_bot': self.p_bot,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'pNML Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='pNML_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'pNML Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.p_parallel = params_dict['p_parallel']
        self.p_bot = params_dict['p_bot']
#%%

