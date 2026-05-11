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

class NNGuide:
# Nearest Neighbor Guidance

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
        self.bank_guide = None
        self.proportion = None

    def tune_hyperparameters(self,activations_train: ArrayType,
                                        activations_val: ArrayType,
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType|None = None,
                                        logits_train:ArrayType|None = None,
                                        k_clusters_bounds:tuple=(10,500),
                                        proportion_bounds:tuple=(0.1,0.5),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'NNGuide Score: Tunung hyper-parameters required to minimize AUGRC using the validation set...')
        def _get_metric(k_clusters, proportion):
            self.compute_NNGuide_params(activations_train,
                                        labels_train=labels_train,
                                        logits_train=logits_train,
                                        proportion=proportion,
                                        k_clusters=int(k_clusters) )
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'k_clusters': k_clusters_bounds, 
                                                'proportion': proportion_bounds,},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.compute_NNGuide_params(activations_train,
                                        labels_train=labels_train,
                                        logits_train=logits_train,
                                        proportion=best_params['proportion'],
                                        k_clusters=int(best_params['k_clusters']) )
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")

    def compute_NNGuide_params(self, activations_train: ArrayType,
                                        labels_train:ArrayType|None = None,
                                        logits_train:ArrayType|None = None,
                                        proportion:float=0.1,
                                        k_clusters:int=10):
        self.k_clusters = 10 if k_clusters is None else k_clusters
        self.proportion = 0.1 if proportion is None else proportion
        logger.info(f'NNGuide Score: Fitting parameters with alpha={self.proportion}...')
        activations_train = activations_train.clone()
        if logits_train is None:
            logits_train = activations_train @ self.w.T + self.b
        assert labels_train is not None, f'Labels need to be provided to sample features by classes...'
        labels_train = labels_train.clone()
        # proportion = 0.1 # Proportion of samples
        size_per_class = int((len(labels_train)/len(labels_train.unique())*proportion))
        logger.info(f'NNGuide Score: Each class is represented by {size_per_class} samples...')
        np.random.seed(12345) # For consistency
        idx_train = set([ x for y in (labels_train).unique() for x in (np.random.choice(np.argwhere(labels_train==y).squeeze(),size=size_per_class,replace=False)) ])
        sampled_activations_train = torch.stack([activations_train[idx] for idx in idx_train])
        bank_features = self.normalizer(sampled_activations_train)
        bank_logits = torch.stack([logits_train[idx] for idx in idx_train])
        bank_energy = torch.logsumexp(bank_logits, dim=1)
        self.bank_guide = bank_features * bank_energy[:, None]
        # activations_aug_train = self.transform_activations(activations_train) # add bias term
        # p = activations_aug_train.T @ activations_aug_train
        # self.p_parallel = torch.linalg.pinv(p)
        # self.p_bot = torch.eye(activations_aug_train.shape[1]) - self.p_parallel @ p
    
    def get_scores( self, activations_eval: ArrayType ):
        # logger.info('ViM: Computing scores')
        logger.info(f'NNGuide Score: Computing scores...')
        activations_eval = activations_eval.clone()
        logits_eval = activations_eval @ self.w.T + self.b
        energy_eval = torch.logsumexp(logits_eval, dim=1)
        features_eval = self.normalizer(activations_eval)
        # conf_eval = knn_score(self.bank_guide, features_eval, k=10)
        
        feat_data_loader = torch.utils.data.DataLoader(features_eval, batch_size=35, shuffle=False)
        # conf_eval = torch.Tensor(np.concatenate([self.knn_score(self.bank_guide, x, k=k_clusters) for x in feat_data_loader]))
        conf_eval = torch.cat([self.knn_score(self.bank_guide, x, k=self.k_clusters) for x in feat_data_loader])
        scores_eval = conf_eval * energy_eval
        
        return scores_eval
    
    def knn_score(self,bankfeas, queryfeas, k=100, min=False):

        inner_prod_topk = torch.topk(torch.mm(queryfeas, bankfeas.T), k=k)
        if min:
            # scores = np.array(D.min(axis=1))
            scores = inner_prod_topk.values.amin(dim=1)
        else:
            # scores = np.array(D.mean(axis=1))
            scores = inner_prod_topk.values.mean(dim=1)
        
        return scores
    
    def save_params(self, path:str|None=None, filename:str='NNGuide_params'):
        assert self.bank_guide is not None, 'bank_guide has not been computed'
        assert self.proportion is not None, 'proportion has not been computed'
        assert self.k_clusters is not None, 'k_clusters has not been computed'
        params_dict = {
                    'bank_guide': self.bank_guide,
                    'proportion': self.proportion,
                    'k_clusters': self.k_clusters,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'NNGuide Score: Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='NNGuide_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'NNGuide Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.bank_guide = params_dict['bank_guide']
        self.proportion = params_dict['proportion']
        self.k_clusters = params_dict['k_clusters']

#%%
