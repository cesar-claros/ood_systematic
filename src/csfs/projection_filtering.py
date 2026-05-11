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

class ProjectionFiltering:

    def __init__(self, module, study_name:str, cf, mode:str='global'):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = self.cf.data.num_classes 
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.mode = mode
        self.condition_transform = ('ViT' in [self.cf.model.network.name, self.cf.model.network.backbone]) 
        logger.info('Using non-standardized inputs' if self.condition_transform else 'Standardizing inputs...')
        self.pca_estimator = None
        self.tss = None
        self.variance_explained = None
        

    def compute_ProjectionFiltering_params(self, activations_train: ArrayType,
                                labels:ArrayType|None = None, only_correct: bool = False,
                                variance_explained:float|None=0.90, ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        if self.variance_explained is None:
            logger.info(f'Projection Filtering: Setting minimum explained variance to {variance_explained}...')
            self.variance_explained = variance_explained
            assert 0<self.variance_explained<=1, f'Required explained variance (variance_explained={variance_explained}) should be between 0 and 1.'

        logger.info(f'Projection Filtering with {self.mode} mode: Fitting parameters...')
        activations_train = activations_train.clone()
        # dimensions = activations_train.shape[1]
        
        if 'global' in self.mode:
            if only_correct:
                self.labels = labels.clone() if labels is not None else None
                assert self.labels is not None, "Labels have not been provided..." 
                logits_train = activations_train@self.w.T + self.b
                softmax_train = F.softmax(logits_train, dim=1, dtype=torch.float64)
                predictions = softmax_train.max(dim=1).indices
                activations_train = activations_train[predictions==self.labels]
            self.tss = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
            self.tss.fit(activations_train)
            self.activations_scaled = self.tss.transform(activations_train) 
            # Principal Compoment Analysis 
            self.pca_estimator = PCA(n_components=None, svd_solver='full')
            self.pca_estimator.fit(self.activations_scaled)
        elif 'class' in self.mode:
            assert (labels is not None) and (len(labels)==len(activations_train)) , f'Labels are required to compute class subspaces.\n N_labels={len(labels)}, N_activations={len(activations_train)}'
            self.labels = labels.clone()
            # self.classes = self.labels.unique().to(int).numpy()
            self.pca_estimator = []
            self.tss = []
            for c in range(self.num_classes):
                labels_per_class_tensor = self.labels[self.labels==c]
                activations_per_class_tensor = activations_train[self.labels==c]
                logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                if only_correct:
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The PCA for this class uses all the activations.')
                tss_class = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
                tss_class.fit(activations_per_class_tensor)
                # activations_per_class_tensor_scaled = activations_per_class_tensor if self.condition_transform else tss_class.transform(activations_per_class_tensor)
                activations_per_class_tensor_scaled = tss_class.transform(activations_per_class_tensor)
                pca_estimator_sigma_W_class = PCA(n_components=None, svd_solver='full')
                pca_estimator_sigma_W_class.fit(activations_per_class_tensor_scaled)
                self.tss.append(tss_class)
                self.pca_estimator.append(pca_estimator_sigma_W_class)

    def get_projection(self, inputs:ArrayType, components:ArrayType, d:int|None=None, proj_mode:str='projection'):
        # self.proj_mode = proj_mode
        D = components.shape[0]
        if d is None:
            d = D
        elif d==0:
            logger.info(f'Reduced dimension cannot be d={d}. Reassign d=1...')
            d = 1
        # print(f'The number of components (d={d}) cannot be bigger that the dimensionality of the input (D={D}).')
        assert (d<=D) and (d>0) 
        components = (
                            components.to(torch.float16)
                            if inputs.dtype == torch.float16
                            else components
                    )
        components = components[:d,:] # = RankFeat?
        # components = components[:d,:] # dimensionality reduction
        if proj_mode == 'projection':
            return inputs @ components.T
        elif proj_mode == 'back-projection':
            return inputs @ components.T @ components
        else:
            raise ValueError(f"Unknown projection mode {proj_mode}")

    def get_backprojection(self, activations_eval: ArrayType,  ):
        
        logger.info(f'Projection Filtering: Computing scores...')
        # activations_eval = activations_eval.clone()
        # condition = 'ViT' in [self.cf.model.network.name, self.cf.model.network.backbone]
        if self.mode == 'global':
            # X_eval = activations_eval if self.condition_transform else self.tss.transform(activations_eval)
            X_eval = self.tss.transform(activations_eval)
            self.n_components = (self.pca_estimator.explained_variance_ratio_.cumsum(0)<=self.variance_explained).sum().item() + 1
            X_back_projected = self.get_projection(X_eval, self.pca_estimator.components_, d=self.n_components, proj_mode='back-projection')
            # X_back_projected = X_back_projected if self.condition_transform else self.tss.inverse_transform(X_back_projected)
            X_back_projected = self.tss.inverse_transform(X_back_projected)
            return X_back_projected
        elif self.mode == 'class':
            self.n_components = []
            X_back_projected_list = []
            for c in range(self.num_classes):
                # X_eval = activations_eval if self.condition_transform else self.tss[c].transform(activations_eval)
                X_eval = self.tss[c].transform(activations_eval)
                n_components_class = (self.pca_estimator[c].explained_variance_ratio_.cumsum(0)<=self.variance_explained).sum().item() + 1
                X_back_projected = self.get_projection(X_eval, self.pca_estimator[c].components_, d=n_components_class, proj_mode='back-projection')
                # X_back_projected = X_back_projected if self.condition_transform else self.tss[c].inverse_transform(X_back_projected)
                X_back_projected = self.tss[c].inverse_transform(X_back_projected)
                X_back_projected_list.append( X_back_projected )
                self.n_components.append( n_components_class )
            return X_back_projected_list
    
    def get_combined_backprojection(self, X_back_projected_list:List, combine:str|None=None, preds:ArrayType|None=None):
        assert self.mode == 'class', f'Combined backrpojection not defined for mode={self.mode}'
        assert isinstance(combine,str) , f'Combine has to be a string'
        N = X_back_projected_list[0].shape[0]
        if combine == 'prediction':
            X_back_projected = torch.vstack([X_back_projected_list[preds[t]][t] for t in range(N)])
        elif combine == 'average':
            # Backprojections for class averaged
            X_back_projected = []
            for t in range(N):
                avg_sampled = []
                for c in range(self.num_classes):
                    avg_sampled.append(X_back_projected_list[c][t])
                X_back_projected.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
            X_back_projected = torch.stack(X_back_projected, dim=0)
        logits_eval = X_back_projected @ self.w.T + self.b
        return X_back_projected, logits_eval

    def get_logits( self, activations_eval: ArrayType):
        logger.info(f'Projection Filtering: Computing logits...')
        activations_eval = activations_eval.clone()
        if self.mode == 'global':
            X_back_projected = self.get_backprojection(activations_eval)
            logits_eval = X_back_projected @ self.w.T + self.b
        elif self.mode == 'class':
            logit_eval_list = []
            X_back_projected = self.get_backprojection(activations_eval)
            for c in range(self.num_classes):
                logit_class_eval = X_back_projected[c] @ self.w.T[:,c] + self.b[c]
                logit_eval_list.append(logit_class_eval)
            logits_eval = torch.stack(logit_eval_list, dim=1)
        return logits_eval
    
    def get_scores( self, activations_eval:ArrayType, 
                            predictions_eval:ArrayType|None=None,
                            X_back_projected_eval:ArrayType|List|None=None,):
        logger.info(f'Projection Filtering: Computing scores...')
        if isinstance(activations_eval,ArrayType):
            activations_eval = activations_eval.clone()
        if self.mode == 'global':
            if X_back_projected_eval is None:
                X_back_projected = self.get_backprojection(activations_eval)
            else:
                X_back_projected = X_back_projected_eval
            scores = -1 * torch.norm(activations_eval - X_back_projected, p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)
        elif self.mode == 'class':
            scores_list = []
            if X_back_projected_eval is None:
                X_back_projected = self.get_backprojection(activations_eval)
            else:
                X_back_projected = X_back_projected_eval
            if isinstance(X_back_projected,List):            # X_back_projected = self.get_backprojection(activations_eval)
                for c in range(self.num_classes):
                    score = -1 * torch.norm(activations_eval - X_back_projected[c], p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)
                    scores_list.append(score)
                scores = torch.stack(scores_list, dim=1)
                if predictions_eval is None:
                    scores = scores.amax(dim=1) # pick best reconstruction
                else:
                    scores = torch.gather(scores, 1, predictions_eval.reshape(-1,1)).squeeze() # pick reconstruction guided by prediction
            else:
                scores = -1 * torch.norm(activations_eval - X_back_projected, p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)

        return scores
    
    def tune_hyperparameters( self, activations_train: ArrayType,
                                        activations_val: ArrayType, 
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType = None,
                                        only_correct: bool = False, 
                                        var_bounds:tuple=(0.85,0.99),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Projection Filtering: Tuning variance explained required to minimize AUGRC using the validation set...')
        activations_val = activations_val.clone()
        self.compute_ProjectionFiltering_params(activations_train, labels=labels_train, only_correct=only_correct)
        def _get_metric(explained_variance):
            self.variance_explained = explained_variance
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={'explained_variance': var_bounds,},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.variance_explained = best_params['explained_variance'] 
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
        

    def save_params(self, path:str|None=None, filename:str='ProjectionFiltering_params'):
        assert self.tss is not None, 'Standardizer has not been computed'
        assert self.pca_estimator is not None, 'PCA estimator has not been computed'
        assert self.variance_explained is not None, 'Variance explained has not been computed'
        params_dict = {
                        'tss': self.tss,
                        'pca_estimator': self.pca_estimator,
                        'variance_explained': self.variance_explained, }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'ProjectionFiltering Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='ProjectionFiltering_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'ProjectionFiltering Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.tss = params_dict['tss']
        self.pca_estimator = params_dict['pca_estimator']
        self.variance_explained = params_dict['variance_explained']


#%%
