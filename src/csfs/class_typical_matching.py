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

class ClassTypicalMatching:

    def __init__(self, module, study_name:str, cf, mode:str='global', alpha:float = 1.0):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        # self.ext_confid_name = self.cf.eval.ext_confid_name
        self.study_name = study_name
        # self.query_confids = self.cf.eval.confidence_measures
        # self.test_mcd_samples = self.cf.model.test_mcd_samples
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        self.mode = mode
        self.alpha = alpha
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.class_means = None
    
    def compute_CTM_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType,
                                    only_correct: bool = False ):
        logger.info("Class Typical Matching: Fitting parameters...")
        labels_train = labels_train.clone()
        if self.mode == 'global':
            self.class_means = []
            activations_train = activations_train.clone()
            for c in range(self.num_classes):
                activations_per_class_tensor = activations_train[labels_train==c]
                if only_correct:
                    labels_per_class_tensor = labels_train[labels_train==c]
                    logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
                else:
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
        elif self.mode == 'class':
            self.class_means = []
            for c in range(self.num_classes):
                activations_per_class_tensor = activations_train[c][labels_train==c]
                if only_correct:
                    labels_per_class_tensor = labels_train[labels_train==c]
                    logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
                else:
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
    
    def get_scores( self, backprojected_activations_eval: ArrayType|List[ArrayType], similarity:str='weight', batch_size=25):
        logger.info(f"Class Typical Matching: Computing scores using similarity={similarity} and mode={self.mode}...")
        
        if self.mode == 'global':
            X_back_projected = backprojected_activations_eval
            if X_back_projected.dim()==2:
                feat_data_loader = torch.utils.data.DataLoader(X_back_projected, batch_size=batch_size, shuffle=False)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    print("Warning: CUDA not available, using CPU. This will be slower.")
                if similarity == 'weight':
                    weights_expanded = self.w.T.unsqueeze(0).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append(F.cosine_similarity(x, weights_expanded, dim=1).amax(dim=1))
                elif similarity == 'mean' and self.class_means is not None:
                    class_means_ = torch.stack(self.class_means, dim=0).T.unsqueeze(0).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append(F.cosine_similarity(x, class_means_, dim=1).amax(dim=1))
                cosine_similarity_eval = torch.cat(sim, dim=0).cpu()
                return cosine_similarity_eval
            elif X_back_projected.dim()==3:
                logger.info(f"Class Typical Matching: Operating on distribution of activations...")
                feat_data_loader = torch.utils.data.DataLoader(X_back_projected, batch_size=batch_size, shuffle=False)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    print("Warning: CUDA not available, using CPU. This will be slower.")
                
                if similarity == 'weight':
                    weights_expanded = self.w.T.unsqueeze(0).unsqueeze(3).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append( F.cosine_similarity(x, weights_expanded, dim=1).amax(dim=1) )
                elif similarity == 'mean' and self.class_means is not None:
                    class_means_ = torch.stack(self.class_means, dim=0).T.unsqueeze(0).unsqueeze(3).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append( F.cosine_similarity(x, class_means_, dim=1).amax(dim=1) )
                
                cosine_similarity_eval = torch.cat(sim, dim=0).cpu()
                return cosine_similarity_eval.mean(dim=1)

        elif self.mode == 'class':
            self.n_components = []
            cosine_similarity_eval_list = []
            X_back_projected_list = backprojected_activations_eval
            for c in range(self.num_classes):
                X_back_projected = X_back_projected_list[c]
                if similarity == 'weight': 
                    cosine_similarity_eval_list.append(F.cosine_similarity(X_back_projected,self.w.T[:,c]))
                elif similarity == 'mean' and self.class_means is not None:
                    cosine_similarity_eval_list.append(F.cosine_similarity(X_back_projected,self.class_means[c]))
            cosine_similarity_eval = torch.stack(cosine_similarity_eval_list, dim=1)
        return cosine_similarity_eval.amax(dim=1)

    def save_params(self, path:str|None=None, filename:str='CTM_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.class_means is not None, 'Class means constant have not been computed...'
        # assert self.alpha is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'alpha': self.alpha,
                        'class_means': self.class_means,
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
        logger.info(f'CTM: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='CTM_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'CTM: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.alpha = params_dict['alpha']
        self.class_means = params_dict['class_means']

# %%
