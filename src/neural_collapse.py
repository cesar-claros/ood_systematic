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

class NeuralCollapseMetrics:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.class_means = None
        self.global_mean = None
        self.nc_metrics = None
        self.class_variance = None
    
    def equiangular(self, C:ArrayType):
        n = C.shape[0]
        diagonal_mask = torch.eye(n, dtype=torch.bool)
        non_diagonal_mask = ~diagonal_mask
        row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
        # il = torch.tril_indices(row=len(C), col=len(C), offset=-1)
        equi_angle = torch.std(C[row_indices,col_indices])
        max_angle = torch.mean(torch.abs( C[row_indices,col_indices] + 1/(self.num_classes-1) )) 
        return equi_angle, max_angle
    
    def cdnv(self, class_means, class_variance):
        # FIXED 2026-08-30 (audit #11 metric review): CDNV (Galanti et al.)
        # divides by twice the SQUARED distance,
        # (V_c + V_c') / (2 ||mu_c - mu_c'||^2); the old code divided by
        # the unsquared distance. Not consumed by any claim in either
        # paper (dataset extra column only); fixed for future runs.
        mu_dist = pairwise_euclidean_distance(torch.vstack(class_means), zero_diagonal=False)
        variances = torch.stack(class_variance)
        var_sum = variances.unsqueeze(0) + variances.unsqueeze(-1)
        cdnv_matrix = var_sum/(2*mu_dist.pow(2))

        n = cdnv_matrix.shape[0]
        diagonal_mask = torch.eye(n, dtype=torch.bool)
        non_diagonal_mask = ~diagonal_mask
        row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
        return torch.mean(cdnv_matrix[row_indices,col_indices])
    
    # def equiangular(self, C:ArrayType):
    #     n = matrix.shape[0]
    #     diagonal_mask = torch.eye(n, dtype=torch.bool)
    #     non_diagonal_mask = ~diagonal_mask
    #     row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
    #     # il = torch.tril_indices(row=len(C), col=len(C), offset=-1)
    #     # return torch.std(C[il[0],il[1]])
    #     return torch.std(C[row_indices,col_indices])

    # def max_angular(self, C:ArrayType):
    #     # il = torch.tril_indices(row=len(C), col=len(C), offset=0)
    #     # return torch.mean( torch.abs( C[il[0],il[1]] + 1/(self.num_classes-1) ) )
    #     return torch.mean( torch.abs( C + 1/(self.num_classes-1) ) )
    
    def etf_simplex_difference(self, A:ArrayType, B:ArrayType|None=None):
        etf_simplex = (1/np.sqrt(self.num_classes-1)) * (torch.eye(self.num_classes)-(1/self.num_classes)*torch.ones((self.num_classes,self.num_classes)))
        if B is None:
            ABt = A@A.T
        else:
            ABt = A@B.T
        # torch.linalg.norm(, ord='fro')
        diff = (ABt)/torch.linalg.norm(ABt, ord='fro') - etf_simplex
        return torch.linalg.norm(diff, ord='fro')
    
    def compute_NeuralCollapse_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType,
                                    only_correct: bool = False ):
        logger.info("Neural Collapse Metrics: Computing global, class means, and metrics...")
        labels_train = labels_train.clone()
        activations_train = activations_train.clone()
        self.global_mean = activations_train.mean(dim=0)
        # self.unique_labels = torch.unique(labels_train)
        dim_B = self.global_mean.shape[0]
        sigma_B = torch.zeros(dim_B, dim_B)
        sigma_W = torch.zeros(dim_B, dim_B)
        self.class_means = []
        self.class_variance = []
        n_sigma_w = 0
        for c in range(self.num_classes):

            activations_per_class_tensor = activations_train[labels_train==c]
            if only_correct:
                labels_per_class_tensor = labels_train[labels_train==c]
                logits_per_class_tensor = activations_per_class_tensor @ self.w.T + self.b
                predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                    activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                else:
                    logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                class_mean = activations_per_class_tensor.mean(dim=0)
                self.class_means.append( class_mean )
            else:
                class_mean = activations_per_class_tensor.mean(dim=0)
                self.class_means.append( class_mean )
            # Compute metrics
            mu_cG = (class_mean - self.global_mean).reshape(-1,1)
            sigma_B = sigma_B + mu_cG @ mu_cG.T 
            # dim_W = class_mean.shape[0]
            # sigma_W = torch.zeros(dim_W, dim_W)
            H_k = torch.zeros(len(activations_per_class_tensor), dim_B)
            for j, h_ki in enumerate(activations_per_class_tensor):
                h_ki_c = (h_ki - class_mean).reshape(-1,1)
                H_k[j] = h_ki_c.T
                sigma_W = sigma_W + h_ki_c @ h_ki_c.T
            self.class_variance.append( torch.linalg.norm(H_k, dim=1, ord=2).pow(2).mean() )
            n_sigma_w += len(activations_per_class_tensor)
        # Variability Collapse (Within-class variation collapse)
        # FIXED 2026-08-30 (audit #11, NC1 normalization): sigma_W is the
        # average over ALL N samples (1/N), not 1/(N*K); the old extra 1/K
        # made var_collapse = NC1/C for the stated definition
        # Tr(Sigma_W Sigma_B^+)/C (verified analytically on the isotropic
        # ETF and empirically on 280 checkpoints, ratio == C per source).
        # The pseudoinverse is pinned (float64, hermitian, rtol 1e-6) to
        # match the audited pilot0 implementation: torch's default rtol on
        # the rank-(C-1) Sigma_B made high-D values pinv-fragile
        # (nc1_tinyimagenet_fragility report).
        K = self.num_classes
        # GUARD 2026-08-30: normalize by the number of samples actually
        # accumulated into sigma_W (n_sigma_w == N when only_correct is
        # False, the production path; under only_correct=True the old
        # N-based normalization would deflate sigma_W by the accuracy
        # factor - a latent hazard, never triggered in production).
        N = n_sigma_w
        sigma_B = (1/K) * sigma_B
        sigma_W = (1/N) * sigma_W
        var_collapse = (1/K)*torch.trace(
            sigma_W.double() @ torch.linalg.pinv(
                sigma_B.double(), rtol=1e-6, hermitian=True))
        # Equiangularity and Max-angle
        M = torch.vstack(self.class_means) - self.global_mean
        cos_uc = pairwise_cosine_similarity( M, zero_diagonal=False )
        cos_wc = pairwise_cosine_similarity( self.w, zero_diagonal=False )
        equiangular_uc, max_equiangular_uc = self.equiangular( cos_uc )
        equiangular_wc, max_equiangular_wc = self.equiangular( cos_wc )
        # Equinormality
        M_norm_class = torch.linalg.norm( M, dim=1, ord=2 )
        W_norm_class = torch.linalg.norm( self.w, dim=1, ord=2 )
        equinorm_uc = torch.std(M_norm_class)/torch.mean(M_norm_class)
        equinorm_wc = torch.std(W_norm_class)/torch.mean(W_norm_class)
        # Maximal-angle Equiangularity
        # max_equiangular_uc = self.max_angular( cos_uc )
        # max_equiangular_wc = self.max_angular( cos_wc )
        # Self-duality
        M_tilde = M/torch.linalg.norm(M, ord='fro')
        W_tilde = self.w/torch.linalg.norm(self.w, ord='fro')
        # ETF Simplex differences norm
        w_etf_diff = self.etf_simplex_difference(self.w/W_norm_class.unsqueeze(-1))
        M_etf_diff = self.etf_simplex_difference(M/M_norm_class.unsqueeze(-1))
        wM_etf_diff = self.etf_simplex_difference(self.w/W_norm_class.unsqueeze(-1), M/M_norm_class.unsqueeze(-1))
        # CDNV
        cdnv_score = self.cdnv(self.class_means, self.class_variance)
        # Bias collapse
        bias_collapse = torch.linalg.norm(self.b + self.w@self.global_mean, ord=2)
        self_duality = torch.linalg.norm(W_tilde-M_tilde, ord='fro').pow(2)
        
        logger.info(f"Variability collapse: {var_collapse}")
        logger.info(f"CDNV score: {cdnv_score}")
        logger.info(f"Bias collapse: {bias_collapse}")
        logger.info(f"Equiangularity means: {equiangular_uc}")
        logger.info(f"Equiangularity weights: {equiangular_wc}")
        logger.info(f"Max-Equiangularity means: {max_equiangular_uc}")
        logger.info(f"Max-Equiangularity weights: {max_equiangular_wc}")
        logger.info(f"Equinormality means: {equinorm_uc}")
        logger.info(f"Equinormality weights: {equinorm_wc}")
        logger.info(f"Self-duality: {self_duality}")
        logger.info(f"ETF difference (W): {w_etf_diff}")
        logger.info(f"ETF difference (M): {M_etf_diff}")
        logger.info(f"ETF difference (WMt): {wM_etf_diff}")

        self.nc_metrics = {'var_collapse':var_collapse,
                        'cdnv_score': cdnv_score,
                        'bias_collapse': bias_collapse,
                        'equiangular_uc':equiangular_uc,
                        'equiangular_wc':equiangular_wc,
                        'equinorm_uc':equinorm_uc,
                        'equinorm_wc':equinorm_wc,
                        'max_equiangular_uc':max_equiangular_uc,
                        'max_equiangular_wc':max_equiangular_wc,
                        'self_duality':self_duality,
                        'w_etf_diff':w_etf_diff,
                        'M_etf_diff':M_etf_diff,
                        'wM_etf_diff':wM_etf_diff,}

    def save_params(self, path:str|None=None, filename:str='NeuralCollapse_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.class_means is not None, 'Class means have not been computed...'
        assert self.global_mean is not None, 'Global mean have not been computed...'
        assert self.class_variance is not None, 'Class variances have not been computed...'
        assert self.nc_metrics is not None, 'Neural Collapse metrics have not been computed...'
        # assert self.alpha is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'global_mean': self.global_mean,
                        'class_means': self.class_means,
                        'class_variance': self.class_variance,
                        'nc_metrics': self.nc_metrics,
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
        logger.info(f'Neural Collapse Metrics: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='NeuralCollapse_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Neural Collapse Metrics: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.global_mean = params_dict['global_mean']
        self.class_means = params_dict['class_means']
        self.class_variance = params_dict['class_variance']
        self.nc_metrics = params_dict['nc_metrics']

# SLURM_EXPORT_ENV=NONE salloc --nodes=1 --partition=ececis_research --gres=gpu:p100   --mem-per-cpu=20G --cpus-per-task=12  --time=12:00:00
# %%
