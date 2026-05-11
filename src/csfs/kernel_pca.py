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

class KernelPCA:
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
        self.X_ref, self.K_c, self.normalization, self.u_q = None, None, None, None
        self.explained_variance = None
        self.kernel = None
        self.gamma = None
    
    def _feat_normalization(self, X):
        X = X / (torch.norm(X, p=2, dim=-1, keepdim=True) + 1e-12)
        X = X.contiguous()
        return X
    
    def _rbf_kernel(self, X, Y, gamma=None):
        """
        RBF (Gaussian) kernel between X (n, d) and Y (m, d).
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        X_norm = (X**2).sum(dim=1).view(-1, 1)
        Y_norm = (Y**2).sum(dim=1).view(1, -1)
        K = X_norm + Y_norm - 2 * X @ Y.t()
        return torch.exp(-gamma * K)
    
    def _center_kernel(self, K, K_ref=None, center_on='all'):
        """
        Center the kernel matrix.
        - K: (n, m), kernel between data and reference.
        - K_ref: (m, m), kernel between reference and reference.
        - center_on: 'none', 'all', or 'mixed'.
        """
        n, m = K.shape
        if center_on == 'none':
            return K
        if center_on == 'all':
            # Center with respect to all data (classic double centering)
            K_mean_rows = K.mean(dim=1, keepdim=True)      # (n, 1)
            K_mean_cols = K.mean(dim=0, keepdim=True)      # (1, m)
            K_mean_all  = K.mean()
            # return (K_mean_rows + K_mean_cols - K_mean_all)
            return (K_mean_rows,K_mean_cols,K_mean_all)
        elif center_on == 'mixed':
            # Mixed centering: center K_{nm} using K_ref
            # K: (n, m), K_ref: (m, m)
            K_mean_rows = K.mean(dim=1, keepdim=True)      # (n, 1)
            K_ref_mean_cols = K_ref.mean(dim=0, keepdim=True)  # (1, m)
            K_ref_mean = K_ref.mean()
            # return (K_mean_rows + K_ref_mean_cols - K_ref_mean)
            return (K_mean_rows,K_ref_mean_cols,K_ref_mean)
        else:
            raise ValueError("center_on must be 'none', 'all', or 'mixed'.")

    def _kernel(self, X, Y):
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y, gamma=self.gamma)
        else:
            raise NotImplementedError("Only 'rbf' kernel implemented.")

    def _get_eigendecomposition(self, M):
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     eigvals, eigvecs = torch.linalg.eigh(M.to(device))
        #     eigvals, eigvecs = eigvals.cpu(), eigvecs.cpu()
        # else:
        #     eigvals, eigvecs = torch.linalg.eigh(M)
        # Returns eigenvalues (descending), eigenvectors
        try:
            eigvals, eigvecs = torch.linalg.eigh(M)
        except:
            n,m = M.shape
            epsilon = 1e-6
            M_p = M + epsilon*torch.eye(min(m,n),device=self.device) 
            eigvals, eigvecs = torch.linalg.eigh(M_p)
        idx = torch.argsort(eigvals, descending=True)
        return eigvals[idx], eigvecs[:, idx]
    
    def KPCA(self, X, ref_indices=None, m_samples=None, center_on=None, verbose=False):
        X = self._feat_normalization(X)
        
        if ref_indices is None:
            if m_samples is None:
                n = X.shape[0]
                raise ValueError("Specify either m_samples or ref_indices.")
            ref_indices = torch.randperm(n)[:m_samples]
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        X = X.to(self.device)
        X_ref = X[ref_indices]
        # Compute K_mm and K_nm
        K_mm = self._kernel(X_ref, X_ref)  # (m, m)
        # Eigendecomposition (only top n_components)
        evals, evecs = self._get_eigendecomposition(K_mm)
        evals = torch.maximum(evals, torch.tensor(1e-12))
        normalization = (evecs / torch.sqrt(evals))@ evecs.T
        # ---- low-rank projection
        K_nm = self._kernel(X, X_ref) @ normalization.T
        # K_c = self._center_kernel(K_nm, K_mm, center_on=center_on)
        K_c = K_nm.mean(dim=0)
        # # Explained variance selection
        # sigma = (K_nm-K_c).T@((K_nm-K_c))
        sigma = cov(K_nm-K_c, centered=True, rowvar=False)
        evals_full, evecs_full = self._get_eigendecomposition(sigma)
        evals_full_accuml = evals_full.cumsum(0)
        explained_variance_ratio_ = evals_full_accuml/evals_full_accuml[-1]
        n_components = (explained_variance_ratio_<= self.explained_variance).sum().item() + 1
        u_q = evecs_full[:,:n_components]
        if verbose:
            logger.info(f"n_components = {n_components}, variance ratio = {explained_variance_ratio_[n_components - 1]}")
        return X_ref.cpu(), K_c.cpu(), normalization.cpu(), u_q.cpu()

    def compute_KernelPCA_params(self, activations_train: ArrayType,
                                        labels:ArrayType|None = None, 
                                        only_correct: bool = False, 
                                        temperature:float|None=None, n_landmarks:int|None=None,
                                        explained_variance:float|None=None, gamma:float|None=None,
                                        center_on:str|None=None, kernel:str|None=None,
                                        verbose:bool=False):
        """
        X: (n, d) torch tensor.
        m_samples: number of landmark points for Nystrom.
        ref_indices: (m,) landmark indices. If None, random sample.
        """
        if verbose:
            logger.info(f'Kernel PCA with {self.mode} mode: Fitting parameters...')
        activations_train = activations_train.clone()
        self.explained_variance = 0.95 if explained_variance is None else explained_variance
        self.gamma = 0.2 if gamma is None else gamma 
        self.temperature = 1.0 if temperature is None else temperature
        self.n_landmarks = activations_train.shape[0]/self.num_classes if n_landmarks is None else n_landmarks 
        self.center_on = 'all' if center_on is None else center_on
        self.kernel = 'rbf' if kernel is None else kernel
        if 'global' in self.mode:
            if only_correct:
                assert self.labels is not None, "Labels have not been provided..."
                self.labels = labels.clone()
                logits_train = activations_train@self.w.T + self.b
                softmax_train = F.softmax(logits_train, dim=1, dtype=torch.float64)
                predictions = softmax_train.max(dim=1).indices
                correct_idx = predictions==self.labels
                activations_train = activations_train[correct_idx]
                energy_train = scores_funcs.energy(logits_train[correct_idx], temperature=self.temperature)
                energy_idx = energy_train.argsort()[:self.n_landmarks]
                self.X_ref, self.K_c, self.normalization, self.u_q = self.KPCA(activations_train, ref_indices=energy_idx, center_on=self.center_on)
            else:
                logits_train = activations_train@self.w.T + self.b
                energy_train = scores_funcs.energy(logits_train, temperature=self.temperature)
                energy_idx = energy_train.argsort()[:self.n_landmarks]
                self.X_ref, self.K_c, self.normalization, self.u_q = self.KPCA(activations_train, ref_indices=energy_idx, center_on=self.center_on, verbose=verbose)
        elif 'class' in self.mode:
            assert (labels is not None) and (len(labels)==len(activations_train)) , f'Labels are required to compute class subspaces.\n N_labels={len(labels)}, N_activations={len(activations_train)}'
            self.labels = labels.clone()
            self.X_ref = []
            self.K_c = []
            self.normalization = []
            self.u_q = []
            for c in range(self.num_classes):
                labels_per_class_tensor = self.labels[self.labels==c]
                activations_per_class_tensor = activations_train[self.labels==c]
                logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                if only_correct:
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                        logits_per_class_tensor = logits_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The PCA for this class uses all the activations.')
                energy_per_class_tensor = scores_funcs.energy(logits_per_class_tensor, temperature=self.temperature)
                energy_per_class_tensor_idx = energy_per_class_tensor.argsort()[:self.n_landmarks]
                X_ref, K_c, normalization, u_q = self.KPCA(activations_per_class_tensor, ref_indices=energy_per_class_tensor_idx, center_on=self.center_on, verbose=verbose)
                self.X_ref.append(X_ref)
                self.K_c.append(K_c)
                self.normalization.append(normalization)
                self.u_q.append(u_q)

    def get_scores(self, activations_eval,
                            predictions_eval:ArrayType|None=None,
                            combine:str|None=None):
        logger.info(f'Kernel PCA: Computing scores...')
        X = activations_eval.clone()
        X = self._feat_normalization(X)
        N = X.shape[0]
        if self.mode == 'global':
            K_nm = self._kernel(X, self.X_ref) @ self.normalization.T
            # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
            K_nm_backprojected = (self.u_q@self.u_q.T@(K_nm-self.K_c).T).T
            scores = -1 * torch.linalg.norm(K_nm_backprojected - (K_nm-self.K_c), ord=2, dim=1)
        elif self.mode == 'class':
            if combine=='average':
                K_nm_backprojected_list = []
                K_nm_centered_list = []
                for c in range(self.num_classes):
                    K_nm = self._kernel(X, self.X_ref[c]) @ self.normalization[c].T
                    # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
                    K_nm_centered = K_nm - self.K_c[c]
                    K_nm_backprojected = (self.u_q[c]@self.u_q[c].T@(K_nm_centered).T).T
                    K_nm_backprojected_list.append(K_nm_backprojected)
                    K_nm_centered_list.append(K_nm_centered)
                K_nm_backprojected_avg = []
                for t in range(N):
                    avg_sampled = []
                    for c in range(self.num_classes):
                        avg_sampled.append(K_nm_backprojected_list[c][t])
                    K_nm_backprojected_avg.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
                K_nm_backprojected_avg = torch.stack(K_nm_backprojected_avg, dim=0)

                K_nm_centered_avg = []
                for t in range(N):
                    avg_sampled = []
                    for c in range(self.num_classes):
                        avg_sampled.append(K_nm_centered_list[c][t])
                    K_nm_centered_avg.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
                K_nm_centered_avg = torch.stack(K_nm_centered_avg, dim=0)
                scores = -1 * torch.linalg.norm(K_nm_backprojected_avg - K_nm_centered_avg, ord=2, dim=1)
            
            else:
                scores_list = []
                for c in range(self.num_classes):
                    K_nm = self._kernel(X, self.X_ref[c]) @ self.normalization[c].T
                    # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
                    K_nm_backprojected = (self.u_q[c]@self.u_q[c].T@(K_nm-self.K_c[c]).T).T
                    score = -1 * torch.linalg.norm(K_nm_backprojected - (K_nm-self.K_c[c]), ord=2, dim=1)
                    scores_list.append(score)
                scores = torch.stack(scores_list, dim=1)
                if predictions_eval is None:
                    scores = scores.amax(dim=1) # pick best reconstruction
                else:
                    scores = torch.gather(scores, 1, predictions_eval.reshape(-1,1)).squeeze() # pick reconstruction guided by prediction

        return scores

    def tune_hyperparameters( self, activations_train: ArrayType,
                                        activations_val: ArrayType, 
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType = None, 
                                        only_correct: bool = False, 
                                        temperature:float|None=None,
                                        center_on:str = 'all',
                                        kernel:str = 'rbf',
                                        var_bounds:tuple = (0.85, 0.99),
                                        gamma_bounds:tuple = (0.10, 1.0),
                                        landmarks_bounds:tuple = (2000, 5000),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Kernel PCA: Tuning hyper-parameters required to minimize AUGRC using the validation set...')
        activations_val = activations_val.clone()
        def _get_metric(explained_variance, gamma, n_landmarks):
            self.compute_KernelPCA_params(activations_train,
                                        labels=labels_train, 
                                        only_correct=False, 
                                        temperature=temperature,
                                        n_landmarks=int(n_landmarks),
                                        explained_variance=explained_variance, 
                                        gamma=gamma,
                                        center_on=center_on,
                                        kernel=kernel)
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'explained_variance': var_bounds, 
                                                'gamma': gamma_bounds, 
                                                'n_landmarks': landmarks_bounds},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.compute_KernelPCA_params(activations_train,
                                        labels=labels_train, 
                                        only_correct=False, 
                                        temperature=temperature,
                                        n_landmarks=int(best_params['n_landmarks']),
                                        explained_variance=best_params['explained_variance'], 
                                        gamma=best_params['gamma'],
                                        center_on=center_on,
                                        kernel=kernel,
                                        verbose=True)
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
    
    def save_params(self, path:str|None=None, filename:str='KernelPCA_params'):
        assert self.X_ref is not None, 'X_ref has not been computed'
        assert self.K_c is not None, 'K_c has not been computed'
        assert self.normalization is not None, 'normalization has not been computed'
        assert self.u_q is not None, 'u_q has not been computed'
        assert self.kernel is not None, 'kernel has not been defined'
        assert self.gamma is not None, 'gamma has not been defined'
        params_dict = {
                        'X_ref': self.X_ref,
                        'K_c': self.K_c,
                        'normalization': self.normalization,
                        'u_q':self.u_q,
                        'kernel':self.kernel,
                        'gamma':self.gamma,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'KernelPCA Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='KernelPCA_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'KernelPCA Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.X_ref = params_dict['X_ref']
        self.K_c = params_dict['K_c']
        self.normalization = params_dict['normalization']
        self.u_q = params_dict['u_q']
        self.kernel = params_dict['kernel']
        self.gamma = params_dict['gamma']


#%%
