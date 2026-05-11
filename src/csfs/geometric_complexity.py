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

class GeometricComplexity:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = cf.data.num_classes
        self.study_name = study_name
        self.model, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name,return_model=True)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.encoder = self.model.encoder
        self.h_grad_x = None
        self.g_grad_x = None
        # self.fc = torch.nn.Linear(*self.w.shape[::-1])
        # self.fc.weight.data[...] = self.w
        # self.fc.bias.data[...] = self.b
        # self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
    
    def _rademacher(self, shape, device, dtype):
        # ±1 with prob 1/2, returned as dtype
        return (torch.randint(0, 2, shape, device=device, dtype=torch.int8).to(dtype) * 2 - 1)

    @torch.no_grad()
    def _check_shapes(self, encoder, W_L, x):
        h = encoder(x)
        assert h.ndim == 2, f"encoder(x) must be [B, D], got {tuple(h.shape)}"
        assert W_L.ndim == 2, f"W_L must be [C, D], got {tuple(W_L.shape)}"
        assert W_L.shape[1] == h.shape[1], f"W_L.shape[1]={W_L.shape[1]} must equal D={h.shape[1]}"

    def get_grad_frob_norms(
        self,
        # encoder,
        # W_L: torch.Tensor,                  # [C, D]
        x: torch.Tensor,                    # [B, ...]
        n_probes_h: int = 8,
        n_probes_logits: int = 8,
        rademacher: bool = True,
        recompute_forward: bool = False,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        frob_h:      [B] estimate of || d h(x) / d x ||_F
        frob_logits: [B] estimate of || d (W_L h(x)+b) / d x ||_F
        """
        self.encoder.eval()
        W_L = self.w.to(device=x.device)

        # Quick shape sanity check (runs a forward once, no grad)
        self._check_shapes(self.encoder, W_L, x)

        x = x.requires_grad_(True)
        B = x.shape[0]

        acc_h = torch.zeros(B, device=x.device, dtype=x.dtype)
        acc_g = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not recompute_forward:
            h = self.encoder(x)  # [B, D]
            D = h.shape[1]
            C = W_L.shape[0]

            total_grads = n_probes_h + n_probes_logits
            done = 0

            # --- Hutchinson for h(x): E_v ||J_h^T v||^2
            for _ in range(n_probes_h):
                v = self._rademacher((B, D), x.device, h.dtype) if rademacher else torch.randn(B, D, device=x.device, dtype=h.dtype)
                s = (h * v).sum()  # sum_b <v_b, h_b>
                done += 1
                retain = (done < total_grads)
                (gx,) = torch.autograd.grad(s, x, retain_graph=retain, create_graph=create_graph)
                acc_h += gx.reshape(B, -1).pow(2).sum(dim=1)

            # --- Hutchinson for logits g(x): E_u ||J_g^T u||^2, with J_g^T u = J_h^T (u W_L)
            for _ in range(n_probes_logits):
                u = self._rademacher((B, C), x.device, h.dtype) if rademacher else torch.randn(B, C, device=x.device, dtype=h.dtype)
                v = u @ W_L  # [B, D]  (this is W_L^T u, in batch row-vector convention)
                s = (h * v).sum()
                done += 1
                retain = (done < total_grads)
                (gx,) = torch.autograd.grad(s, x, retain_graph=retain, create_graph=create_graph)
                acc_g += gx.reshape(B, -1).pow(2).sum(dim=1)

        else:
            # Recompute h each probe (less graph retention, more forward passes)
            for _ in range(n_probes_h):
                h = self.encoder(x)
                D = h.shape[1]
                v = self._rademacher((B, D), x.device, h.dtype) if rademacher else torch.randn(B, D, device=x.device, dtype=h.dtype)
                s = (h * v).sum()
                (gx,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=create_graph)
                acc_h += gx.reshape(B, -1).pow(2).sum(dim=1)

            C = W_L.shape[0]
            for _ in range(n_probes_logits):
                h = self.encoder(x)
                u = self._rademacher((B, C), x.device, h.dtype) if rademacher else torch.randn(B, C, device=x.device, dtype=h.dtype)
                v = u @ W_L
                s = (h * v).sum()
                (gx,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=create_graph)
                acc_g += gx.reshape(B, -1).pow(2).sum(dim=1)

        frob_h = (acc_h / max(n_probes_h, 1))
        frob_g = (acc_g / max(n_probes_logits, 1))

        if not create_graph:
            frob_h = frob_h.detach()
            frob_g = frob_g.detach()

        return frob_h, frob_g
    
    def compute_GC_params( self, datamodule ):
        logger.info('Geometric Complexity: Computing gradients...')
        dataloaders = datamodule.train_dataloader()
        frob_norm_list_set = [ self.get_grad_frob_norms( batch[0] ) for i,batch in enumerate(tqdm(dataloaders, position=0, leave=True)) ]
        h_grad_x = torch.concat([h  for h,_ in frob_norm_list_set])
        g_grad_x = torch.concat([g  for _,g in frob_norm_list_set])
        self.h_grad_x = h_grad_x.mean()
        self.g_grad_x = g_grad_x.mean()
    
    def save_params(self, path:str|None=None, filename:str='GC_params'):
        assert self.h_grad_x is not None, 'h_grad_x has not been computed'
        assert self.g_grad_x is not None, 'h_grad_x has not been computed'
        params_dict = {
                    'h_grad_x': self.h_grad_x,
                    'g_grad_x': self.g_grad_x
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Saving GC parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='GC_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Loading GC params from {path}')
        params_dict = torch.load(path)
        self.h_grad_x = params_dict['h_grad_x']
        self.g_grad_x = params_dict['g_grad_x']
        # logger.info(f'temperature={self.temperature:.3f}')


# %%
