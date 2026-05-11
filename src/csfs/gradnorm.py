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

class GradNorm:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.num_classes = cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.fc = torch.nn.Linear(*self.w.shape[::-1])
        self.fc.weight.data[...] = self.w
        self.fc.bias.data[...] = self.b
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def get_scores( self, activations_eval: ArrayType, temperature:float=1.0, use_cuda=True ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.fc.to(self.device)
        self.logsoftmax.to(self.device)
        logger.info('GradNorm: Computing scores...')
        confs = []
        activations_eval = activations_eval.clone()
        for x in tqdm(activations_eval):
            targets = torch.ones((1, self.num_classes))
            input_var = Variable(x, requires_grad=True)
            targets = targets.to(self.device)
            input_var = input_var.to(self.device)    
            self.fc.zero_grad()
            logits = self.fc(input_var[None])
            # if self.study_name == 'dg':
            #     logits = logits[:,:-1]
            logits = logits/temperature
            loss = torch.mean(
                torch.sum( -targets * self.logsoftmax( logits ), dim=-1)
                )
            loss.backward()
            layer_grad_norm = torch.sum( torch.abs(self.fc.weight.grad.data) ).detach().cpu()
            confs.append(layer_grad_norm)
        return torch.Tensor(confs)

#%%

