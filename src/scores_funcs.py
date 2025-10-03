
import os
import copy
# import numpy as np
import torch
# from torch_pca import PCA
# from fd_shifts import analysis, models, reporting, configs
# from fd_shifts.loaders import dataset_collection
# from fd_shifts.utils import exp_utils
# from fd_shifts.models import get_model
# from fd_shifts.loaders.data_loader import FDShiftsDataLoader
import logging
from fd_shifts import logger
# from pytorch_grad_cam.grad_cam import GradCAM
from tqdm import tqdm
# from utils import get_study_name, get_conf, get_model_and_last_layer,\
                #  save_evaluations, load_evaluations, is_dropout_enabled
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast
from torch.multiprocessing import Process, Queue, set_start_method
#%%
ArrayType = torch.Tensor
T = TypeVar(
    "T", Callable[[ArrayType], ArrayType], Callable[[ArrayType, ArrayType], ArrayType]
)

#%%

def _assert_softmax_logit_finite(softmax_logit: ArrayType):
    assert torch.isfinite(softmax_logit).all(), "NaN or INF in softmax/logit output"

def _assert_softmax_logit_distribution(softmax_logit: ArrayType):
    assert softmax_logit.ndimension()>2, "softmax/logit distribution needs to be shaped\
                                    as (N,C,M), where N is the number of instances,\
                                    C is the number of classes, and M is the number\
                                    of Monte Carlo dropout samples"
    assert softmax_logit.shape[2] > 1, 'softmax/logit distribution has only one Monte\
                                        Carlo sample'
    

def _assert_softmax_numerically_stable(softmax: ArrayType):
    msr,indices = softmax.max(dim=1)
    errors = (msr == 1) & ((softmax > 0) & (softmax < 1)).any(dim=1)

    if softmax.dtype != torch.float64:
        logging.warning("Softmax is not 64bit, not checking for numerical stability")
        return

    # alert if more than 10% are erroneous
    assert (
        errors.float().mean() < 0.1
    ), f"Numerical errors in softmax: {errors.float().mean() * 100:.2f}%"


def validate_softmax(func: T) -> T:
    """Decorator to validate softmax before using it for computations

    Args:
        func (T): callable to decorate

    Returns:
        decorated callable
    """

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            _assert_softmax_logit_finite(arg)
            _assert_softmax_numerically_stable(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)

def validate_logit(func: T) -> T:
    """Decorator to validate logit before using it for computations
    Args:
        func (T): callable to decorate

    Returns:
        decorated callable
    """

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            _assert_softmax_logit_finite(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)

def validate_softmax_logit_distribution(func: T) -> T:
    """Decorator to validate softmax before using it for computations
    Args:
        func (T): callable to decorate

    Returns:
        decorated callable
    """

    def _inner_wrapper(*args: ArrayType, **kwargs) -> ArrayType:
        for arg in args:
            if torch.is_tensor(arg):
                _assert_softmax_logit_finite(arg)
                _assert_softmax_logit_distribution(arg)
        return func(*args, **kwargs)

    return cast(T, _inner_wrapper)

#%%
# MSR
# @validate_softmax
def maximum_softmax_response(softmax: ArrayType) -> ArrayType:
    """Maximum softmax probability CSF/Maximum logit score CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes or logits of shape NxC
        T (float): temperature scaling

    Returns:
        maximum score array of shape N
    """
    softmax = softmax.clone()
    msr = torch.max(softmax,dim=1).values
    return msr
#%%     
# PE
@validate_softmax
def predictive_entropy(softmax: ArrayType) -> ArrayType:
    """Predictive entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC
        T (float): temperature scaling
    Returns:
        pe score array of shape N
    """
    softmax = softmax.clone()
    epsilon = torch.finfo(softmax.dtype).eps
    pred_ent = - torch.sum( (softmax * ( torch.log(softmax + epsilon) )), dim=1 ) 
    return -pred_ent
    # return Categorical(probs = softmax).entropy()

# predictive_entropy(results_dataset['softmax'])
#%%     
# GenEnt
@validate_softmax
def generalized_entropy(softmax: ArrayType, gamma:float=0.1, M:int|None=None) -> ArrayType:
    """Predictive entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC
        T (float): temperature scaling
    Returns:
        pe score array of shape N
    """
    if M is None:
        M = softmax.shape[1]
    softmax = softmax.clone()
    sorted_softmax = torch.sort( softmax, descending=True, dim=1 ).values # descending order
    sorted_softmax = sorted_softmax[:,:M]
    gen_ent = torch.sum( (sorted_softmax * ( 1-sorted_softmax ))**gamma, dim=1)
    return -gen_ent

# predictive_entropy(results_dataset['softmax'])
#%%     
# RenyiEnt
@validate_softmax
def renyi_entropy(softmax: ArrayType, gamma:float=0.1, M:int|None=None) -> ArrayType:
    """Predictive entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC
        T (float): temperature scaling
    Returns:
        pe score array of shape N
    """
    if M is None:
        M = softmax.shape[1]
    epsilon = torch.finfo(softmax.dtype).eps
    softmax = softmax.clone()
    sorted_softmax = torch.sort( softmax, descending=True, dim=1 ).values # descending order
    sorted_softmax = sorted_softmax[:,:M]
    if gamma == 1.0:
        return predictive_entropy(sorted_softmax)
    elif gamma == 2.0:
        return predictive_collision_entropy(sorted_softmax)
    else:
        ren_ent = (gamma/(1-gamma)) * torch.log( torch.norm(sorted_softmax, p=gamma, dim=1 ) + epsilon ) 
        return -ren_ent
 
# renyi_entropy(results_dataset['softmax'])
#%%     
# RenyiEnt
@validate_softmax
def tsallis_entropy(softmax: ArrayType, gamma:float=0.1, M:int|None=None) -> ArrayType:
    """Predictive entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC
        T (float): temperature scaling
    Returns:
        pe score array of shape N
    """
    if M is None:
        M = softmax.shape[1]
    softmax = softmax.clone()
    sorted_softmax = torch.sort( softmax, descending=True, dim=1 ).values # descending order
    sorted_softmax = sorted_softmax[:,:M]
    if gamma == 1.0:
        return predictive_entropy(sorted_softmax)
    else:    
        tsa_ent = (1/(gamma-1)) * (1 - torch.sum(sorted_softmax**gamma, dim=1 )) 
        return -tsa_ent
 
# renyi_entropy(results_dataset['softmax'])

#%%
# MLS
@validate_logit
def maximum_logit_score(logit: ArrayType, temperature:float=1) -> ArrayType:
    """Maximum softmax probability CSF/Maximum logit score CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes or logits of shape NxC

    Returns:
        maximum score array of shape N
    """
    logit = logit.clone()
    mls = torch.max(logit/temperature, dim=1).values
    return mls

# maximum_logit_score(results_dataset['logits'])
#%%
# MLS
@validate_logit
def maximum_cosine_similarity(similarity: ArrayType) -> ArrayType:
    """Maximum softmax probability CSF/Maximum logit score CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes or logits of shape NxC

    Returns:
        maximum score array of shape N
    """
    similarity = similatiry.clone()
    mcs = torch.max(similarity, dim=1).values
    return mcs

#%%
# PCE
@validate_softmax
def predictive_collision_entropy(softmax: ArrayType) -> ArrayType:
    """Predictive collision entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC

    Returns:
        pe score array of shape N
    """
    # epsilon = np.finfo(softmax.dtype).eps
    epsilon = torch.finfo(softmax.dtype).eps
    softmax = softmax.clone()
    pred_col_ent = -torch.log( torch.sum( torch.square(softmax) , dim=1 ) + epsilon )
    return -pred_col_ent

#%%
# GE
@validate_softmax
def guessing_entropy(softmax: ArrayType, M:int|None=None) -> ArrayType:
    """Predictive collision entropy CSF
    Args:
        softmax (ArrayType): array-like containing softmaxes of shape NxC

    Returns:
        pe score array of shape N
    """
    if M is None:
        M = softmax.shape[1]
    # epsilon = np.finfo(softmax.dtype).eps
    # epsilon = torch.finfo(softmax.dtype).eps
    softmax = softmax.clone()
    k_guesses = torch.tile(
        torch.tensor([i+1 for i in range(softmax.shape[1])]), (softmax.shape[0],1)
        )
    sorted_softmax = torch.sort( softmax, descending=True, dim=1 ).values # descending order
    sorted_softmax = sorted_softmax[:,:M]
    k_guesses = k_guesses[:,:M]
    guess_ent = torch.sum( k_guesses * sorted_softmax, dim=1 ) 
    return -guess_ent

#%%
# Energy
@validate_logit
def energy(logit: ArrayType, temperature:float=1) -> ArrayType:
    """Energy CSF
    Args:
        logit (ArrayType): array-like containing softmaxes of shape NxC
        T (float): temperature
    Returns:
        pe score array of shape N
    """
    energy_score = -temperature*torch.logsumexp(logit/temperature, dim=1)
    # epsilon = np.finfo(logit.dtype).eps
    # epsilon = torch.finfo(softmax.dtype).eps
    return -energy_score

#%%
# MCD-function
@validate_softmax_logit_distribution
def mcd_function(func, logit_softmax_distribution: ArrayType, **kwargs):
    mean_logit_softmax_distribution = logit_softmax_distribution.mean(dim=2)
    if 'temperature' in kwargs.keys():
        temperature = kwargs['temperature']
        mcd_score = func(mean_logit_softmax_distribution, temperature=temperature)
    elif 'similarity' in kwargs.keys():
        similarity = kwargs['similarity']
        mcd_score = func(mean_logit_softmax_distribution, similarity=similarity)
    elif 'logits_eval' in kwargs.keys():
        logits_eval = kwargs['logits_eval']
        mean_logit_eval_distribution = logits_eval.mean(dim=2)
        mcd_score = func(mean_logit_softmax_distribution, logits_eval=mean_logit_eval_distribution)
    elif 'use_cuda' in kwargs.keys():
        use_cuda = kwargs['use_cuda']
        mcd_score = func(mean_logit_softmax_distribution, use_cuda=use_cuda)
    elif 'combine' in kwargs.keys():
        combine = kwargs['combine']
        mcd_score = func(mean_logit_softmax_distribution, combine=combine)    
    elif ('predictions_eval' in kwargs.keys()) and ('X_back_projected_eval' in kwargs.keys()):
        X_back_projected_eval = kwargs['X_back_projected_eval']
        predictions_eval = kwargs['predictions_eval']
        mcd_score = func(mean_logit_softmax_distribution, predictions_eval=predictions_eval, X_back_projected=X_back_projected)
    elif ('X_back_projected_eval' in kwargs.keys()):
        X_back_projected_eval = kwargs['X_back_projected_eval']
        mcd_score = func(mean_logit_softmax_distribution, X_back_projected_eval=X_back_projected_eval)
    elif 'predictions_eval' in kwargs.keys():
        predictions_eval = kwargs['predictions_eval']
        mcd_score = func(mean_logit_softmax_distribution, predictions_eval=predictions_eval)
    elif ('use_cuda' in kwargs.keys()) and ('temperature' in kwargs.keys()):
        use_cuda = kwargs['use_cuda']
        temperature = kwargs['temperature']
        mcd_score = func(mean_logit_softmax_distribution, use_cuda=use_cuda, temperature=temperature)
    else:
        mcd_score = func(mean_logit_softmax_distribution)
    return mcd_score


#%%
# MCD-Expected-function
@validate_softmax_logit_distribution
def mcd_expected_function(func, logit_softmax_distribution: ArrayType, **kwargs):
    mcd_repetitions = logit_softmax_distribution.shape[2]
    if 'temperature' in kwargs.keys():
        temperature = kwargs['temperature']
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j], temperature=temperature) for j in range(mcd_repetitions) ])
    elif 'similarity' in kwargs.keys():
        similarity = kwargs['similarity']
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j], similarity=similarity) for j in range(mcd_repetitions) ])
    elif 'logits_eval' in kwargs.keys():
        logits_eval = kwargs['logits_eval']
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j], logits_eval=logits_eval[:,:,j]) for j in range(mcd_repetitions) ])
    elif 'predictions_eval' in kwargs.keys():
        predictions_eval = kwargs['predictions_eval']
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j], predictions_eval=predictions_eval[:,j]) for j in range(mcd_repetitions) ])
    elif 'use_cuda' in kwargs.keys():
        use_cuda = kwargs['use_cuda']
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j], use_cuda=use_cuda) for j in range(mcd_repetitions) ])
    else:
        mcd_dist = torch.vstack([func(logit_softmax_distribution[:,:,j]) for j in range(mcd_repetitions) ])
    mcd_expected_score = mcd_dist.mean(dim=0)
    return mcd_expected_score

#%%
# MCD-SV
@validate_softmax_logit_distribution
def mcd_softmax_variance(softmax_distribution: ArrayType,) -> ArrayType:
    """Maximum softmax probability CSF based on Monte Carlo dropout samples
    Args:
        softmax_distribution (ArrayType): array-like containing softmaxes or 
        logits of shape NxCxM, where M is the number of Monte Carlo samples 

    Returns:
        maximum score array of shape N
    """
    var_log_softmax_distribution = (torch.log(softmax_distribution)).var(dim=2)
    return torch.mean(-var_log_softmax_distribution, dim=1)

# mcd_softmax_variance(results_dataset['softmax_dist'])

#%%
# MCD-WAIC
@validate_softmax_logit_distribution
def mcd_watanabe_aic(softmax_distribution: ArrayType,) -> ArrayType:
    """Maximum softmax probability CSF based on Monte Carlo dropout samples
    Args:
        softmax_distribution (ArrayType): array-like containing softmaxes or 
        logits of shape NxCxM, where M is the number of Monte Carlo samples 

    Returns:
        maximum score array of shape N
    """
    mean_softmax_distribution = softmax_distribution.mean(dim=2)
    var_log_softmax_distribution = (torch.log(softmax_distribution)).var(dim=2)
    waic = torch.mean( torch.log(mean_softmax_distribution) - var_log_softmax_distribution, dim=1)
    return -waic

# mcd_watanabe_aic(results_dataset['softmax_dist']).shape
#%%
# MCD-MI
@validate_softmax_logit_distribution
def mcd_mutual_information(softmax_distribution: ArrayType) -> ArrayType:
    """Expected entropy CSF based on Monte Carlo dropout samples
    Args:
        softmax_distribution (ArrayType): array-like containing softmaxes or 
        logits of shape NxCxM, where M is the number of Monte Carlo samples

    Returns:
        pe score array of shape N
    """
    mcd_pe = -mcd_function(predictive_entropy, softmax_distribution )
    mcd_ee = -mcd_expected_function(predictive_entropy, softmax_distribution )
    mutual_information = mcd_pe - mcd_ee 
    return -mutual_information

# mcd_mutual_information(results_dataset['softmax_dist'])

# #%%
# # MCD-PE
# @validate_softmax_logit_distribution
# def mcd_predictive_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Predictive entropy CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples

#     Returns:
#         pe score array of shape N
#     """
#     mean_softmax_distribution = softmax_distribution.mean(dim=2)
#     epsilon = torch.finfo(softmax_distribution.dtype).eps
#     predictive_entropy = -torch.sum(
#                                     mean_softmax_distribution * ( torch.log(mean_softmax_distribution + epsilon) ),
#                                     dim=1
#                                     )
#     # epsilon = np.finfo(mean_softmax_distribution.dtype).eps
#     return -predictive_entropy

# mcd_predictive_entropy(results_dataset['softmax_dist'])
#%%
# MCD-EPE
# @validate_softmax_logit_distribution
# def mcd_expected_predictive_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Expected entropy CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples

#     Returns:
#         pe score array of shape N
#     """
#     # epsilon = np.finfo(softmax_distribution.dtype).eps
#     epsilon = torch.finfo(softmax_distribution.dtype).eps
#     predictive_entropy = -torch.sum(
#                             softmax_distribution * ( torch.log(softmax_distribution + epsilon) ),
#                             dim=1,
#                         ) 
#     return torch.mean(-predictive_entropy, dim=1)

# mcd_expected_predictive_entropy(results_dataset['softmax_dist'])
#%%



# mcd_predictive_collision_entropy(results_dataset['softmax_dist'])
#%%
# MCD-ECE
# @validate_softmax_logit_distribution
# def mcd_expected_predictive_collision_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Expected entropy CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples

#     Returns:
#         pe score array of shape N
#     """

#     # epsilon = np.finfo(softmax_distribution.dtype).eps
#     epsilon = torch.finfo(softmax_distribution.dtype).eps
#     ce_distribution = -torch.log( torch.sum( torch.square(softmax_distribution), dim=1 )
#                                 + epsilon ) 
#     return torch.mean(ce_distribution, dim=1)

# mcd_expected_collision_entropy(results_dataset['softmax_dist'])

#%%
# MCD-MLS
# @validate_softmax_logit_distribution
# def mcd_maximum_logit_score(logit_distribution: ArrayType, temperature:float=1) -> ArrayType:
#     """Maximum softmax probability CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples 

#     Returns:
#         maximum score array of shape N
#     """
#     # torch.max(logit, dim=1).values
#     mean_logit_distribution = logit_distribution.mean(dim=2)
#     return torch.max(mean_logit_distribution/temperature, dim=1).values

# mcd_maximum_logit_score(results_dataset['logits_dist'])


# energy(results_dataset['logits'])

#%%
# MCD-Energy
# @validate_softmax_logit_distribution
# def mcd_energy(logit_distribution: ArrayType, temperature:float=1) -> ArrayType:
#     """Energy CSF
#     Args:
#         logit (ArrayType): array-like containing softmaxes of shape NxC
#         T (float): temperature
#     Returns:
#         pe score array of shape N
#     """
#     mean_logit_distribution = logit_distribution.mean(dim=2)
#     return temperature*torch.logsumexp(mean_logit_distribution/temperature, dim=1)

# mcd_energy(results_dataset['logits_dist'])
#%%
# MCD-EE
# @validate_softmax_logit_distribution
# def mcd_expected_energy(logit_distribution: ArrayType, temperature:float=1) -> ArrayType:
#     """Energy CSF
#     Args:
#         logit (ArrayType): array-like containing softmaxes of shape NxC
#         T (float): temperature
#     Returns:
#         pe score array of shape N
#     """
#     energy_distribution = temperature*torch.logsumexp(logit_distribution/temperature, dim=1)
#     return energy_distribution.mean(dim=1)
#%%
# MCD-MSR
# @validate_softmax_logit_distribution
# def mcd_maximum_softmax_response(softmax_distribution: ArrayType,) -> ArrayType:
#     """Maximum softmax probability CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples 

#     Returns:
#         maximum score array of shape N
#     """
#     mean_softmax_distribution = softmax_distribution.mean(dim=2)
#     return torch.max(mean_softmax_distribution, dim=1).values

#%%
# MCD-PCE
# @validate_softmax_logit_distribution
# def mcd_predictive_collision_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Predictive entropy CSF based on Monte Carlo dropout samples
#     Args:
#         softmax_distribution (ArrayType): array-like containing softmaxes or 
#         logits of shape NxCxM, where M is the number of Monte Carlo samples

#     Returns:
#         pe score array of shape N
#     """
#     mean_softmax_distribution = softmax_distribution.mean(dim=2)
#     epsilon = torch.finfo(softmax_distribution.dtype).eps
#     predictive_collision_entropy = -torch.log(
#                 torch.sum( torch.square(mean_softmax_distribution), dim=1) + epsilon
#                 )
#     # epsilon = np.finfo(mean_softmax_distribution.dtype).eps
#     return -predictive_collision_entropy
# predictive_collision_entropy(results_dataset['softmax'])
#%%
# MCD-GE
# @validate_softmax
# def mcd_guessing_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Predictive collision entropy CSF
#     Args:
#         softmax (ArrayType): array-like containing softmaxes of shape NxC

#     Returns:
#         pe score array of shape N
#     """
#     # epsilon = np.finfo(softmax.dtype).eps
#     # epsilon = torch.finfo(softmax.dtype).eps
#     mean_softmax_distribution = softmax_distribution.mean(dim=2)
#     k_guesses = torch.tile(
#         torch.tensor([i+1 for i in range(softmax_distribution.shape[1])]), (softmax_distribution.shape[0],1)
#         )
#     sorted_mean_softmax = torch.sort( mean_softmax_distribution, descending=True, dim=1 ).values # descending order
#     guessing_entropy = torch.sum( k_guesses * sorted_mean_softmax, dim=1 )
#     return -guessing_entropy
#%%
# MCD-EGE
# @validate_softmax
# def mcd_expected_guessing_entropy(softmax_distribution: ArrayType) -> ArrayType:
#     """Predictive collision entropy CSF
#     Args:
#         softmax (ArrayType): array-like containing softmaxes of shape NxC

#     Returns:
#         pe score array of shape N
#     """
#     # epsilon = np.finfo(softmax.dtype).eps
#     # epsilon = torch.finfo(softmax.dtype).eps
#     # mean_softmax_distribution = softmax_distribution.mean(dim=2)
#     k_guesses = torch.tile(
#         torch.tensor([i+1 for i in range(softmax_distribution.shape[1])]), (softmax_distribution.shape[0],1)
#         )
#     sorted_softmax_distribution = torch.sort( softmax_distribution, descending=True, dim=1 ).values # descending order
#     guessing_entropy = torch.sum( k_guesses[:,:,None] * sorted_softmax_distribution, dim=1 )
#     return torch.mean(-guessing_entropy, dim=1)
