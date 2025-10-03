import pandas as pd
import os
from fd_shifts.analysis import metrics
from src import scores_methods
from src import scores_funcs
from src.rc_stats import RiskCoverageStats
from torch.nn import functional as F
import torch
from fd_shifts import logger
from tqdm import tqdm
#%%
# Detect GPU if available
use_cuda = True if torch.cuda.is_available() else False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {DEVICE}...')
#%%
def run_score_methods(cf, module, study_name, model_evaluations, do_enabled:bool, model_opts:str='', temp_scaled:bool=False):
    # Temperature 
    temperature_scale = scores_methods.TemperatureScaling(cf)
    temperature_scale.load_params(filename='Temperature_params'+model_opts)
    #
    labels_train = model_evaluations['train']['labels']
    labels_val = model_evaluations['val']['labels']
    # Train evaluations
    encoded_train = model_evaluations['train']['encoded']
    logits_train = model_evaluations['train']['logits']
    encoded_val = model_evaluations['val']['encoded']
    correct_val = model_evaluations['val']['correct']
    # Neural Collapse Metrics
    neural_collapse = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
    neural_collapse.compute_NeuralCollapse_params(encoded_train, labels_train)
    neural_collapse.save_params(filename='NeuralCollapse_params'+model_opts)
    # Global
    # Kernel PCA Global
    kpca_global = scores_methods.KernelPCA(module, study_name, cf, mode='global')
    kpca_global.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, 
                                        labels_train=labels_train, only_correct=True, 
                                        temperature=temperature_scale.temperature, 
                                        center_on='all', kernel='rbf')
    kpca_global.save_params(filename='KernelPCA_global_params'+model_opts)
    # Projection Filtering Global
    projection_filtering_global = scores_methods.ProjectionFiltering(module, study_name, cf, mode='global')
    projection_filtering_global.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, labels_train=labels_train, only_correct=True)
    projection_filtering_global.save_params(filename='ProjectionFiltering_global_params'+model_opts)
    logits_global_train = projection_filtering_global.get_logits(encoded_train)
    # Backprojections for global
    encoded_global_train = projection_filtering_global.get_backprojection(encoded_train)
    encoded_global_val = projection_filtering_global.get_backprojection(encoded_val)
    # Neural Collapse Global Metrics
    neural_collapse_global = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
    neural_collapse_global.compute_NeuralCollapse_params(encoded_global_train, labels_train)
    neural_collapse_global.save_params(filename='NeuralCollapse_global_params'+model_opts)
    # Class Typical Matching Global
    ctm_global = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_global.compute_CTM_params(encoded_global_train, labels_train)
    ctm_global.save_params(filename='CTM_global_params'+model_opts)
    del ctm_global
    # NNGuide PCA global
    nnguide_global = scores_methods.NNGuide(module,study_name,cf)
    nnguide_global.tune_hyperparameters(encoded_global_train, encoded_global_val, 1-correct_val,
                                        labels_train = labels_train, logits_train= logits_global_train,)
    nnguide_global.save_params(filename='NNGuide_global_params'+model_opts)
    del nnguide_global
    # fDBD PCA global
    fDBD_global = scores_methods.fDBD(module,study_name,cf)
    fDBD_global.compute_fDBD_params(encoded_global_train)
    fDBD_global.save_params(filename='fDBD_global_params'+model_opts)
    del fDBD_global
    # Mahalanobis distance global
    maha_distance_global = scores_methods.MahalanobisDistance(cf) 
    maha_distance_global.compute_MahaDist_params(encoded_global_train, labels_train)
    maha_distance_global.save_params(filename='MahalanobisDistance_global_params'+model_opts)
    del maha_distance_global
    # pNML global
    pnml_global = scores_methods.pNML(module,study_name,cf)
    pnml_global.compute_pNML_params(encoded_global_train)
    pnml_global.save_params(filename='pNML_global_params'+model_opts)
    del pnml_global
    del encoded_global_train
    #
    logits_global_val = projection_filtering_global.get_logits(encoded_val)
    del projection_filtering_global
    # Temperature Global
    temperature_global = scores_methods.TemperatureScaling(cf)
    temperature_global.compute_temperature(logits_global_val, labels_val)
    temperature_global.save_params(filename='Temperature_global_params'+model_opts)
    # Softmax global
    softmax_global_val = temperature_global.get_scaled_softmax(logits_global_val) if temp_scaled else F.softmax(logits_global_val, dim=1, dtype=torch.float64)
    del logits_global_val
    del temperature_global
    # Generalized entropy Global
    generalized_entropy_global = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_global.compute_entropy_params(softmax_global_val, 1-correct_val)
    generalized_entropy_global.save_params(filename='GEN_global_params'+model_opts)
    del generalized_entropy_global
    # Renyi entropy Global
    renyi_entropy_global = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_global.compute_entropy_params(softmax_global_val, 1-correct_val)
    renyi_entropy_global.save_params(filename='REN_global_params'+model_opts)
    del renyi_entropy_global
    del softmax_global_val
    # Kernel PCA Class
    kpca_class = scores_methods.KernelPCA(module, study_name, cf, mode='class')
    kpca_class.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, 
                                        labels_train=labels_train, only_correct=True, 
                                        temperature=temperature_scale.temperature, 
                                        center_on='all', kernel='rbf')
    kpca_class.save_params(filename='KernelPCA_class_params'+model_opts)
    # Projection Filtering Class
    projection_filtering_class = scores_methods.ProjectionFiltering(module, study_name, cf, mode='class')
    projection_filtering_class.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val, labels_train=labels_train, only_correct=True)
    projection_filtering_class.save_params(filename='ProjectionFiltering_class_params'+model_opts)
    logits_class_train = projection_filtering_class.get_logits(encoded_train)
    # Backprojections for class
    encoded_class_train = projection_filtering_class.get_backprojection(encoded_train)
    encoded_class_val = projection_filtering_class.get_backprojection(encoded_val)
    
    # Backprojections for class w/predictions
    softmax_train = model_evaluations['train']['softmax_scaled'] if temp_scaled else model_evaluations['train']['softmax']
    preds_train = softmax_train.max(dim=1).indices
    softmax_val = model_evaluations['val']['softmax_scaled'] if temp_scaled else model_evaluations['val']['softmax']
    preds_val = softmax_val.max(dim=1).indices 
    del softmax_train
    # logger.info(f'Arranging activations after class ...')
    # encoded_class_pred_train = torch.vstack([encoded_class_train[preds_train[t]][t] for t in range(encoded_train.shape[0])])
    encoded_class_pred_train, logits_class_pred_train = projection_filtering_class.get_combined_backprojection(encoded_class_train, combine='prediction', preds=preds_train)
    # encoded_class_pred_val = torch.vstack([encoded_class_val[preds_val[t]][t] for t in range(encoded_val.shape[0])])
    encoded_class_pred_val, logits_class_pred_val = projection_filtering_class.get_combined_backprojection(encoded_class_val, combine='prediction', preds=preds_val)
    # Neural Collapse Global Metrics
    neural_collapse_class_pred = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
    neural_collapse_class_pred.compute_NeuralCollapse_params(encoded_class_pred_train, labels_train)
    neural_collapse_class_pred.save_params(filename='NeuralCollapse_class_pred_params'+model_opts)
    # Class Typical Matching Class
    ctm_class = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='class')
    ctm_class.compute_CTM_params(encoded_class_train, labels_train)
    ctm_class.save_params(filename='CTM_class_params'+model_opts)
    del ctm_class
    #
    # Temperature Class Pred
    temperature_class_pred = scores_methods.TemperatureScaling(cf)
    temperature_class_pred.compute_temperature(logits_class_pred_val, labels_val)
    temperature_class_pred.save_params(filename='Temperature_class_pred_params'+model_opts)
    # Softmax from filtered logits
    softmax_class_pred_val = temperature_class_pred.get_scaled_softmax(logits_class_pred_val) if temp_scaled else F.softmax(logits_class_pred_val, dim=1, dtype=torch.float64)
    del logits_class_pred_val
    del temperature_class_pred
    # Generalized entropy Class Pred
    generalized_entropy_class_pred = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class_pred.compute_entropy_params(softmax_class_pred_val, 1-correct_val)
    generalized_entropy_class_pred.save_params(filename='GEN_class_pred_params'+model_opts)
    del generalized_entropy_class_pred
    # Renyi entropy Class Pred
    renyi_entropy_class_pred = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class_pred.compute_entropy_params(softmax_class_pred_val, 1-correct_val)
    renyi_entropy_class_pred.save_params(filename='REN_class_pred_params'+model_opts)
    del renyi_entropy_class_pred
    del softmax_class_pred_val
    #
    # Class Typical Matching Class w/predictions
    ctm_class_pred = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_class_pred.compute_CTM_params(encoded_class_pred_train, labels_train)
    ctm_class_pred.save_params(filename='CTM_class_pred_params'+model_opts)
    del ctm_class_pred
    # Backprojections for class averaged
    # encoded_class_avg_train = []
    # for t in range(encoded_train.shape[0]):
    #     avg_sampled = []
    #     for c in range(cf.data.num_classes):
    #         avg_sampled.append(encoded_class_train[c][t])
    #     encoded_class_avg_train.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
    # encoded_class_avg_train = torch.stack(encoded_class_avg_train, dim=0)
    encoded_class_avg_train, logits_class_avg_train = projection_filtering_class.get_combined_backprojection(encoded_class_train, combine='average')
    # Backprojections for class averaged
    # encoded_class_avg_val = []
    # for t in range(encoded_val.shape[0]):
    #     avg_sampled = []
    #     for c in range(cf.data.num_classes):
    #         avg_sampled.append(encoded_class_val[c][t])
    #     encoded_class_avg_val.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
    # encoded_class_avg_val = torch.stack(encoded_class_avg_val, dim=0)
    encoded_class_avg_val, logits_class_avg_val = projection_filtering_class.get_combined_backprojection(encoded_class_val, combine='average')
    del encoded_class_train
    # Neural Collapse Global Metrics
    neural_collapse_class_avg = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
    neural_collapse_class_avg.compute_NeuralCollapse_params(encoded_class_avg_train, labels_train)
    neural_collapse_class_avg.save_params(filename='NeuralCollapse_class_avg_params'+model_opts)
    #
    #
    # NNGuide PCA class w/predictions
    nnguide_class_pred = scores_methods.NNGuide(module,study_name,cf)
    nnguide_class_pred.tune_hyperparameters(encoded_class_pred_train, encoded_class_pred_val, 1-correct_val,
                                        labels_train = labels_train, logits_train= logits_class_train,)
    nnguide_class_pred.save_params(filename='NNGuide_class_pred_params'+model_opts)
    del nnguide_class_pred
    # fDBD PCA class w/predictions
    fDBD_class_pred = scores_methods.fDBD(module,study_name,cf)
    fDBD_class_pred.compute_fDBD_params(encoded_class_pred_train)
    fDBD_class_pred.save_params(filename='fDBD_class_pred_params'+model_opts)
    del fDBD_class_pred
    # Mahalanobis distance class w/predictions
    maha_distance_class_pred = scores_methods.MahalanobisDistance(cf) 
    maha_distance_class_pred.compute_MahaDist_params(encoded_class_pred_train, labels_train)
    maha_distance_class_pred.save_params(filename='MahalanobisDistance_class_pred_params'+model_opts)
    del maha_distance_class_pred
    # pNML class w/predictions
    pnml_class_pred = scores_methods.pNML(module,study_name,cf)
    pnml_class_pred.compute_pNML_params(encoded_class_pred_train)
    pnml_class_pred.save_params(filename='pNML_class_pred_params'+model_opts)
    del pnml_class_pred
    del encoded_class_pred_train
    # Validation set logits from backprojections
    logits_class_val = projection_filtering_class.get_logits(encoded_val)
    del projection_filtering_class
    # Temperature Class
    temperature_class = scores_methods.TemperatureScaling(cf)
    temperature_class.compute_temperature(logits_class_val, labels_val)
    temperature_class.save_params(filename='Temperature_class_params'+model_opts)
    # Softmax from filtered logits
    softmax_class_val = temperature_class.get_scaled_softmax(logits_class_val) if temp_scaled else F.softmax(logits_class_val, dim=1, dtype=torch.float64)
    del logits_class_val
    del temperature_class
    # Generalized entropy Class 
    generalized_entropy_class = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class.compute_entropy_params(softmax_class_val, 1-correct_val)
    generalized_entropy_class.save_params(filename='GEN_class_params'+model_opts)
    del generalized_entropy_class
    # Renyi entropy Class
    renyi_entropy_class = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class.compute_entropy_params(softmax_class_val, 1-correct_val)
    renyi_entropy_class.save_params(filename='REN_class_params'+model_opts)
    del renyi_entropy_class
    del softmax_class_val
    #
    # Temperature Class Avg
    temperature_class_avg = scores_methods.TemperatureScaling(cf)
    temperature_class_avg.compute_temperature(logits_class_avg_val, labels_val)
    temperature_class_avg.save_params(filename='Temperature_class_avg_params'+model_opts)
    # Softmax from filtered logits
    softmax_class_avg_val = temperature_class_avg.get_scaled_softmax(logits_class_avg_val) if temp_scaled else F.softmax(logits_class_avg_val, dim=1, dtype=torch.float64)
    del logits_class_avg_val
    del temperature_class_avg
    # Generalized entropy Class Pred
    generalized_entropy_class_avg = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class_avg.compute_entropy_params(softmax_class_avg_val, 1-correct_val)
    generalized_entropy_class_avg.save_params(filename='GEN_class_avg_params'+model_opts)
    del generalized_entropy_class_avg
    # Renyi entropy Class Pred
    renyi_entropy_class_avg = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class_avg.compute_entropy_params(softmax_class_avg_val, 1-correct_val)
    renyi_entropy_class_avg.save_params(filename='REN_class_avg_params'+model_opts)
    del renyi_entropy_class_avg
    del softmax_class_avg_val
    #
    # Class Typical Matching Class averaged
    ctm_class_avg = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_class_avg.compute_CTM_params(encoded_class_avg_train, labels_train)
    ctm_class_avg.save_params(filename='CTM_class_avg_params'+model_opts)
    del ctm_class_avg
    # NNGuide PCA class averaged
    nnguide_class_avg = scores_methods.NNGuide(module,study_name,cf)
    nnguide_class_avg.tune_hyperparameters(encoded_class_avg_train, encoded_class_avg_val, 1-correct_val,
                                        labels_train = labels_train, logits_train= logits_class_train,)
    nnguide_class_avg.save_params(filename='NNGuide_class_avg_params'+model_opts)
    del nnguide_class_avg
    # fDBD PCA class averaged
    fDBD_class_avg = scores_methods.fDBD(module,study_name,cf)
    fDBD_class_avg.compute_fDBD_params(encoded_class_avg_train)
    fDBD_class_avg.save_params(filename='fDBD_class_avg_params'+model_opts)
    del fDBD_class_avg
    # Mahalanobis distance class averaged
    maha_distance_class_avg = scores_methods.MahalanobisDistance(cf) 
    maha_distance_class_avg.compute_MahaDist_params(encoded_class_avg_train, labels_train)
    maha_distance_class_avg.save_params(filename='MahalanobisDistance_class_avg_params'+model_opts)
    del maha_distance_class_avg
    # pNML class averaged
    pnml_class_avg = scores_methods.pNML(module,study_name,cf)
    pnml_class_avg.compute_pNML_params(encoded_class_avg_train)
    pnml_class_avg.save_params(filename='pNML_class_avg_params'+model_opts)
    del pnml_class_avg
    del encoded_class_avg_train
    del logits_class_train
    # Validation evaluations
    softmax_val = model_evaluations['val']['softmax_scaled'] if temp_scaled else model_evaluations['val']['softmax'] 
    # Class Typical Matching
    ctm = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm.compute_CTM_params(encoded_train, labels_train)
    ctm.save_params(filename='CTM_params'+model_opts)
    del ctm
    # Class Typical Matching for correct only
    ctm_oc = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_oc.compute_CTM_params(encoded_train, labels_train, only_correct=True)
    ctm_oc.save_params(filename='CTM_oc_params'+model_opts)
    del ctm_oc
    # Generalized entropy
    generalized_entropy = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy.compute_entropy_params(softmax_val, 1-correct_val)
    generalized_entropy.save_params(filename='GEN_params'+model_opts)
    del generalized_entropy
    # Renyi entropy
    renyi_entropy = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy.compute_entropy_params(softmax_val, 1-correct_val)
    renyi_entropy.save_params(filename='REN_params'+model_opts)
    del renyi_entropy
    # NNGuide
    nnguide = scores_methods.NNGuide(module,study_name,cf)
    nnguide.tune_hyperparameters(encoded_train, encoded_val, 1-correct_val,
                                        labels_train = labels_train, logits_train= logits_train,)
    nnguide.save_params(filename='NNGuide_params'+model_opts)
    del nnguide
    # fDBD 
    fDBD = scores_methods.fDBD(module,study_name,cf)
    fDBD.compute_fDBD_params(encoded_train)
    fDBD.save_params(filename='fDBD_params'+model_opts)
    del fDBD
    # Mahalanobis distance 
    maha_distance = scores_methods.MahalanobisDistance(cf) 
    maha_distance.compute_MahaDist_params(encoded_train, labels_train)
    maha_distance.save_params(filename='MahalanobisDistance_params'+model_opts)
    del maha_distance
    # pNML 
    pnml = scores_methods.pNML(module,study_name,cf)
    pnml.compute_pNML_params(encoded_train)
    pnml.save_params(filename='pNML_params'+model_opts)
    del pnml
    # ViM Score 
    vim = scores_methods.ViMScore(module,study_name,cf)
    vim.compute_ViM_params(encoded_train)
    vim.save_params(filename='ViM_params'+model_opts)
    del vim
    # Residual Score 
    residual = scores_methods.ResidualScore(module,study_name,cf)
    residual.compute_Residual_params(encoded_train)
    residual.save_params(filename='Residual_params'+model_opts)
    del residual
    # NeCo Score 
    neco = scores_methods.NeCo(module,study_name,cf)
    neco.compute_NeCo_params(encoded_train)
    neco.save_params(filename='NeCo_params'+model_opts)
    del neco
    del encoded_train

    if do_enabled:
        # Temperature distribution 
        temperature_scale_dist = scores_methods.TemperatureScaling(cf)
        temperature_scale_dist.load_params(filename='Temperature_distribution_params'+model_opts)
        #
        encoded_dist_train = model_evaluations['train']['encoded_dist']
        logits_dist_train = model_evaluations['train']['logits_dist']
        encoded_dist_val = model_evaluations['val']['encoded_dist']
        correct_mcd_val = model_evaluations['val']['correct_mcd']
        # Neural Collapse Metrics for Distribution
        neural_collapse_dist = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_dist.compute_NeuralCollapse_params(encoded_dist_train.mean(dim=2), labels_train)
        neural_collapse_dist.save_params(filename='NeuralCollapse_distribution_params'+model_opts)
        # Global
        # Kernel PCA Global
        kpca_global_dist = scores_methods.KernelPCA(module, study_name, cf, mode='global')
        kpca_global_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                            labels_train=labels_train, only_correct=True, 
                                            temperature=temperature_scale_dist.temperature, 
                                            center_on='all', kernel='rbf')
        kpca_global_dist.save_params(filename='KernelPCA_global_distribution_params'+model_opts)
        # Projection Filtering Global for distribution
        projection_filtering_global_dist = scores_methods.ProjectionFiltering(module, study_name, cf, mode='global')
        projection_filtering_global_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                                                labels_train=labels_train, only_correct=True)
        projection_filtering_global_dist.save_params(filename='ProjectionFiltering_global_distribution_params'+model_opts)
        logits_global_dist_train = projection_filtering_global_dist.get_logits(encoded_dist_train.mean(dim=2))
        # Backprojections global for distribution
        encoded_global_dist_train = projection_filtering_global_dist.get_backprojection(encoded_dist_train.mean(dim=2))
        encoded_global_dist_val = projection_filtering_global_dist.get_backprojection(encoded_dist_val.mean(dim=2))
        # Neural Collapse Global Metrics for Distribution
        neural_collapse_global_dist = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_global_dist.compute_NeuralCollapse_params(encoded_global_dist_train, labels_train)
        neural_collapse_global_dist.save_params(filename='NeuralCollapse_global_distribution_params'+model_opts)
        # Class Typical Matching Global for distribution
        ctm_global_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_global_dist.compute_CTM_params(encoded_global_dist_train, labels_train)
        ctm_global_dist.save_params(filename='CTM_global_distribution_params'+model_opts)
        del ctm_global_dist
        # NNGuide PCA global for distribution
        nnguide_global_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_global_dist.tune_hyperparameters(encoded_global_dist_train, encoded_global_dist_val, 1-correct_mcd_val,
                                        labels_train = labels_train, logits_train= logits_global_dist_train,)
        nnguide_global_dist.save_params(filename='NNGuide_global_distribution_params'+model_opts)
        del nnguide_global_dist
        # fDBD PCA global for distribution
        fDBD_global_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_global_dist.compute_fDBD_params(encoded_global_dist_train)
        fDBD_global_dist.save_params(filename='fDBD_global_distribution_params'+model_opts)
        del fDBD_global_dist
        # Mahalanobis distance global for distribution
        maha_distance_global_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_global_dist.compute_MahaDist_params(encoded_global_dist_train, labels_train)
        maha_distance_global_dist.save_params(filename='MahalanobisDistance_global_distribution_params'+model_opts)
        del maha_distance_global_dist
        # pNML global for distribution
        pnml_global_dist = scores_methods.pNML(module,study_name,cf)
        pnml_global_dist.compute_pNML_params(encoded_global_dist_train)
        pnml_global_dist.save_params(filename='pNML_global_distribution_params'+model_opts)
        del pnml_global_dist
        del encoded_global_dist_train
        #
        logits_global_dist_val = projection_filtering_global_dist.get_logits(encoded_dist_val.mean(dim=2))
        del projection_filtering_global_dist
        # Temperature global for distribution
        temperature_global_dist = scores_methods.TemperatureScaling(cf)
        temperature_global_dist.compute_temperature(logits_global_dist_val, labels_val)
        temperature_global_dist.save_params(filename='Temperature_global_distribution_params'+model_opts)
        # Softmax global for distribution
        softmax_global_dist_val = temperature_global_dist.get_scaled_softmax(logits_global_dist_val) if temp_scaled else F.softmax(logits_global_dist_val, dim=1, dtype=torch.float64)
        del temperature_global_dist
        del logits_global_dist_val
        # Generalized entropy Global for distribution
        generalized_entropy_global_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_global_dist.compute_entropy_params(softmax_global_dist_val, 1-correct_mcd_val)
        generalized_entropy_global_dist.save_params(filename='GEN_global_distribution_params'+model_opts)
        del generalized_entropy_global_dist
        # Renyi entropy Global for distribution
        renyi_entropy_global_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_global_dist.compute_entropy_params(softmax_global_dist_val, 1-correct_mcd_val)
        renyi_entropy_global_dist.save_params(filename='REN_global_distribution_params'+model_opts)
        del renyi_entropy_global_dist
        del softmax_global_dist_val
        # Kernel PCA Class
        kpca_class_dist = scores_methods.KernelPCA(module, study_name, cf, mode='class')
        kpca_class_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                            labels_train=labels_train, only_correct=True, 
                                            temperature=temperature_scale_dist.temperature, 
                                            center_on='all', kernel='rbf')
        kpca_class_dist.save_params(filename='KernelPCA_class_distribution_params'+model_opts)
        # Projection Filtering Class for distribution
        projection_filtering_class_dist = scores_methods.ProjectionFiltering(module, study_name, cf, mode='class')
        projection_filtering_class_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val, 
                                                                labels_train=labels_train, only_correct=True)
        projection_filtering_class_dist.save_params(filename='ProjectionFiltering_class_distribution_params'+model_opts)
        logits_class_dist_train = projection_filtering_class_dist.get_logits(encoded_dist_train.mean(dim=2))
        # Backprojections for class for distribution
        encoded_class_dist_train = projection_filtering_class_dist.get_backprojection(encoded_dist_train.mean(dim=2))
        encoded_class_dist_val = projection_filtering_class_dist.get_backprojection(encoded_dist_val.mean(dim=2))
        # Backprojections for class w/predictions for distribution
        softmax_dist_train = model_evaluations['train']['softmax_scaled_dist'] if temp_scaled else model_evaluations['train']['softmax_dist'] 
        preds_dist_train = softmax_dist_train.mean(dim=2).max(dim=1).indices 
        softmax_dist_val = model_evaluations['val']['softmax_scaled_dist'] if temp_scaled else model_evaluations['val']['softmax_dist'] 
        preds_dist_val = softmax_dist_val.mean(dim=2).max(dim=1).indices
        del softmax_dist_train
        # Backprojections for class w/predictions for distribution
        # encoded_class_pred_dist_train = torch.vstack([encoded_class_dist_train[preds_dist_train[t]][t] for t in range(encoded_dist_train.shape[0])])
        encoded_class_pred_dist_train, logits_class_pred_dist_train = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_train, combine='prediction', preds=preds_dist_train)
        # encoded_class_pred_dist_val = torch.vstack([encoded_class_dist_val[preds_dist_val[t]][t] for t in range(encoded_dist_val.shape[0])])
        encoded_class_pred_dist_val, logits_class_pred_dist_val = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_val, combine='prediction', preds=preds_dist_val)
        # Neural Collapse Global Metrics for Distribution
        neural_collapse_class_pred_dist = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_class_pred_dist.compute_NeuralCollapse_params(encoded_class_pred_dist_train, labels_train)
        neural_collapse_class_pred_dist.save_params(filename='NeuralCollapse_class_pred_distribution_params'+model_opts)
        # Class Typical Matching Class for distribution
        ctm_class_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='class')
        ctm_class_dist.compute_CTM_params(encoded_class_dist_train, labels_train)
        ctm_class_dist.save_params(filename='CTM_class_distribution_params'+model_opts)
        del ctm_class_dist
        #
        # Temperature Class for distribution
        temperature_class_pred_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_pred_dist.compute_temperature(logits_class_pred_dist_val, labels_val)
        temperature_class_pred_dist.save_params(filename='Temperature_class_pred_distribution_params'+model_opts)
        # Softmax from filtered logits
        softmax_class_pred_dist_val = temperature_class_pred_dist.get_scaled_softmax(logits_class_pred_dist_val) if temp_scaled else F.softmax(logits_class_pred_dist_val, dim=1, dtype=torch.float64)
        del logits_class_pred_dist_val
        del temperature_class_pred_dist
        # Generalized entropy Class for distribution
        generalized_entropy_class_pred_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_pred_dist.compute_entropy_params(softmax_class_pred_dist_val, 1-correct_mcd_val)
        generalized_entropy_class_pred_dist.save_params(filename='GEN_class_pred_distribution_params'+model_opts)
        del generalized_entropy_class_pred_dist
        # Renyi entropy Class for distribution
        renyi_entropy_class_pred_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_pred_dist.compute_entropy_params(softmax_class_pred_dist_val, 1-correct_mcd_val)
        renyi_entropy_class_pred_dist.save_params(filename='REN_class_pred_distribution_params'+model_opts)
        del renyi_entropy_class_pred_dist
        del softmax_class_pred_dist_val
        #
        # Class Typical Matching Class w/predictions for distribution
        ctm_class_pred_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_pred_dist.compute_CTM_params(encoded_class_pred_dist_train, labels_train)
        ctm_class_pred_dist.save_params(filename='CTM_class_pred_distribution_params'+model_opts)
        del ctm_class_pred_dist
        # Backprojections for class averaged for distribution
        # encoded_class_avg_dist_train = []
        # for t in range(encoded_dist_train.shape[0]):
        #     avg_sampled = []
        #     for c in range(cf.data.num_classes):
        #         avg_sampled.append(encoded_class_dist_train[c][t])
        #     encoded_class_avg_dist_train.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
        # encoded_class_avg_dist_train = torch.stack(encoded_class_avg_dist_train, dim=0)
        encoded_class_avg_dist_train, logits_class_avg_dist_train = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_train, combine='average')
        #
        # encoded_class_avg_dist_val = []
        # for t in range(encoded_dist_val.shape[0]):
        #     avg_sampled = []
        #     for c in range(cf.data.num_classes):
        #         avg_sampled.append(encoded_class_dist_val[c][t])
        #     encoded_class_avg_dist_val.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
        # encoded_class_avg_dist_val = torch.stack(encoded_class_avg_dist_val, dim=0)
        encoded_class_avg_dist_val, logits_class_avg_dist_val = projection_filtering_class_dist.get_combined_backprojection(encoded_class_dist_val, combine='average')
        del encoded_class_dist_train
        # Neural Collapse Global Metrics for Distribution
        neural_collapse_class_avg_dist = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
        neural_collapse_class_avg_dist.compute_NeuralCollapse_params(encoded_class_avg_dist_train, labels_train)
        neural_collapse_class_avg_dist.save_params(filename='NeuralCollapse_class_avg_distribution_params'+model_opts)
        # NNGuide PCA class w/predictions for distribution
        nnguide_class_pred_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_class_pred_dist.tune_hyperparameters(encoded_class_pred_dist_train, encoded_class_pred_dist_val, 1-correct_mcd_val,
                                        labels_train = labels_train, logits_train= logits_class_dist_train,)
        nnguide_class_pred_dist.save_params(filename='NNGuide_class_pred_distribution_params'+model_opts)
        del nnguide_class_pred_dist
        # fDBD PCA class w/predictions for distribution
        fDBD_class_pred_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_class_pred_dist.compute_fDBD_params(encoded_class_pred_dist_train)
        fDBD_class_pred_dist.save_params(filename='fDBD_class_pred_distribution_params'+model_opts)
        del fDBD_class_pred_dist
        # Mahalanobis distance class w/predictions
        maha_distance_class_pred_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_class_pred_dist.compute_MahaDist_params(encoded_class_pred_dist_train, labels_train)
        maha_distance_class_pred_dist.save_params(filename='MahalanobisDistance_class_pred_distribution_params'+model_opts)
        del maha_distance_class_pred_dist
        # pNML class w/predictions
        pnml_class_pred_dist = scores_methods.pNML(module,study_name,cf)
        pnml_class_pred_dist.compute_pNML_params(encoded_class_pred_dist_train)
        pnml_class_pred_dist.save_params(filename='pNML_class_pred_distribution_params'+model_opts)
        del pnml_class_pred_dist
        del encoded_class_pred_dist_train
        # Validation set logits from backprojections 
        logits_class_dist_val = projection_filtering_class_dist.get_logits(encoded_dist_val.mean(dim=2))
        # del encoded_dist_val
        del projection_filtering_class_dist
        # Temperature Class for distribution
        temperature_class_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_dist.compute_temperature(logits_class_dist_val, labels_val)
        temperature_class_dist.save_params(filename='Temperature_class_distribution_params'+model_opts)
        # Softmax from filtered logits for distribution
        softmax_class_dist_val = temperature_class_dist.get_scaled_softmax(logits_class_dist_val) if temp_scaled else F.softmax(logits_class_dist_val, dim=1, dtype=torch.float64)
        del temperature_class_dist
        del logits_class_dist_val
        # Generalized entropy Class for distribution
        generalized_entropy_class_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_dist.compute_entropy_params(softmax_class_dist_val, 1-correct_mcd_val)
        generalized_entropy_class_dist.save_params(filename='GEN_class_distribution_params'+model_opts)
        del generalized_entropy_class_dist
        # Renyi entropy Class for distribution
        renyi_entropy_class_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_dist.compute_entropy_params(softmax_class_dist_val, 1-correct_mcd_val)
        renyi_entropy_class_dist.save_params(filename='REN_class_distribution_params'+model_opts)
        del renyi_entropy_class_dist
        del softmax_class_dist_val
        #
        # Temperature Class for distribution
        temperature_class_avg_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_avg_dist.compute_temperature(logits_class_avg_dist_val, labels_val)
        temperature_class_avg_dist.save_params(filename='Temperature_class_avg_distribution_params'+model_opts)
        # Softmax from filtered logits
        softmax_class_avg_dist_val = temperature_class_avg_dist.get_scaled_softmax(logits_class_avg_dist_val) if temp_scaled else F.softmax(logits_class_avg_dist_val, dim=1, dtype=torch.float64)
        del logits_class_avg_dist_val
        del temperature_class_avg_dist
        # Generalized entropy Class for distribution
        generalized_entropy_class_avg_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_avg_dist.compute_entropy_params(softmax_class_avg_dist_val, 1-correct_mcd_val)
        generalized_entropy_class_avg_dist.save_params(filename='GEN_class_avg_distribution_params'+model_opts)
        del generalized_entropy_class_avg_dist
        # Renyi entropy Class for distribution
        renyi_entropy_class_avg_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_avg_dist.compute_entropy_params(softmax_class_avg_dist_val, 1-correct_mcd_val)
        renyi_entropy_class_avg_dist.save_params(filename='REN_class_avg_distribution_params'+model_opts)
        del renyi_entropy_class_avg_dist
        del softmax_class_avg_dist_val
        #
        # Class Typical Matching Class averaged for distribution
        ctm_class_avg_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_avg_dist.compute_CTM_params(encoded_class_avg_dist_train, labels_train)
        ctm_class_avg_dist.save_params(filename='CTM_class_avg_distribution_params'+model_opts)
        del ctm_class_avg_dist
        # NNGuide PCA class averaged for distribution
        nnguide_class_avg_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_class_avg_dist.tune_hyperparameters(encoded_class_avg_dist_train, encoded_class_avg_dist_val, 1-correct_mcd_val,
                                        labels_train = labels_train, logits_train= logits_class_dist_train,)
        nnguide_class_avg_dist.save_params(filename='NNGuide_class_avg_distribution_params'+model_opts)
        del nnguide_class_avg_dist
        # fDBD PCA class averaged for distribution
        fDBD_class_avg_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_class_avg_dist.compute_fDBD_params(encoded_class_avg_dist_train)
        fDBD_class_avg_dist.save_params(filename='fDBD_class_avg_distribution_params'+model_opts)
        del fDBD_class_avg_dist
        # Mahalanobis distance class averaged for distribution
        maha_distance_class_avg_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_class_avg_dist.compute_MahaDist_params(encoded_class_avg_dist_train, labels_train)
        maha_distance_class_avg_dist.save_params(filename='MahalanobisDistance_class_avg_distribution_params'+model_opts)
        del maha_distance_class_avg_dist
        # pNML class averaged for distribution
        pnml_class_avg_dist = scores_methods.pNML(module,study_name,cf)
        pnml_class_avg_dist.compute_pNML_params(encoded_class_avg_dist_train)
        pnml_class_avg_dist.save_params(filename='pNML_class_avg_distribution_params'+model_opts)
        del pnml_class_avg_dist
        del encoded_class_avg_dist_train
        del logits_class_dist_train
        # Validation evaluations for distribution
        softmax_dist_val = model_evaluations['val']['softmax_scaled_dist'] if temp_scaled else model_evaluations['val']['softmax_dist']
        # Class Typical Matching
        ctm_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_dist.compute_CTM_params(encoded_dist_train.mean(dim=2), labels_train)
        ctm_dist.save_params(filename='CTM_distribution_params'+model_opts)
        del ctm_dist
        # Class Typical Matching for only correct predictions
        ctm_oc_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_oc_dist.compute_CTM_params(encoded_dist_train.mean(dim=2), labels_train, only_correct=True)
        ctm_oc_dist.save_params(filename='CTM_oc_distribution_params'+model_opts)
        del ctm_oc_dist
        # Generalized entropy for distribution
        generalized_entropy_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_dist.compute_entropy_params(softmax_dist_val.mean(dim=2), 1-correct_mcd_val)
        generalized_entropy_dist.save_params(filename='GEN_distribution_params'+model_opts)
        del generalized_entropy_dist
        # Renyi entropy for distribution
        renyi_entropy_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_dist.compute_entropy_params(softmax_dist_val.mean(dim=2), 1-correct_mcd_val)
        renyi_entropy_dist.save_params(filename='REN_distribution_params'+model_opts)
        del renyi_entropy_dist
        # del correct_mcd_val
        # NNGuide for distribution
        nnguide_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_dist.tune_hyperparameters(encoded_dist_train.mean(dim=2), encoded_dist_val.mean(dim=2), 1-correct_mcd_val,
                                        labels_train = labels_train, logits_train= logits_dist_train.mean(dim=2),)
        nnguide_dist.save_params(filename='NNGuide_distribution_params'+model_opts)
        del nnguide_dist
        # fDBD for distribution
        fDBD_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_dist.compute_fDBD_params(encoded_dist_train.mean(dim=2))
        fDBD_dist.save_params(filename='fDBD_distribution_params'+model_opts)
        del fDBD_dist
        # Mahalanobis distance for distribution
        maha_distance_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_dist.compute_MahaDist_params(encoded_dist_train.mean(dim=2), labels_train)
        maha_distance_dist.save_params(filename='MahalanobisDistance_distribution_params'+model_opts)
        del maha_distance_dist
        # pNML for distribution
        pnml_dist = scores_methods.pNML(module,study_name,cf)
        pnml_dist.compute_pNML_params(encoded_dist_train.mean(dim=2))
        pnml_dist.save_params(filename='pNML_distribution_params'+model_opts)
        del pnml_dist
        # ViM Score for distribution
        vim_dist = scores_methods.ViMScore(module,study_name,cf)
        vim_dist.compute_ViM_params(encoded_dist_train.mean(dim=2))
        vim_dist.save_params(filename='ViM_distribution_params'+model_opts)
        del vim_dist
        # Residual Score for distribution
        residual_dist = scores_methods.ResidualScore(module,study_name,cf)
        residual_dist.compute_Residual_params(encoded_dist_train.mean(dim=2))
        residual_dist.save_params(filename='Residual_distribution_params'+model_opts)
        del residual_dist
        # NeCo Score for distribution
        neco_dist = scores_methods.NeCo(module,study_name,cf)
        neco_dist.compute_NeCo_params(encoded_dist_train.mean(dim=2))
        neco_dist.save_params(filename='NeCo_distribution_params'+model_opts)
        del neco_dist
        del encoded_dist_train
        
        
#%%
# Load parameters
def load_score_methods(cf, module, study_name, do_enabled:bool, model_opts:str=''):
    # Temperature 
    temperature_scale = scores_methods.TemperatureScaling(cf)
    temperature_scale.load_params(filename='Temperature_params'+model_opts)
    # Global
    # Kernel PCA Global
    kpca_global = scores_methods.KernelPCA(module, study_name, cf, mode='global')
    kpca_global.load_params(filename='KernelPCA_global_params'+model_opts)
    # Projection Filtering Global
    projection_filtering_global = scores_methods.ProjectionFiltering(module, study_name, cf, mode='global')
    projection_filtering_global.load_params(filename='ProjectionFiltering_global_params'+model_opts)
    # Class Typical Matching Global
    ctm_global = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_global.load_params(filename='CTM_global_params'+model_opts)
    # NNGuide PCA global
    nnguide_global = scores_methods.NNGuide(module,study_name,cf)
    nnguide_global.load_params(filename='NNGuide_global_params'+model_opts)
    # fDBD PCA global
    fDBD_global = scores_methods.fDBD(module,study_name,cf)
    fDBD_global.load_params(filename='fDBD_global_params'+model_opts)
    # Mahalanobis distance global
    maha_distance_global = scores_methods.MahalanobisDistance(cf) 
    maha_distance_global.load_params(filename='MahalanobisDistance_global_params'+model_opts)
    # pNML global
    pnml_global = scores_methods.pNML(module,study_name,cf)
    pnml_global.load_params(filename='pNML_global_params'+model_opts)
    # Temperature Global
    temperature_global = scores_methods.TemperatureScaling(cf)
    temperature_global.load_params(filename='Temperature_global_params'+model_opts)
    # Generalized entropy Global
    generalized_entropy_global = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_global.load_params(filename='GEN_global_params'+model_opts)
    # Renyi entropy Global
    renyi_entropy_global = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_global.load_params(filename='REN_global_params'+model_opts)
    # Kernel PCA Class
    kpca_class = scores_methods.KernelPCA(module, study_name, cf, mode='class')
    kpca_class.load_params(filename='KernelPCA_class_params'+model_opts)
    # Projection Filtering Class
    projection_filtering_class = scores_methods.ProjectionFiltering(module, study_name, cf, mode='class')
    projection_filtering_class.load_params(filename='ProjectionFiltering_class_params'+model_opts)
    # Class Typical Matching Class
    ctm_class = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='class')
    ctm_class.load_params(filename='CTM_class_params'+model_opts)
    #
    # Temperature Class Pred
    temperature_class_pred = scores_methods.TemperatureScaling(cf)
    temperature_class_pred.load_params(filename='Temperature_class_pred_params'+model_opts)
    # Generalized entropy Class Pred
    generalized_entropy_class_pred = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class_pred.load_params(filename='GEN_class_pred_params'+model_opts)
    # Renyi entropy Class Pred
    renyi_entropy_class_pred = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class_pred.load_params(filename='REN_class_pred_params'+model_opts)
    #
    # Class Typical Matching Class w/predictions
    ctm_class_pred = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_class_pred.load_params(filename='CTM_class_pred_params'+model_opts)
    # NNGuide PCA class w/predictions
    nnguide_class_pred = scores_methods.NNGuide(module,study_name,cf)
    nnguide_class_pred.load_params(filename='NNGuide_class_pred_params'+model_opts)
    # fDBD PCA class w/predictions
    fDBD_class_pred = scores_methods.fDBD(module,study_name,cf)
    fDBD_class_pred.load_params(filename='fDBD_class_pred_params'+model_opts)
    # Mahalanobis distance class w/predictions
    maha_distance_class_pred = scores_methods.MahalanobisDistance(cf) 
    maha_distance_class_pred.load_params(filename='MahalanobisDistance_class_pred_params'+model_opts)
    # pNML class w/predictions
    pnml_class_pred = scores_methods.pNML(module,study_name,cf)
    pnml_class_pred.load_params(filename='pNML_class_pred_params'+model_opts)
    # Temperature Class
    temperature_class = scores_methods.TemperatureScaling(cf)
    temperature_class.load_params(filename='Temperature_class_params'+model_opts)
    # Generalized entropy Class
    generalized_entropy_class = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class.load_params(filename='GEN_class_params'+model_opts)
    # Renyi entropy Class
    renyi_entropy_class = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class.load_params(filename='REN_class_params'+model_opts)
    # Temperature Class Avg
    temperature_class_avg = scores_methods.TemperatureScaling(cf)
    temperature_class_avg.load_params(filename='Temperature_class_avg_params'+model_opts)
    # Generalized entropy Class Pred
    generalized_entropy_class_avg = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy_class_avg.load_params(filename='GEN_class_avg_params'+model_opts)
    # Renyi entropy Class Pred
    renyi_entropy_class_avg = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy_class_avg.load_params(filename='REN_class_avg_params'+model_opts)

    # Class Typical Matching Class averaged
    ctm_class_avg = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_class_avg.load_params(filename='CTM_class_avg_params'+model_opts)
    # NNGuide PCA class averaged
    nnguide_class_avg = scores_methods.NNGuide(module,study_name,cf)
    nnguide_class_avg.load_params(filename='NNGuide_class_avg_params'+model_opts)
    # fDBD PCA class averaged
    fDBD_class_avg = scores_methods.fDBD(module,study_name,cf)
    fDBD_class_avg.load_params(filename='fDBD_class_avg_params'+model_opts)
    # Mahalanobis distance class averaged
    maha_distance_class_avg = scores_methods.MahalanobisDistance(cf) 
    maha_distance_class_avg.load_params(filename='MahalanobisDistance_class_avg_params'+model_opts)
    # pNML class averaged
    pnml_class_avg = scores_methods.pNML(module,study_name,cf)
    pnml_class_avg.load_params(filename='pNML_class_avg_params'+model_opts)
    # Class Typical Matching
    ctm = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm.load_params(filename='CTM_params'+model_opts)
    # Class Typical Matching for correct only
    ctm_oc = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
    ctm_oc.load_params(filename='CTM_oc_params'+model_opts)
    # Generalized entropy
    generalized_entropy = scores_methods.EntropyScores(cf, 'generalized')
    generalized_entropy.load_params(filename='GEN_params'+model_opts)
    # Renyi entropy
    renyi_entropy = scores_methods.EntropyScores(cf, 'renyi')
    renyi_entropy.load_params(filename='REN_params'+model_opts)
    # NNGuide
    nnguide = scores_methods.NNGuide(module,study_name,cf)
    nnguide.load_params(filename='NNGuide_params'+model_opts)
    # fDBD 
    fDBD = scores_methods.fDBD(module,study_name,cf)
    fDBD.load_params(filename='fDBD_params'+model_opts)
    # Mahalanobis distance 
    maha_distance = scores_methods.MahalanobisDistance(cf) 
    maha_distance.load_params(filename='MahalanobisDistance_params'+model_opts)
    # pNML 
    pnml = scores_methods.pNML(module,study_name,cf)
    pnml.load_params(filename='pNML_params'+model_opts)
    # ViM Score 
    vim = scores_methods.ViMScore(module,study_name,cf)
    vim.load_params(filename='ViM_params'+model_opts)
    # Residual Score 
    residual = scores_methods.ResidualScore(module,study_name,cf)
    residual.load_params(filename='Residual_params'+model_opts)
    # NeCo Score 
    neco = scores_methods.NeCo(module,study_name,cf)
    neco.load_params(filename='NeCo_params'+model_opts)

    funcs = {
            'temperature_scale':temperature_scale,
            'kpca_global':kpca_global,
            'projection_filtering_global':projection_filtering_global,
            'ctm_global':ctm_global,
            'nnguide_global':nnguide_global,
            'fDBD_global':fDBD_global,
            'maha_distance_global':maha_distance_global,
            'pnml_global':pnml_global,
            'temperature_global':temperature_global,
            'generalized_entropy_global':generalized_entropy_global,
            'renyi_entropy_global':renyi_entropy_global,
            'kpca_class':kpca_class,
            'projection_filtering_class':projection_filtering_class,
            'ctm_class':ctm_class,
            'temperature_class_pred':temperature_class_pred,
            'generalized_entropy_class_pred':generalized_entropy_class_pred,
            'renyi_entropy_class_pred':renyi_entropy_class_pred,
            'ctm_class_pred':ctm_class_pred,
            'nnguide_class_pred':nnguide_class_pred,
            'fDBD_class_pred':fDBD_class_pred,
            'maha_distance_class_pred':maha_distance_class_pred,
            'pnml_class_pred':pnml_class_pred,
            'temperature_class':temperature_class,
            'generalized_entropy_class':generalized_entropy_class,
            'renyi_entropy_class':renyi_entropy_class,
            'temperature_class_avg':temperature_class_avg,
            'generalized_entropy_class_avg':generalized_entropy_class_avg,
            'renyi_entropy_class_avg':renyi_entropy_class_avg,
            'ctm_class_avg':ctm_class_avg,
            'nnguide_class_avg':nnguide_class_avg,
            'fDBD_class_avg':fDBD_class_avg,
            'maha_distance_class_avg':maha_distance_class_avg,
            'pnml_class_avg':pnml_class_avg,
            'ctm':ctm,
            'ctm_oc':ctm_oc,
            'generalized_entropy':generalized_entropy,
            'renyi_entropy':renyi_entropy,
            'nnguide':nnguide,
            'fDBD':fDBD,
            'maha_distance':maha_distance,
            'pnml':pnml,
            'vim':vim,
            'residual':residual,
            'neco':neco,
            }   

    if do_enabled:
        # Temperature for distribution
        temperature_scale_dist = scores_methods.TemperatureScaling(cf)
        temperature_scale_dist.load_params(filename='Temperature_distribution_params'+model_opts)
        # Kernel PCA Global
        kpca_global_dist = scores_methods.KernelPCA(module, study_name, cf, mode='global')
        kpca_global_dist.load_params(filename='KernelPCA_global_distribution_params'+model_opts)
        # Projection Filtering Global
        projection_filtering_global_dist = scores_methods.ProjectionFiltering(module, study_name, cf, mode='global')
        projection_filtering_global_dist.load_params(filename='ProjectionFiltering_global_distribution_params'+model_opts)
        # Class Typical Matching Global
        ctm_global_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_global_dist.load_params(filename='CTM_global_distribution_params'+model_opts)
        # NNGuide PCA global
        nnguide_global_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_global_dist.load_params(filename='NNGuide_global_distribution_params'+model_opts)
        # fDBD PCA global
        fDBD_global_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_global_dist.load_params(filename='fDBD_global_distribution_params'+model_opts)
        # Mahalanobis distance global for distribution
        maha_distance_global_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_global_dist.load_params(filename='MahalanobisDistance_global_distribution_params'+model_opts)
        # pNML global for distribution
        pnml_global_dist = scores_methods.pNML(module,study_name,cf)
        pnml_global_dist.load_params(filename='pNML_global_distribution_params'+model_opts)
        # Temperature global
        temperature_global_dist = scores_methods.TemperatureScaling(cf)
        temperature_global_dist.load_params(filename='Temperature_global_distribution_params'+model_opts)
        # Generalized entropy Global for distribution
        generalized_entropy_global_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_global_dist.load_params(filename='GEN_global_distribution_params'+model_opts)
        # Renyi entropy Global for distribution
        renyi_entropy_global_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_global_dist.load_params(filename='REN_global_distribution_params'+model_opts)
        # Kernel PCA Class
        kpca_class_dist = scores_methods.KernelPCA(module, study_name, cf, mode='class')
        kpca_class_dist.load_params(filename='KernelPCA_class_distribution_params'+model_opts)
        # Projection Filtering Class for distribution
        projection_filtering_class_dist = scores_methods.ProjectionFiltering(module, study_name, cf, mode='class')
        projection_filtering_class_dist.load_params(filename='ProjectionFiltering_class_distribution_params'+model_opts)
        # Class Typical Matching Class for distribution
        ctm_class_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='class')
        ctm_class_dist.load_params(filename='CTM_class_distribution_params'+model_opts)
        #
        # Temperature Class for distribution
        temperature_class_pred_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_pred_dist.load_params(filename='Temperature_class_pred_distribution_params'+model_opts)
        # Generalized entropy Class for distribution
        generalized_entropy_class_pred_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_pred_dist.load_params(filename='GEN_class_pred_distribution_params'+model_opts)
        # Renyi entropy Class for distribution
        renyi_entropy_class_pred_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_pred_dist.load_params(filename='REN_class_pred_distribution_params'+model_opts)
        #
        # Class Typical Matching Class w/predictions for distribution
        ctm_class_pred_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_pred_dist.load_params(filename='CTM_class_pred_distribution_params'+model_opts)
        # NNGuide PCA class w/predictions for distribution
        nnguide_class_pred_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_class_pred_dist.load_params(filename='NNGuide_class_pred_distribution_params'+model_opts)
        # fDBD PCA class w/predictions for distribution
        fDBD_class_pred_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_class_pred_dist.load_params(filename='fDBD_class_pred_distribution_params'+model_opts)
        # Mahalanobis distance class w/predictions
        maha_distance_class_pred_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_class_pred_dist.load_params(filename='MahalanobisDistance_class_pred_distribution_params'+model_opts)
        # pNML class w/predictions
        pnml_class_pred_dist = scores_methods.pNML(module,study_name,cf)
        pnml_class_pred_dist.load_params(filename='pNML_class_pred_distribution_params'+model_opts)
        # Temperature Class for distribution
        temperature_class_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_dist.load_params(filename='Temperature_class_distribution_params'+model_opts)
        # Generalized entropy Class for distribution
        generalized_entropy_class_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_dist.load_params(filename='GEN_class_distribution_params'+model_opts)
        # Renyi entropy Class for distribution
        renyi_entropy_class_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_dist.load_params(filename='REN_class_distribution_params'+model_opts)
        #
        # Temperature Class for distribution
        temperature_class_avg_dist = scores_methods.TemperatureScaling(cf)
        temperature_class_avg_dist.load_params(filename='Temperature_class_avg_distribution_params'+model_opts)
        # Generalized entropy Class for distribution
        generalized_entropy_class_avg_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_class_avg_dist.load_params(filename='GEN_class_avg_distribution_params'+model_opts)
        # Renyi entropy Class for distribution
        renyi_entropy_class_avg_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_class_avg_dist.load_params(filename='REN_class_avg_distribution_params'+model_opts)
        #
        # Class Typical Matching Class averaged for distribution
        ctm_class_avg_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_class_avg_dist.load_params(filename='CTM_class_avg_distribution_params'+model_opts)
        # NNGuide PCA class averaged for distribution
        nnguide_class_avg_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_class_avg_dist.load_params(filename='NNGuide_class_avg_distribution_params'+model_opts)
        # fDBD PCA class averaged for distribution
        fDBD_class_avg_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_class_avg_dist.load_params(filename='fDBD_class_avg_distribution_params'+model_opts)
        # Mahalanobis distance class averaged for distribution
        maha_distance_class_avg_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_class_avg_dist.load_params(filename='MahalanobisDistance_class_avg_distribution_params'+model_opts)
        # pNML class averaged for distribution
        pnml_class_avg_dist = scores_methods.pNML(module,study_name,cf)
        pnml_class_avg_dist.load_params(filename='pNML_class_avg_distribution_params'+model_opts)
        # Class Typical Matching
        ctm_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_dist.load_params(filename='CTM_distribution_params'+model_opts)
        # Class Typical Matching for only correct predictions
        ctm_oc_dist = scores_methods.ClassTypicalMatching(module, study_name, cf, mode='global')
        ctm_oc_dist.load_params(filename='CTM_oc_distribution_params'+model_opts)
        # Generalized entropy for distribution
        generalized_entropy_dist = scores_methods.EntropyScores(cf, 'generalized')
        generalized_entropy_dist.load_params(filename='GEN_distribution_params'+model_opts)
        # Renyi entropy for distribution
        renyi_entropy_dist = scores_methods.EntropyScores(cf, 'renyi')
        renyi_entropy_dist.load_params(filename='REN_distribution_params'+model_opts)
        # NNGuide for distribution
        nnguide_dist = scores_methods.NNGuide(module,study_name,cf)
        nnguide_dist.load_params(filename='NNGuide_distribution_params'+model_opts)
        # fDBD for distribution
        fDBD_dist = scores_methods.fDBD(module,study_name,cf)
        fDBD_dist.load_params(filename='fDBD_distribution_params'+model_opts)
        # Mahalanobis distance for distribution
        maha_distance_dist = scores_methods.MahalanobisDistance(cf) 
        maha_distance_dist.load_params(filename='MahalanobisDistance_distribution_params'+model_opts)
        # pNML for distribution
        pnml_dist = scores_methods.pNML(module,study_name,cf)
        pnml_dist.load_params(filename='pNML_distribution_params'+model_opts)
        # ViM Score for distribution
        vim_dist = scores_methods.ViMScore(module,study_name,cf)
        vim_dist.load_params(filename='ViM_distribution_params'+model_opts)
        # Residual Score for distribution
        residual_dist = scores_methods.ResidualScore(module,study_name,cf)
        residual_dist.load_params(filename='Residual_distribution_params'+model_opts)
        # NeCo Score for distribution
        neco_dist = scores_methods.NeCo(module,study_name,cf)
        neco_dist.load_params(filename='NeCo_distribution_params'+model_opts)

        funcs_do = {
            'temperature_scale_dist':               temperature_scale_dist,
            'kpca_global_dist':                     kpca_global_dist,
            'projection_filtering_global_dist':     projection_filtering_global_dist,
            'ctm_global_dist':                      ctm_global_dist,
            'nnguide_global_dist':                  nnguide_global_dist,
            'fDBD_global_dist':                     fDBD_global_dist,
            'maha_distance_global_dist':            maha_distance_global_dist,
            'pnml_global_dist':                     pnml_global_dist,
            'temperature_global_dist':              temperature_global_dist,
            'generalized_entropy_global_dist':      generalized_entropy_global_dist,
            'renyi_entropy_global_dist':            renyi_entropy_global_dist,
            'kpca_class_dist':                      kpca_class_dist,
            'projection_filtering_class_dist':      projection_filtering_class_dist,
            'ctm_class_dist':                       ctm_class_dist,
            'temperature_class_pred_dist':          temperature_class_pred_dist,
            'generalized_entropy_class_pred_dist':  generalized_entropy_class_pred_dist,
            'renyi_entropy_class_pred_dist':        renyi_entropy_class_pred_dist,
            'ctm_class_pred_dist':                  ctm_class_pred_dist,
            'nnguide_class_pred_dist':              nnguide_class_pred_dist,
            'fDBD_class_pred_dist':                 fDBD_class_pred_dist,
            'maha_distance_class_pred_dist':        maha_distance_class_pred_dist,
            'pnml_class_pred_dist':                 pnml_class_pred_dist,
            'temperature_class_dist':               temperature_class_dist,
            'generalized_entropy_class_dist':       generalized_entropy_class_dist,
            'renyi_entropy_class_dist':             renyi_entropy_class_dist,
            'temperature_class_avg_dist':           temperature_class_avg_dist,
            'generalized_entropy_class_avg_dist':   generalized_entropy_class_avg_dist,
            'renyi_entropy_class_avg_dist':         renyi_entropy_class_avg_dist,
            'ctm_class_avg_dist':                   ctm_class_avg_dist,
            'nnguide_class_avg_dist':               nnguide_class_avg_dist,
            'fDBD_class_avg_dist':                  fDBD_class_avg_dist,
            'maha_distance_class_avg_dist':         maha_distance_class_avg_dist,
            'pnml_class_avg_dist':                  pnml_class_avg_dist,
            'ctm_dist':                             ctm_dist,
            'ctm_oc_dist':                          ctm_oc_dist,
            'generalized_entropy_dist':             generalized_entropy_dist,
            'renyi_entropy_dist':                   renyi_entropy_dist,
            'nnguide_dist':                         nnguide_dist,
            'fDBD_dist':                            fDBD_dist,
            'maha_distance_dist':                   maha_distance_dist,
            'pnml_dist':                            pnml_dist,
            'vim_dist':                             vim_dist,
            'residual_dist':                        residual_dist,
            'neco_dist':                            neco_dist,
            }

    if do_enabled: 
        output = (funcs, funcs_do)
    else:
        output = (funcs) 
    return output 

#%%
def stats(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False):
    if do_enabled:
        score_methods, score_methods_do = load_score_methods(cf, module, study_name, do_enabled, model_opts=model_opts)    
        gradnorm_score = scores_methods.GradNorm(module, study_name, cf)
        encoded_distribution = model_evaluations['encoded_dist']
        logits_distribution = model_evaluations['logits_dist']
        softmax_distribution = model_evaluations['softmax_scaled_dist'] if temp_scaled else model_evaluations['softmax_dist']
        preds_distribution = softmax_distribution.mean(dim=2).max(dim=1).indices
        #
        encoded_global_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_backprojection, encoded_distribution)  
        encoded_class_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_backprojection, encoded_distribution)
        # encoded_class_pred_distribution_mcd = torch.vstack([encoded_class_distribution_mcd[preds_distribution[t]][t] for t in range(encoded_distribution.shape[0])])  
        encoded_class_pred_distribution_mcd, logits_class_pred_distribution_mcd = score_methods_do['projection_filtering_class_dist'].get_combined_backprojection(encoded_class_distribution_mcd, combine='prediction', preds=preds_distribution)
        # Backprojections for class averaged for distribution
        # encoded_class_avg_distribution_mcd = []
        # for t in range(encoded_distribution.shape[0]):
        #     avg_sampled = []
        #     for c in range(cf.data.num_classes):
        #         avg_sampled.append(encoded_class_distribution_mcd[c][t])
        #     encoded_class_avg_distribution_mcd.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
        # encoded_class_avg_distribution_mcd = torch.stack(encoded_class_avg_distribution_mcd, dim=0)
        encoded_class_avg_distribution_mcd, logits_class_avg_distribution_mcd = score_methods_do['projection_filtering_class_dist'].get_combined_backprojection(encoded_class_distribution_mcd, combine='average')
        # 
        logits_global_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_logits, encoded_distribution)
        logits_class_distribution_mcd = scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_logits, encoded_distribution)
        softmax_global_distribution_mcd = score_methods_do['temperature_global_dist'].get_scaled_softmax(logits_global_distribution_mcd) if temp_scaled else F.softmax(logits_global_distribution_mcd, dim=1, dtype=torch.float64)
        softmax_class_distribution_mcd = score_methods_do['temperature_class_dist'].get_scaled_softmax(logits_class_distribution_mcd) if temp_scaled else F.softmax(logits_class_distribution_mcd, dim=1, dtype=torch.float64)
        softmax_class_pred_distribution_mcd = score_methods_do['temperature_class_pred_dist'].get_scaled_softmax(logits_class_pred_distribution_mcd) if temp_scaled else F.softmax(logits_class_pred_distribution_mcd, dim=1, dtype=torch.float64)
        softmax_class_avg_distribution_mcd = score_methods_do['temperature_class_avg_dist'].get_scaled_softmax(logits_class_avg_distribution_mcd) if temp_scaled else F.softmax(logits_class_avg_distribution_mcd, dim=1, dtype=torch.float64)        
        # 
        confid_distribution = model_evaluations['confid_dist']
        correct_distribution = model_evaluations['correct_mcd']
        residuals_distribution = 1-correct_distribution
        mcd_confids = {
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_global' : scores_funcs.mcd_function(score_methods_do['kpca_global_dist'].get_scores, encoded_distribution),
            'MCD-KPCA_ERecError_global' : scores_funcs.mcd_expected_function(score_methods_do['kpca_global_dist'].get_scores, encoded_distribution),
            # RecError global for distribution
            'MCD-PCA_RecError_global' : scores_funcs.mcd_function(score_methods_do['projection_filtering_global_dist'].get_scores, encoded_distribution),
            'MCD-PCA_ERecError_global' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_global_dist'].get_scores, encoded_distribution),
            # CTM global for distribution
            'MCD-CTM_global' :          score_methods_do['ctm_global_dist'].get_scores(encoded_global_distribution_mcd, similarity='weight'),
            'MCD-CTM_global_mean' :     score_methods_do['ctm_global_dist'].get_scores(encoded_global_distribution_mcd, similarity='mean'),
            'MCD-ECTM_global' :         score_methods_do['ctm_global_dist'].get_scores(encoded_distribution, similarity='weight'),
            'MCD-ECTM_global_mean' :    score_methods_do['ctm_global_dist'].get_scores(encoded_distribution, similarity='mean'),
            # NNGuide global for distribution
            'MCD-NNGuide_global':       score_methods_do['nnguide_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-ENNGuide_global':      scores_funcs.mcd_expected_function(score_methods_do['nnguide_global_dist'].get_scores, encoded_distribution),
            # fDBD global for distribution
            'MCD-fDBD_global':          score_methods_do['fDBD_global_dist'].get_scores(encoded_global_distribution_mcd, logits_eval=logits_global_distribution_mcd),
            'MCD-EfDBD_global':         scores_funcs.mcd_expected_function(score_methods_do['fDBD_global_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            # Maha Distance global for distribution
            'MCD-Maha_global':          score_methods_do['maha_distance_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-EMaha_global':         scores_funcs.mcd_expected_function(score_methods_do['maha_distance_global_dist'].get_scores, encoded_distribution),
            # pNML global for distribution
            'MCD-pNML_global':          score_methods_do['pnml_global_dist'].get_scores(encoded_global_distribution_mcd),
            'MCD-EpNML_global':         scores_funcs.mcd_expected_function(score_methods_do['pnml_global_dist'].get_scores, encoded_distribution),
            # Entropies global for distribution
            'MCD-GEN_global' :          score_methods_do['generalized_entropy_global_dist'].get_scores(softmax_global_distribution_mcd),
            'MCD-REN_global' :          score_methods_do['renyi_entropy_global_dist'].get_scores(softmax_global_distribution_mcd),
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_class' : scores_funcs.mcd_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution),
            'MCD-KPCA_ERecError_class' : scores_funcs.mcd_expected_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution),
            # RecError class for distribution
            'MCD-PCA_RecError_class' : scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution),
            'MCD-PCA_ERecError_class' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution),
            # Kernel RecError global for distribution
            'MCD-KPCA_RecError_class_pred' : scores_funcs.mcd_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution, predictions_eval=preds_distribution),
            'MCD-KPCA_ERecError_class_pred' : scores_funcs.mcd_expected_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution, predictions_eval=softmax_distribution.max(dim=1).indices),
            # RecError class pred for distribution
            'MCD-PCA_RecError_class_pred' : scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution, X_back_projected_eval=encoded_class_pred_distribution_mcd),
            'MCD-PCA_ERecError_class_pred' : scores_funcs.mcd_expected_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution, predictions_eval=softmax_distribution.max(dim=1).indices),
            # RecError class avg for distribution
            'MCD-PCA_RecError_class_avg' : scores_funcs.mcd_function(score_methods_do['projection_filtering_class_dist'].get_scores, encoded_distribution, X_back_projected_eval=encoded_class_avg_distribution_mcd),
            # Kernel RecError class avg for distribution
            'MCD-KPCA_RecError_class_avg' : scores_funcs.mcd_function(score_methods_do['kpca_class_dist'].get_scores, encoded_distribution, combine='average'),
            # CTM class for distribution
            'MCD-CTM_class' :           score_methods_do['ctm_class_dist'].get_scores(encoded_class_distribution_mcd, similarity='weight'),
            'MCD-CTM_class_mean' :      score_methods_do['ctm_class_dist'].get_scores(encoded_class_distribution_mcd, similarity='mean'),
            # CTM class pred for distribution
            'MCD-CTM_class_pred' :      score_methods_do['ctm_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, similarity='weight'),
            'MCD-CTM_class_pred_mean' : score_methods_do['ctm_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, similarity='mean'),
            'MCD-ECTM_class_pred' :     score_methods_do['ctm_class_pred_dist'].get_scores( encoded_distribution, similarity='weight'),
            'MCD-ECTM_class_pred_mean': score_methods_do['ctm_class_pred_dist'].get_scores( encoded_distribution, similarity='mean'),   
            # NNGuide class pred for distribution
            'MCD-NNGuide_class_pred':   score_methods_do['nnguide_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-ENNGuide_class_pred':  scores_funcs.mcd_expected_function(score_methods_do['nnguide_class_pred_dist'].get_scores, encoded_distribution),
            # fDBD class pred for distribution
            'MCD-fDBD_class_pred':      score_methods_do['fDBD_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd, logits_eval=logits_class_distribution_mcd),
            'MCD-EfDBD_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['fDBD_class_pred_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            # Maha class pred for distribution
            'MCD-Maha_class_pred':      score_methods_do['maha_distance_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-EMaha_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['maha_distance_class_pred_dist'].get_scores, encoded_distribution),
            # pNML class pred for distribution
            'MCD-pNML_class_pred':      score_methods_do['pnml_class_pred_dist'].get_scores(encoded_class_pred_distribution_mcd),
            'MCD-EpNML_class_pred':     scores_funcs.mcd_expected_function(score_methods_do['pnml_class_pred_dist'].get_scores, encoded_distribution),
            # Entropies class for distribution
            'MCD-GEN_class' :           score_methods_do['generalized_entropy_class_dist'].get_scores(softmax_class_distribution_mcd),
            'MCD-REN_class' :           score_methods_do['renyi_entropy_class_dist'].get_scores(softmax_class_distribution_mcd),
            'MCD-GEN_class_avg' :       score_methods_do['generalized_entropy_class_avg_dist'].get_scores(softmax_class_avg_distribution_mcd),
            'MCD-REN_class_avg' :       score_methods_do['renyi_entropy_class_avg_dist'].get_scores(softmax_class_avg_distribution_mcd),
            'MCD-GEN_class_pred' :       score_methods_do['generalized_entropy_class_pred_dist'].get_scores(softmax_class_pred_distribution_mcd),
            'MCD-REN_class_pred' :       score_methods_do['renyi_entropy_class_pred_dist'].get_scores(softmax_class_pred_distribution_mcd),
            # CTM class avg for distribution
            'MCD-CTM_class_avg' :      score_methods_do['ctm_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd, similarity='weight'),
            'MCD-CTM_class_avg_mean' : score_methods_do['ctm_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd, similarity='mean'),
            'MCD-ECTM_class_avg' :     score_methods_do['ctm_class_avg_dist'].get_scores( encoded_distribution, similarity='weight'),
            'MCD-ECTM_class_avg_mean': score_methods_do['ctm_class_avg_dist'].get_scores( encoded_distribution, similarity='mean'),   
            # NNGuide class avg for distribution
            'MCD-NNGuide_class_avg':   score_methods_do['nnguide_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd),
            'MCD-ENNGuide_class_avg':  scores_funcs.mcd_expected_function(score_methods_do['nnguide_class_avg_dist'].get_scores, encoded_distribution),
            # fDBD class avg for distribution
            'MCD-fDBD_class_avg':      score_methods_do['fDBD_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd, logits_eval=logits_class_distribution_mcd),
            'MCD-EfDBD_class_avg':     scores_funcs.mcd_expected_function(score_methods_do['fDBD_class_avg_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            # Maha class avg for distribution
            'MCD-Maha_class_avg':      score_methods_do['maha_distance_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd),
            'MCD-EMaha_class_avg':     scores_funcs.mcd_expected_function(score_methods_do['maha_distance_class_avg_dist'].get_scores, encoded_distribution),
            # pNML class avg for distribution
            'MCD-pNML_class_avg':      score_methods_do['pnml_class_avg_dist'].get_scores(encoded_class_avg_distribution_mcd),
            'MCD-EpNML_class_avg':     scores_funcs.mcd_expected_function(score_methods_do['pnml_class_avg_dist'].get_scores, encoded_distribution),
            # CTM for distribution
            'MCD-CTM' :                scores_funcs.mcd_function(score_methods_do['ctm_dist'].get_scores, encoded_distribution, similarity='weight'),
            'MCD-ECTM' :               score_methods_do['ctm_dist'].get_scores(encoded_distribution, similarity='weight'),
            'MCD-CTM_mean' :           scores_funcs.mcd_function(score_methods_do['ctm_dist'].get_scores, encoded_distribution, similarity='mean'),
            'MCD-ECTM_mean' :          score_methods_do['ctm_dist'].get_scores(encoded_distribution, similarity='mean'),
            # CTM (only correct) for distribution
            'MCD-CTM_oc_mean':          scores_funcs.mcd_function(score_methods_do['ctm_oc_dist'].get_scores, encoded_distribution, similarity='mean'),
            'MCD-ECTM_oc_mean':         score_methods_do['ctm_oc_dist'].get_scores(encoded_distribution, similarity='mean'),            
            # Entropies for distribution
            'MCD-GEN' :                 scores_funcs.mcd_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
            'MCD-EGEN' :                scores_funcs.mcd_expected_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
            'MCD-REN' :                 scores_funcs.mcd_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
            'MCD-EREN' :                scores_funcs.mcd_expected_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
            # NNGuide for distribution
            'MCD-NNGuide':              scores_funcs.mcd_function(score_methods_do['nnguide_dist'].get_scores, encoded_distribution),
            'MCD-ENNGuide':             scores_funcs.mcd_expected_function(score_methods_do['nnguide_dist'].get_scores, encoded_distribution),
            # fDBD for distribution
            'MCD-fDBD':                 scores_funcs.mcd_function(score_methods_do['fDBD_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            'MCD-EfDBD':                scores_funcs.mcd_expected_function(score_methods_do['fDBD_dist'].get_scores, encoded_distribution, logits_eval=logits_distribution),
            # Maha for distribution
            'MCD-Maha':                 scores_funcs.mcd_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
            'MCD-EMaha':                scores_funcs.mcd_expected_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
            # pNML for distribution
            'MCD-pNML':                 scores_funcs.mcd_function(score_methods_do['pnml_dist'].get_scores, encoded_distribution),
            'MCD-EpNML':                scores_funcs.mcd_expected_function(score_methods_do['pnml_dist'].get_scores, encoded_distribution),
            # ViM for distribution
            'MCD-ViM':                 scores_funcs.mcd_function(score_methods_do['vim_dist'].get_scores, encoded_distribution),
            'MCD-EViM':                scores_funcs.mcd_expected_function(score_methods_do['vim_dist'].get_scores, encoded_distribution),
            # Residual for distribution
            'MCD-Residual':             scores_funcs.mcd_function(score_methods_do['residual_dist'].get_scores, encoded_distribution),
            'MCD-EResidual':            scores_funcs.mcd_expected_function(score_methods_do['residual_dist'].get_scores, encoded_distribution),
            # NeCo for distribution
            'MCD-NeCo':                 scores_funcs.mcd_function(score_methods_do['neco_dist'].get_scores, encoded_distribution),
            'MCD-ENeCo':                scores_funcs.mcd_expected_function(score_methods_do['neco_dist'].get_scores, encoded_distribution),            
            # Scores that do not requiere preprocessing
            'MCD-MSR' :                 scores_funcs.mcd_function(scores_funcs.maximum_softmax_response, softmax_distribution),
            'MCD-PE' :                  scores_funcs.mcd_function(scores_funcs.predictive_entropy, softmax_distribution),
            'MCD-MLS' :                 scores_funcs.mcd_function(scores_funcs.maximum_logit_score, logits_distribution, temperature=score_methods['temperature_scale'].temperature),
            'MCD-PCE' :                 scores_funcs.mcd_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
            'MCD-GE' :                  scores_funcs.mcd_function(scores_funcs.guessing_entropy, softmax_distribution),
            'MCD-Energy' :              scores_funcs.mcd_function(scores_funcs.energy, logits_distribution, temperature=score_methods_do['temperature_scale_dist'].temperature),
            'MCD-EMSR' :                scores_funcs.mcd_expected_function(scores_funcs.maximum_softmax_response, softmax_distribution),
            'MCD-EPE' :                 scores_funcs.mcd_expected_function(scores_funcs.predictive_entropy, softmax_distribution),
            'MCD-EMLS' :                scores_funcs.mcd_expected_function(scores_funcs.maximum_logit_score, logits_distribution, temperature=score_methods['temperature_scale'].temperature),    
            'MCD-EPCE' :                scores_funcs.mcd_expected_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
            'MCD-EGE' :                 scores_funcs.mcd_expected_function(scores_funcs.guessing_entropy, softmax_distribution),
            'MCD-EEnergy' :             scores_funcs.mcd_expected_function(scores_funcs.energy, logits_distribution, temperature=score_methods_do['temperature_scale_dist'].temperature),    
            # Scores that do not requiere preprocessing using global projection filtering
            'MCD-MSR_global' :         scores_funcs.maximum_softmax_response(softmax_global_distribution_mcd),
            'MCD-PE_global' :          scores_funcs.predictive_entropy(softmax_global_distribution_mcd),
            'MCD-MLS_global' :         scores_funcs.maximum_logit_score(logits_global_distribution_mcd, temperature=score_methods_do['temperature_global_dist'].temperature),
            'MCD-PCE_global' :         scores_funcs.predictive_collision_entropy(softmax_global_distribution_mcd),
            'MCD-GE_global' :          scores_funcs.guessing_entropy(softmax_global_distribution_mcd),
            'MCD-Energy_global' :      scores_funcs.energy(logits_global_distribution_mcd, temperature=score_methods_do['temperature_global_dist'].temperature),
            # Scores that do not requiere preprocessing using class projection filtering
            'MCD-MSR_class' :         scores_funcs.maximum_softmax_response(softmax_class_distribution_mcd),
            'MCD-PE_class' :          scores_funcs.predictive_entropy(softmax_class_distribution_mcd),
            'MCD-MLS_class' :         scores_funcs.maximum_logit_score(logits_class_distribution_mcd, temperature=score_methods_do['temperature_class_dist'].temperature),
            'MCD-PCE_class' :         scores_funcs.predictive_collision_entropy(softmax_class_distribution_mcd),
            'MCD-GE_class' :          scores_funcs.guessing_entropy(softmax_class_distribution_mcd),
            'MCD-Energy_class' :      scores_funcs.energy(logits_class_distribution_mcd, temperature=score_methods_do['temperature_class_dist'].temperature),
            # Scores that do not requiere preprocessing using class projection filtering
            'MCD-MSR_class_avg' :         scores_funcs.maximum_softmax_response(softmax_class_avg_distribution_mcd),
            'MCD-PE_class_avg' :          scores_funcs.predictive_entropy(softmax_class_avg_distribution_mcd),
            'MCD-MLS_class_avg' :         scores_funcs.maximum_logit_score(logits_class_avg_distribution_mcd, temperature=score_methods_do['temperature_class_avg_dist'].temperature),
            'MCD-PCE_class_avg' :         scores_funcs.predictive_collision_entropy(softmax_class_avg_distribution_mcd),
            'MCD-GE_class_avg' :          scores_funcs.guessing_entropy(softmax_class_avg_distribution_mcd),
            'MCD-Energy_class_avg' :      scores_funcs.energy(logits_class_avg_distribution_mcd, temperature=score_methods_do['temperature_class_avg_dist'].temperature),
            # Scores that do not requiere preprocessing using class projection filtering
            'MCD-MSR_class_pred' :         scores_funcs.maximum_softmax_response(softmax_class_pred_distribution_mcd),
            'MCD-PE_class_pred' :          scores_funcs.predictive_entropy(softmax_class_pred_distribution_mcd),
            'MCD-MLS_class_pred' :         scores_funcs.maximum_logit_score(logits_class_pred_distribution_mcd, temperature=score_methods_do['temperature_class_pred_dist'].temperature),
            'MCD-PCE_class_pred' :         scores_funcs.predictive_collision_entropy(softmax_class_pred_distribution_mcd),
            'MCD-GE_class_pred' :          scores_funcs.guessing_entropy(softmax_class_pred_distribution_mcd),
            'MCD-Energy_class_pred' :      scores_funcs.energy(logits_class_pred_distribution_mcd, temperature=score_methods_do['temperature_class_pred_dist'].temperature),
            #             
            'MCD-GradNorm' :            scores_funcs.mcd_function(gradnorm_score.get_scores, encoded_distribution, use_cuda=use_cuda, temperature=score_methods_do['temperature_scale_dist'].temperature),
            'MCD-GradNorm_global' :     gradnorm_score.get_scores(encoded_global_distribution_mcd, use_cuda=use_cuda, temperature=score_methods_do['temperature_global_dist'].temperature),
            'MCD-GradNorm_class_avg' :  gradnorm_score.get_scores(encoded_class_avg_distribution_mcd, use_cuda=use_cuda, temperature=score_methods_do['temperature_class_avg_dist'].temperature),
            'MCD-GradNorm_class_pred' : gradnorm_score.get_scores(encoded_class_pred_distribution_mcd, use_cuda=use_cuda, temperature=score_methods_do['temperature_class_pred_dist'].temperature),
            #
            'MCD-MI' :          scores_funcs.mcd_mutual_information(softmax_distribution),
            'MCD-Confidence' :  confid_distribution.mean(dim=1),
        }
        mcd_confids_df = pd.DataFrame(mcd_confids)
        mcd_confids_df['residuals'] = residuals_distribution
        mcd_stats = {
                        key:[RiskCoverageStats(confids=mcd_confids[key], residuals=residuals_distribution), metrics.StatsCache(mcd_confids[key],correct_distribution,n_bins) ] for key in mcd_confids   
                    }

        mcd_stats_df = pd.DataFrame( {  
                                        'AUGRC': { k:mcd_stats[k][0].augrc for k in mcd_stats },
                                        'AURC': { k:mcd_stats[k][0].aurc for k in mcd_stats },
                                        'AUROC_f': { k:metrics.failauc(mcd_stats[k][1]) for k in mcd_stats },
                                        'FPR@95TPR': { k:metrics.fpr_at_95_tpr(mcd_stats[k][1]) for k in mcd_stats },
                                        'ECE': { k:metrics.expected_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
                                        'MCE': { k:metrics.maximum_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
                                        'AP_ferr': { k:metrics.failap_err(mcd_stats[k][1]) for k in mcd_stats },
                                        'AP_fsuc': { k:metrics.failap_suc(mcd_stats[k][1]) for k in mcd_stats },
                                    } )
        filename = f'mcdstats{model_opts}_{eval_name}.csv'
        filename_confids = f'mcdconfids{model_opts}_{eval_name}.csv'
        if os.path.exists(f'{cf.exp.dir}/analysis'):
            path = f'{cf.exp.dir}/analysis/{filename}'
            path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
        else:
            os.mkdir(f'{cf.exp.dir}/analysis')
            path = f'{cf.exp.dir}/analysis/{filename}'
            path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
        mcd_stats_df.sort_values(by=['AUGRC']).to_csv(path)
        mcd_confids_df.to_csv(path_confids)
    else:
        score_methods = load_score_methods(cf, module, study_name, do_enabled, model_opts=model_opts)

    gradnorm_score = scores_methods.GradNorm(module, study_name, cf)
    encoded = model_evaluations['encoded']
    logits = model_evaluations['logits']
    softmax = model_evaluations['softmax_scaled'] if temp_scaled else model_evaluations['softmax']
    preds = softmax.max(dim=1).indices 
    #
    encoded_global = score_methods['projection_filtering_global'].get_backprojection(encoded)
    encoded_class = score_methods['projection_filtering_class'].get_backprojection(encoded)
    # encoded_class_pred = torch.vstack([encoded_class[preds[t]][t] for t in range(encoded.shape[0])])
    encoded_class_pred, logits_class_pred = score_methods['projection_filtering_class'].get_combined_backprojection(encoded_class, combine='prediction', preds=preds)
    logits_global = score_methods['projection_filtering_global'].get_logits(encoded)
    logits_class = score_methods['projection_filtering_class'].get_logits(encoded)
    softmax_global = score_methods['temperature_global'].get_scaled_softmax(logits_global) if temp_scaled else F.softmax(logits_global, dim=1, dtype=torch.float64)
    softmax_class = score_methods['temperature_class'].get_scaled_softmax(logits_class) if temp_scaled else F.softmax(logits_class, dim=1, dtype=torch.float64)
    
    #
    # Backprojections for class averaged
    # encoded_class_avg = []
    # for t in range(encoded.shape[0]):
    #     avg_sample = []
    #     for c in range(cf.data.num_classes):
    #         avg_sample.append(encoded_class[c][t])
    #     encoded_class_avg.append(torch.stack(avg_sample, dim=0).mean(dim=0))
    # encoded_class_avg = torch.stack(encoded_class_avg, dim=0)
    encoded_class_avg, logits_class_avg = score_methods['projection_filtering_class'].get_combined_backprojection(encoded_class, combine='average')
    softmax_class_avg = score_methods['temperature_class_avg'].get_scaled_softmax(logits_class_avg) if temp_scaled else F.softmax(logits_class_avg, dim=1, dtype=torch.float64)
    softmax_class_pred = score_methods['temperature_class_pred'].get_scaled_softmax(logits_class_pred) if temp_scaled else F.softmax(logits_class_pred, dim=1, dtype=torch.float64)
    #
    confid = model_evaluations['confid']
    correct = model_evaluations['correct']
    residuals = 1-correct

    confids= {
                # KPCA RecError global
                'KPCA_RecError_global':  score_methods['kpca_global'].get_scores(encoded),
                # RecError global
                'PCA_RecError_global':  score_methods['projection_filtering_global'].get_scores(encoded),
                # CTM global
                'CTM_global':           score_methods['ctm_global'].get_scores(encoded_global, similarity='weight'),
                'CTM_global_mean':      score_methods['ctm_global'].get_scores(encoded_global, similarity='mean'),
                # NNGuide global 
                'NNGuide_global':       score_methods['nnguide_global'].get_scores(encoded_global),
                # fDBD global
                'fDBD_global':          score_methods['fDBD_global'].get_scores(encoded_global, logits_eval=logits_global),
                # Maha global
                'Maha_global':          score_methods['maha_distance_global'].get_scores(encoded_global),
                # pNML global
                'pNML_global':          score_methods['pnml_global'].get_scores(encoded_global),
                # Entropies
                'GEN_global' :          score_methods['generalized_entropy_global'].get_scores(softmax_global),
                'REN_global' :          score_methods['renyi_entropy_global'].get_scores(softmax_global),
                # KPCA RecError class
                'KPCA_RecError_class':  score_methods['kpca_class'].get_scores(encoded),
                # KPCA RecError class pred
                'KPCA_RecError_class_pred':  score_methods['kpca_class'].get_scores(encoded,predictions_eval=preds),
                # KPCA RecError class pred
                'KPCA_RecError_class_avg':  score_methods['kpca_class'].get_scores(encoded,combine='average'),
                # RecError class
                'PCA_RecError_class':   score_methods['projection_filtering_class'].get_scores(encoded),
                # RecError class pred
                'PCA_RecError_class_pred':  score_methods['projection_filtering_class'].get_scores(encoded, X_back_projected_eval=encoded_class_pred),
                # RecError class avg
                'PCA_RecError_class_avg':  score_methods['projection_filtering_class'].get_scores(encoded, X_back_projected_eval=encoded_class_avg),
                # CTM class
                'CTM_class':            score_methods['ctm_class'].get_scores(encoded_class, similarity='weight'),
                'CTM_class_mean':       score_methods['ctm_class'].get_scores(encoded_class, similarity='mean'),
                # CTM class pred
                'CTM_class_pred':       score_methods['ctm_class_pred'].get_scores(encoded_class_pred, similarity='weight'),
                'CTM_class_pred_mean':  score_methods['ctm_class_pred'].get_scores(encoded_class_pred, similarity='mean'),
                # NNGuide class pred
                'NNGuide_class_pred':   score_methods['nnguide_class_pred'].get_scores(encoded_class_pred),
                # fDBD class pred
                'fDBD_class_pred':      score_methods['fDBD_class_pred'].get_scores(encoded_class_pred, logits_eval=logits_class),
                # Maha class pred
                'Maha_class_pred':      score_methods['maha_distance_class_pred'].get_scores(encoded_class_pred),
                # pNML class pred
                'pNML_class_pred':      score_methods['pnml_class_pred'].get_scores(encoded_class_pred),
                # Entropies
                'GEN_class' :           score_methods['generalized_entropy_class'].get_scores(softmax_class),
                'REN_class' :           score_methods['renyi_entropy_class'].get_scores(softmax_class),
                'GEN_class_avg' :       score_methods['generalized_entropy_class_avg'].get_scores(softmax_class_avg),
                'REN_class_avg' :       score_methods['renyi_entropy_class_avg'].get_scores(softmax_class_avg),
                'GEN_class_pred' :      score_methods['generalized_entropy_class_pred'].get_scores(softmax_class_pred),
                'REN_class_pred' :      score_methods['renyi_entropy_class_pred'].get_scores(softmax_class_pred),
                # CTM class avg
                'CTM_class_avg':        score_methods['ctm_class_avg'].get_scores(encoded_class_avg, similarity='weight'),
                'CTM_class_avg_mean':   score_methods['ctm_class_avg'].get_scores(encoded_class_avg, similarity='mean'),
                # NNGuide class avg
                'NNGuide_class_avg':    score_methods['nnguide_class_avg'].get_scores(encoded_class_avg),
                # fDBD class avg
                'fDBD_class_avg':       score_methods['fDBD_class_avg'].get_scores(encoded_class_avg, logits_eval=logits_class),
                # Maha class avg
                'Maha_class_avg':       score_methods['maha_distance_class_avg'].get_scores(encoded_class_avg),
                # pNML class avg
                'pNML_class_avg':       score_methods['pnml_class_avg'].get_scores(encoded_class_avg),
                # CTM
                'CTM':                  score_methods['ctm'].get_scores(encoded, similarity='weight'),
                'CTM_mean':             score_methods['ctm'].get_scores(encoded, similarity='mean'),
                'CTM_oc_mean':          score_methods['ctm_oc'].get_scores(encoded, similarity='mean'),
                # Entropies
                'GEN' :                 score_methods['generalized_entropy'].get_scores(softmax),
                'REN' :                 score_methods['renyi_entropy'].get_scores(softmax),
                # NNGuide
                'NNGuide':              score_methods['nnguide'].get_scores(encoded),
                # fDBD
                'fDBD':                 score_methods['fDBD'].get_scores(encoded, logits_eval=logits),
                # Maha
                'Maha':                 score_methods['maha_distance'].get_scores(encoded),
                # pNML
                'pNML':                 score_methods['pnml'].get_scores(encoded),
                # ViM
                'ViM':                  score_methods['vim'].get_scores(encoded),
                # Residual
                'Residual':             score_methods['residual'].get_scores(encoded),
                # NeCo
                'NeCo':                 score_methods['neco'].get_scores(encoded),  
                # Scores that do not requiere preprocessing
                'MSR' :                 scores_funcs.maximum_softmax_response(softmax),
                'PE' :                  scores_funcs.predictive_entropy(softmax),
                'MLS' :                 scores_funcs.maximum_logit_score(logits, temperature=score_methods['temperature_scale'].temperature),
                'PCE' :                 scores_funcs.predictive_collision_entropy(softmax),
                'GE' :                  scores_funcs.guessing_entropy(softmax),
                'Energy' :              scores_funcs.energy(logits, temperature=score_methods['temperature_scale'].temperature),
                # Scores that do not requiere preprocessing using global projection filtering
                'MSR_global' :          scores_funcs.maximum_softmax_response(softmax_global),
                'PE_global' :           scores_funcs.predictive_entropy(softmax_global),
                'MLS_global' :          scores_funcs.maximum_logit_score(logits_global, temperature=score_methods['temperature_global'].temperature),
                'PCE_global' :          scores_funcs.predictive_collision_entropy(softmax_global),
                'GE_global' :           scores_funcs.guessing_entropy(softmax_global),
                'Energy_global' :       scores_funcs.energy(logits_global, temperature=score_methods['temperature_global'].temperature),
                # Scores that do not requiere preprocessing using class projection filtering
                'MSR_class' :           scores_funcs.maximum_softmax_response(softmax_class),
                'MSR_class_avg' :       scores_funcs.maximum_softmax_response(softmax_class_avg),
                'MSR_class_pred' :      scores_funcs.maximum_softmax_response(softmax_class_pred),
                'PE_class' :            scores_funcs.predictive_entropy(softmax_class),
                'PE_class_avg' :        scores_funcs.predictive_entropy(softmax_class_avg),
                'PE_class_pred' :       scores_funcs.predictive_entropy(softmax_class_pred),
                'MLS_class' :           scores_funcs.maximum_logit_score(logits_class, temperature=score_methods['temperature_class'].temperature),
                'MLS_class_avg' :       scores_funcs.maximum_logit_score(logits_class_avg, temperature=score_methods['temperature_class_avg'].temperature),
                'MLS_class_pred' :      scores_funcs.maximum_logit_score(logits_class_pred, temperature=score_methods['temperature_class_pred'].temperature),
                'PCE_class' :           scores_funcs.predictive_collision_entropy(softmax_class),
                'PCE_class_avg' :       scores_funcs.predictive_collision_entropy(softmax_class_avg),
                'PCE_class_pred' :      scores_funcs.predictive_collision_entropy(softmax_class_pred),
                'GE_class' :            scores_funcs.guessing_entropy(softmax_class),
                'GE_class_avg' :        scores_funcs.guessing_entropy(softmax_class_avg),
                'GE_class_pred' :       scores_funcs.guessing_entropy(softmax_class_pred),
                'Energy_class' :        scores_funcs.energy(logits_class, temperature=score_methods['temperature_class'].temperature),
                'Energy_class_avg' :    scores_funcs.energy(logits_class_avg, temperature=score_methods['temperature_class_avg'].temperature),
                'Energy_class_pred' :   scores_funcs.energy(logits_class_pred, temperature=score_methods['temperature_class_pred'].temperature),
                # 
                'GradNorm' :            gradnorm_score.get_scores(encoded, temperature=score_methods['temperature_scale'].temperature, use_cuda=use_cuda),
                'GradNorm_global' :     gradnorm_score.get_scores(encoded_global, temperature=score_methods['temperature_global'].temperature, use_cuda=use_cuda),
                'GradNorm_class_avg' :  gradnorm_score.get_scores(encoded_class_avg, temperature=score_methods['temperature_class_avg'].temperature, use_cuda=use_cuda),
                'GradNorm_class_pred' : gradnorm_score.get_scores(encoded_class_pred, temperature=score_methods['temperature_class_pred'].temperature, use_cuda=use_cuda),    
                'Confidence' :          confid,
    }
    confids_df = pd.DataFrame(confids)
    confids_df['residuals'] = residuals
    stats = {
                key:[ RiskCoverageStats(confids=confids[key], residuals=residuals), metrics.StatsCache(confids[key],correct,n_bins) ] for key in confids
            }
    # print([ print(f'{k}:{stats[k][0].augrc}') for k in stats])
    stats_df = pd.DataFrame( {  
                                'AUGRC': { k:stats[k][0].augrc for k in stats },
                                'AURC': { k:stats[k][0].aurc for k in stats },
                                'AUROC_f': { k:metrics.failauc(stats[k][1]) for k in stats },
                                'FPR@95TPR': { k:metrics.fpr_at_95_tpr(stats[k][1]) for k in stats },
                                'ECE': { k:metrics.expected_calibration_error(stats[k][1]) for k in stats },
                                'MCE': { k:metrics.maximum_calibration_error(stats[k][1]) for k in stats },
                                'AP_ferr': { k:metrics.failap_err(stats[k][1]) for k in stats },
                                'AP_fsuc': { k:metrics.failap_suc(stats[k][1]) for k in stats },
                            } )
    filename = f'stats{model_opts}_{eval_name}.csv'
    filename_confids = f'confids{model_opts}_{eval_name}.csv'
    if os.path.exists(f'{cf.exp.dir}/analysis'):
        path = f'{cf.exp.dir}/analysis/{filename}'
        path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
    else:
        os.mkdir(f'{cf.exp.dir}/analysis')
        path = f'{cf.exp.dir}/analysis/{filename}'
        path_confids = f'{cf.exp.dir}/analysis/{filename_confids}'
    # mcd_stats_df.to_csv(path)
    stats_df.sort_values(by=['AUGRC']).to_csv(path)
    confids_df.to_csv(path_confids)

#%%
def compute_metrics(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False):
    if 'cifar' in cf.data.dataset:
        if eval_name == 'iid_test':
            key_dict = 'test_1'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
        elif eval_name == 'iid_val':
            key_dict = 'val'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
        elif eval_name == 'iid_test_corruptions':
            key_dict = 'test_2_sampled_20'
            evaluations = model_evaluations[key_dict]
            # Make sure that all the keys have the same length of 
            lengths = [ len(evaluations[key]) for key in evaluations.keys() if evaluations[key] is not None ]
            assert len(set(lengths))==1, 'Evaluations do not have the same dimensions'
            n_samples = lengths[0]//5 # Corruptions of 5 different types
            for i in tqdm(range(5)):
                logger.info(f'Evaluating test set with corruption type {i+1}...')
                evaluations_grouped = {key:evaluations[key][n_samples*i:n_samples*(i+1)] for key in evaluations.keys() if evaluations[key] is not None}
                stats(module, study_name, cf, evaluations_grouped, eval_name+f'_{i+1}', do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
        elif 'ood' in eval_name:
            # pick only correct predictions of the iid test set. Predictions are based on max softmax.
            evaluations_iid = model_evaluations['test_1']
            ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
            assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
            evaluations_ood = model_evaluations[ood_set_number[0]]
            # predictions for ood samples should be all incorrect
            evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
            # filtering criteria
            correct = evaluations_iid['correct']
            # print(evaluations_iid.keys())
            evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if ('_dist' not in key) else None) for key in evaluations_iid.keys() }
            if do_enabled:
                # predictions for ood samples should be all incorrect
                evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
                # filtering criteria
                correct_mcd = evaluations_iid['correct_mcd']
                evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
                print(evaluations_iid_filtered.keys())
                print(evaluations_filtered_mcd.keys())
                evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
            assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
            keys = evaluations_iid_filtered.keys()
            lengths = [ (key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) ]
            # print(lengths)
            evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
            stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
    elif 'tiny' in cf.data.dataset:
        logger.info(f'Evaluating {eval_name} with {cf.data.dataset}')
        if eval_name == 'iid_test':
            key_dict = 'test_1'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
        elif eval_name == 'iid_val':
            key_dict = 'val'
            stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled) 
        elif 'ood' in eval_name:
            # pick only correct predictions of the iid test set. Predictions are based on max softmax.
            evaluations_iid = model_evaluations['test_1']
            ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
            assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
            evaluations_ood = model_evaluations[ood_set_number[0]]
            # predictions for ood samples should be all incorrect
            evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
            # filtering criteria
            correct = evaluations_iid['correct']
            # print(evaluations_iid.keys())
            evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if ('_dist' not in key) else None) for key in evaluations_iid.keys() }
            if do_enabled:
                # predictions for ood samples should be all incorrect
                evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
                # filtering criteria
                correct_mcd = evaluations_iid['correct_mcd']
                evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
                print(evaluations_iid_filtered.keys())
                print(evaluations_filtered_mcd.keys())
                evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
            assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
            keys = evaluations_iid_filtered.keys()
            lengths = [ (key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) ]
            # print(lengths)
            evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
            stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
#%%
# def compute_metrics_eval(module, study_name, cf, model_evaluations, eval_name:str, do_enabled:bool, model_opts:str='', n_bins:int=20, temp_scaled:bool=False):
#     # if cf.data?
#     if eval_name == 'iid_test':
#         key_dict = 'test_1'
#         stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
#     elif eval_name == 'iid_val':
#         key_dict = 'val'
#         stats(module, study_name, cf, model_evaluations[key_dict], eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
#     elif eval_name == 'iid_test_corruptions':
#         key_dict = 'test_2'
#         evaluations = model_evaluations[key_dict]
#         # Make sure that all the keys have the same length of 
#         lengths = [ len(evaluations[key]) for key in evaluations.keys() if evaluations[key] is not None ]
#         assert len(set(lengths))==1, 'Evaluations do not have the same dimensions'
#         n_samples = lengths[0]//5 # Corruptions of 5 different types
#         for i in tqdm(range(5)):
#             logger.info(f'Evaluating test set with corruption type {i+1}...')
#             evaluations_grouped = {key:evaluations[key][n_samples*i:n_samples*(i+1)] for key in evaluations.keys() if evaluations[key] is not None}
#             stats(module, study_name, cf, evaluations_grouped, eval_name+f'_{i+1}', do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
#     elif 'ood' in eval_name:
#         # pick only correct predictions of the iid test set. Predictions are based on max softmax.
#         evaluations_iid = model_evaluations['test_1']
#         ood_set_number = [set_name for set_name in model_evaluations.keys() if set_name!='test_1']
#         assert len(ood_set_number)==1, 'Just one OOD set should be include at a time...'
#         evaluations_ood = model_evaluations[ood_set_number[0]]
#         # predictions for ood samples should be all incorrect
#         evaluations_ood['correct'] = torch.zeros_like( evaluations_ood['correct'] ).long()
#         # filtering criteria
#         correct = evaluations_iid['correct']
#         evaluations_iid_filtered = { key:(evaluations_iid[key][correct==1] if (('_dist' not in key) or ('_mcd' not in key)) else None) for key in evaluations_iid.keys() }
#         if do_enabled:
#             # predictions for ood samples should be all incorrect
#             evaluations_ood['correct_mcd'] = torch.zeros_like( evaluations_ood['correct_mcd'] ).long()
#             # filtering criteria
#             correct_mcd = evaluations_iid['correct_mcd']
#             evaluations_filtered_mcd = { key:evaluations_iid[key][correct_mcd==1] for key in evaluations_iid.keys() if (('_dist' in key) or ('_mcd' in key)) }
#             evaluations_iid_filtered = evaluations_iid_filtered | evaluations_filtered_mcd
#         assert evaluations_iid_filtered.keys()==evaluations_ood.keys(), 'IID and OOD dictionaries should have the same keys...'
#         keys = evaluations_iid_filtered.keys()
#         lengths = [(key,len(evaluations_iid_filtered[key]),len(evaluations_ood[key])) for key in keys]
#         # print(lengths)
#         evaluations_joint = {key:torch.concat([evaluations_iid_filtered[key],evaluations_ood[key]],dim=0) for key in keys if ((evaluations_iid_filtered[key] is not None) and (evaluations_ood[key] is not None)) }
#         stats(module, study_name, cf, evaluations_joint, eval_name, do_enabled, model_opts=model_opts, n_bins=n_bins, temp_scaled=temp_scaled)
        # change here
    # if do_enabled:
    #     score_methods, score_methods_do = load_score_methods(cf, module, study_name, do_enabled)    
    #     gradnorm_score = scores_methods.GradNorm(module, study_name, cf)
    #     encoded_distribution = model_evaluations[key_dict]['encoded_dist']
    #     logits_distribution = model_evaluations[key_dict]['logits_dist']
    #     softmax_distribution = model_evaluations[key_dict]['softmax_scaled_dist'] if temp_scaled else model_evaluations[key_dict]['softmax_dist']
    #     # model_evaluations['train']['softmax_scaled_dist'] if temp_scaled else model_evaluations['train']['softmax_dist'] 
    #     confid_distribution = model_evaluations[key_dict]['confid_dist']
    #     correct_distribution = model_evaluations[key_dict]['correct_mcd']
    #     residuals_distribution = 1-correct_distribution
    #     mcd_confids = {
    #         'MCD-Confidence' :  confid_distribution.mean(dim=1),
    #         'MCD-MSR' :         scores_funcs.mcd_function(scores_funcs.maximum_softmax_response, softmax_distribution),
    #         'MCD-EMSR' :        scores_funcs.mcd_expected_function(scores_funcs.maximum_softmax_response, softmax_distribution),
    #         'MCD-PE' :          scores_funcs.mcd_function(scores_funcs.predictive_entropy, softmax_distribution),
    #         'MCD-EPE' :         scores_funcs.mcd_expected_function(scores_funcs.predictive_entropy, softmax_distribution),
    #         'MCD-MLS' :         scores_funcs.mcd_function(scores_funcs.maximum_logit_score, logits_distribution),
    #         'MCD-EMLS' :        scores_funcs.mcd_expected_function(scores_funcs.maximum_logit_score, logits_distribution),    
    #         'MCD-PCE' :         scores_funcs.mcd_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
    #         'MCD-EPCE' :        scores_funcs.mcd_expected_function(scores_funcs.predictive_collision_entropy, softmax_distribution),
    #         'MCD-GE' :          scores_funcs.mcd_function(scores_funcs.guessing_entropy, softmax_distribution),
    #         'MCD-EGE' :         scores_funcs.mcd_expected_function(scores_funcs.guessing_entropy, softmax_distribution),
    #         'MCD-Energy' :      scores_funcs.mcd_function(scores_funcs.energy, logits_distribution, T=score_methods_do['temperature_scale_dist'].temperature),
    #         'MCD-EEnergy' :     scores_funcs.mcd_expected_function(scores_funcs.energy, logits_distribution, T=score_methods_do['temperature_scale_dist'].temperature),    
    #         'MCD-Maha':         scores_funcs.mcd_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
    #         'MCD-EMaha':        scores_funcs.mcd_expected_function(score_methods_do['maha_distance_dist'].get_scores, encoded_distribution),
    #         'MCD-ViM':          scores_funcs.mcd_function(score_methods_do['vim_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EViM':         scores_funcs.mcd_expected_function(score_methods_do['vim_score_dist'].get_scores, encoded_distribution),
    #         'MCD-Residual':     scores_funcs.mcd_function(score_methods_do['residual_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EResidual':    scores_funcs.mcd_expected_function(score_methods_do['residual_score_dist'].get_scores, encoded_distribution),
    #         'MCD-NeCo':         scores_funcs.mcd_function(score_methods_do['neco_score_dist'].get_scores, encoded_distribution),
    #         'MCD-ENeCo':        scores_funcs.mcd_expected_function(score_methods_do['neco_score_dist'].get_scores, encoded_distribution),
    #         'MCD-pNML':         scores_funcs.mcd_function(score_methods_do['pnml_score_dist'].get_scores, encoded_distribution),
    #         'MCD-EpNML':        scores_funcs.mcd_expected_function(score_methods_do['pnml_score_dist'].get_scores, encoded_distribution),
    #         'MCD-KLMatching' :  scores_funcs.mcd_function(score_methods_do['klmatching_score_dist'].get_scores, softmax_distribution),
    #         'MCD-EKLMatching' : scores_funcs.mcd_expected_function(score_methods_do['klmatching_score_dist'].get_scores, softmax_distribution),
    #         'MCD-GEN' :         scores_funcs.mcd_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-EGEN' :        scores_funcs.mcd_expected_function(score_methods_do['generalized_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-REN' :         scores_funcs.mcd_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-EREN' :        scores_funcs.mcd_expected_function(score_methods_do['renyi_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-TEN' :         scores_funcs.mcd_function(score_methods_do['tsallis_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-ETEN' :        scores_funcs.mcd_expected_function(score_methods_do['tsallis_entropy_dist'].get_scores, softmax_distribution),
    #         'MCD-GradNorm' :    scores_funcs.mcd_function(gradnorm_score.get_scores, encoded_distribution),
    #         'MCD-MI' :          scores_funcs.mcd_mutual_information(softmax_distribution),
    #     } 
    #     mcd_stats = {
    #                     key:[RiskCoverageStats(confids=mcd_confids[key], residuals=residuals_distribution), metrics.StatsCache(mcd_confids[key],correct_distribution,n_bins) ] for key in mcd_confids   
    #                 }

    #     mcd_stats_df = pd.DataFrame( {  
    #                                     'AUGRC': { k:mcd_stats[k][0].augrc for k in mcd_stats },
    #                                     'AURC': { k:mcd_stats[k][0].aurc for k in mcd_stats },
    #                                     'AUROC_f': { k:metrics.failauc(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'FPR95': { k:metrics.fpr_at_95_tpr(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'ECE': { k:metrics.expected_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
    #                                     'MCE': { k:metrics.maximum_calibration_error(mcd_stats[k][1]) for k in mcd_stats },
    #                                 } )
    #     filename = f'mcd_stats_{model_name}_{eval_name}.csv'
    #     if os.path.exists(f'{cf.exp.dir}/analysis'):
    #         path = f'{cf.exp.dir}/analysis/{filename}'
    #     else:
    #         os.mkdir(f'{cf.exp.dir}/analysis')
    #         path = f'{cf.exp.dir}/analysis/{filename}'
    #     mcd_stats_df.sort_values(by=['AUGRC']).to_csv(path)
    # else:
    #     score_methods = load_score_methods(cf, module, study_name, do_enabled)

    # gradnorm_score = scores_methods.GradNorm(module, study_name, cf)
    # encoded = model_evaluations[key_dict]['encoded']
    # logits = model_evaluations[key_dict]['logits']
    # softmax = model_evaluations[key_dict]['softmax_scaled'] if temp_scaled else model_evaluations[key_dict]['softmax']
    # # model_evaluations['train']['softmax_scaled'] if temp_scaled else model_evaluations['train']['softmax']
    # confid = model_evaluations[key_dict]['confid']
    # correct = model_evaluations[key_dict]['correct']
    # residuals = 1-correct

    # confids= {
    #             'MSR' :         scores_funcs.maximum_softmax_response(softmax),
    #             'PE' :          scores_funcs.predictive_entropy(softmax),
    #             'MLS' :         scores_funcs.maximum_logit_score(logits),
    #             'PCE' :         scores_funcs.predictive_collision_entropy(softmax),
    #             'GE' :          scores_funcs.guessing_entropy(softmax),
    #             'Energy' :      scores_funcs.energy(logits, T=score_methods['temperature_scale'].temperature),
    #             'Maha':         score_methods['maha_distance'].get_scores(encoded),
    #             'ViM':          score_methods['vim_score'].get_scores(encoded),
    #             'Residual':     score_methods['residual_score'].get_scores(encoded),    
    #             'NeCo':         score_methods['neco_score'].get_scores(encoded),
    #             'pNML':         score_methods['pnml_score'].get_scores(encoded),
    #             'KLMatching':   score_methods['klmatching_score'].get_scores(softmax),
    #             'GEN' :         score_methods['generalized_entropy'].get_scores(softmax),
    #             'REN' :         score_methods['renyi_entropy'].get_scores(softmax),
    #             'TEN' :         score_methods['tsallis_entropy'].get_scores(softmax),    
    #             'GradNorm' :    gradnorm_score.get_scores(encoded, use_cuda=True),    
    #             'Confidence' :  confid,
    # }
    # stats = {
    #             key:[RiskCoverageStats(confids=confids[key], residuals=residuals), metrics.StatsCache(confids[key],correct,n_bins) ] for key in confids
    #         }
    # stats_df = pd.DataFrame( {  
    #                             'AUGRC': { k:stats[k][0].augrc for k in stats },
    #                             'AURC': { k:stats[k][0].aurc for k in stats },
    #                             'AUROC_f': { k:metrics.failauc(stats[k][1]) for k in stats },
    #                             'FPR95': { k:metrics.fpr_at_95_tpr(stats[k][1]) for k in stats },
    #                             'ECE': { k:metrics.expected_calibration_error(stats[k][1]) for k in stats },
    #                             'MCE': { k:metrics.maximum_calibration_error(stats[k][1]) for k in stats },
    #                         } )
    # filename = f'stats_{model_name}_{eval_name}.csv'
    # if os.path.exists(f'{cf.exp.dir}/analysis'):
    #     path = f'{cf.exp.dir}/analysis/{filename}'
    # else:
    #     os.mkdir(f'{cf.exp.dir}/analysis')
    #     path = f'{cf.exp.dir}/analysis/{filename}'
    # # mcd_stats_df.to_csv(path)
    # stats_df.sort_values(by=['AUGRC']).to_csv(path)
