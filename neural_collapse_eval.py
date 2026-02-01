#%%
import os
# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')
#%%
import torch
from fd_shifts.utils import exp_utils
from fd_shifts.models import get_model
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
import argparse
import pandas as pd
from torch.nn import functional as F
from src import utils
from src import utils_funcs 
from src import scores_methods
from fd_shifts import logger
import numpy as np
from src.utils_stats import *
from src.utils import *
#%%
# If paths exists, then read the metrics from csv files
df_metrics = pd.read_csv('neural_collapse_metrics/nc_metrics.csv',index_col=0)
df_metrics_global = pd.read_csv('neural_collapse_metrics/nc_metrics_global.csv',index_col=0)
df_metrics_class_pred = pd.read_csv('neural_collapse_metrics/nc_metrics_class_pred.csv',index_col=0)
#%%
# Otherwise, compute the metrics and save to csv files
nc_metrics_list = []
nc_metrics_global_list = []
nc_metrics_class_pred_list = []
for dataset in ['cifar10','supercifar','cifar100','tinyimagenet']:
    with open(f'configs_exp/configs_{dataset}_iid_train.txt', 'r') as f:
        for k, cfg_line in enumerate(f):
            # if (k>0 and k<=60) or (k>361):
            cfg_line_split = cfg_line.split().copy()
            print(cfg_line_split)
            path = cfg_line_split[1]
            rank_weight_opt = False if 'no' in cfg_line_split[2] else True
            rank_feat_opt = False if 'no' in cfg_line_split[3] else True
            ash_method_opt = cfg_line_split[4]# 'ash_s@90'
            use_cuda_opt = False if 'no' in cfg_line_split[5] else True
            temperature_scale_opt = False if 'no' in cfg_line_split[6] else True
            # if 'do1' in path:
            #     continue
            if rank_weight_opt==True:
                continue
            elif rank_feat_opt==True:
                continue
            elif ash_method_opt!='None':
                continue
            else:
                cuda_available = torch.cuda.is_available()
                if cuda_available and use_cuda_opt:
                    print("Cuda available...")
                else:
                    use_cuda_opt = False
                    print("Cuda not available...")
                # Change string option None to None type 
                if ash_method_opt=='None':
                    ash_method_opt = None
                # Load study configurations
                study_name = utils.get_study_name(path)
                do_enabled = utils.is_dropout_enabled(path)
                cf = utils.get_conf(path, study_name)
                ckpt_path = exp_utils._get_path_to_best_ckpt(
                                cf.exp.dir, 'last', cf.test.selection_mode )
                if 'super' in path:
                    cf.eval.query_studies.noise_study = ['corrupt_cifar100']
                    cf.eval.query_studies.new_class_study = ['cifar10', 'svhn', 'tinyimagenet_resize']
                    if do_enabled:
                        cf.model.avg_pool = False
                if 'vit' in path:
                    cf.data.num_workers = 12
                # Load module
                module = get_model(cf.model.name)(cf) 
                module.load_only_state_dict(ckpt_path, device='cpu')
                if study_name == 'confidnet':
                    module.backbone.encoder.disable_dropout()
                    module.network.encoder.disable_dropout()
                elif (study_name == 'devries') or (study_name == 'dg'):
                    # model = module.model
                    module.model.encoder.disable_dropout()
                elif study_name == 'vit':
                    module.disable_dropout()
                else:
                    raise NotImplementedError
                # 
                if do_enabled and use_cuda_opt:
                    if (study_name=='devries' or study_name=='dg'):
                        new_batch_size = cf.trainer.batch_size//2
                        logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
                        cf.trainer.batch_size = new_batch_size
                    elif (study_name=='confidnet'):
                        new_batch_size = cf.trainer.batch_size//4
                        logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
                        cf.trainer.batch_size = new_batch_size

                if study_name=='vit' and use_cuda_opt and not do_enabled:
                    new_batch_size = cf.trainer.batch_size//2
                    logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
                    cf.trainer.batch_size = new_batch_size
                if study_name=='vit' and use_cuda_opt and do_enabled:
                    new_batch_size = cf.trainer.batch_size//2
                    confids_test = cf.eval.confidence_measures.test
                    no_mcd_confid_test = [i for i in confids_test if 'mcd' not in i]
                    # do_enabled = False
                    cf.eval.confidence_measures.test = no_mcd_confid_test 
                    logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
                    cf.trainer.batch_size = new_batch_size

                if study_name=='vit' and not use_cuda_opt:
                    new_batch_size = 128
                    logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
                    cf.trainer.batch_size = new_batch_size

                # Load datasets
                datamodule = FDShiftsDataLoader(cf)
                datamodule.setup()
                # Instantiate model with added functionality
                model = scores_methods.TrainedModule(module, study_name, cf, 
                                                    rank_weight=rank_weight_opt, 
                                                    rank_feat=rank_feat_opt, 
                                                    ash_method=ash_method_opt, 
                                                    use_cuda=use_cuda_opt)
                # Compute evaluations
                model_opts = f'_RW{int(rank_weight_opt)}_RF{int(rank_feat_opt)}_ASH{str(ash_method_opt)}'
                model_cfgs = {'dataset': dataset,
                                'architecture' : 'VGG13' if study_name!='vit' else 'ViT',
                                'study': study_name,
                                'dropout': do_enabled,
                                'run': int(extract_char_after_substring(path,'run')), 
                                'reward': float(extract_char_after_substring(path,'rew')), 
                                'lr': extract_char_after_substring(path,'lr'),}
                neural_collapse = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
                neural_collapse.load_params(filename='NeuralCollapse_params'+model_opts)
                nc_metrics = {k:neural_collapse.nc_metrics[k].item() for k in  neural_collapse.nc_metrics}
                nc_metrics.update(model_cfgs)
                nc_metrics_list.append(nc_metrics)
                #
                neural_collapse_global = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
                neural_collapse_global.load_params(filename='NeuralCollapse_global_params'+model_opts)
                nc_metrics_global = {k:neural_collapse_global.nc_metrics[k].item() for k in  neural_collapse_global.nc_metrics}
                nc_metrics_global.update(model_cfgs)
                nc_metrics_global_list.append(nc_metrics_global)
                #
                neural_collapse_class_pred = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
                neural_collapse_class_pred.load_params(filename='NeuralCollapse_class_pred_params'+model_opts)
                nc_metrics_class_pred = {k:neural_collapse_class_pred.nc_metrics[k].item() for k in  neural_collapse_class_pred.nc_metrics}
                nc_metrics_class_pred.update(model_cfgs)
                nc_metrics_class_pred_list.append(nc_metrics_class_pred)
        # break
            # break
#%%
df_metrics = pd.DataFrame(nc_metrics_list)
df_metrics_global = pd.DataFrame(nc_metrics_global_list)
df_metrics_class_pred = pd.DataFrame(nc_metrics_class_pred_list)

df_metrics.to_csv('neural_collapse_metrics/nc_metrics.csv')
df_metrics_global.to_csv('neural_collapse_metrics/nc_metrics_global.csv')
df_metrics_class_pred.to_csv('neural_collapse_metrics/nc_metrics_class_pred.csv')


#%%
drop_cols = ['lr','reward','run','study']
means_cols = ['equiangular_uc','equinorm_uc','max_equiangular_uc','self_duality','var_collapse','cdnv_score','bias_collapse','M_etf_diff','wM_etf_diff']
weights_cols = ['equiangular_wc','equinorm_wc','max_equiangular_wc','w_etf_diff']
group_list = ['dropout','architecture','dataset']
df_metrics_weights_grouped = df_metrics.drop(drop_cols,axis=1)[weights_cols+group_list].groupby(group_list)
df_metrics_grouped = df_metrics.drop(drop_cols,axis=1)[means_cols+group_list].groupby(group_list)
df_metrics_global_grouped = df_metrics_global.drop(drop_cols,axis=1)[means_cols+group_list].groupby(group_list)
df_metrics_class_pred_grouped = df_metrics_class_pred.drop(drop_cols,axis=1)[means_cols+group_list].groupby(group_list)

#%%
df_metrics_weights_grouped.mean().to_csv('neural_collapse_metrics/nc_metrics_weights_grouped.csv')
df_metrics_grouped.mean().to_csv('neural_collapse_metrics/nc_metrics_grouped.csv')
df_metrics_global_grouped.mean().to_csv('neural_collapse_metrics/nc_metrics_global_grouped.csv')
df_metrics_class_pred_grouped.mean().to_csv('neural_collapse_metrics/nc_metrics_class_pred_grouped.csv')   

# %%
