import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from loguru import logger
from src.utils_stats import *
from src.scores_retrieval_utils import (
    read_results_vit,
    read_results,
    select_best_hyperparameters,
    select_best_scores,
    set_name_dict,
    methods_dict
)

# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')

def main():
    parser = argparse.ArgumentParser(description="Retrieve scores from experiment results.")
    parser.add_argument("--dataset", 
                        type=str, 
                        required=True, 
                        help="Name of the dataset (e.g., cifar10, supercifar, cifar100)",
                        choices=['cifar10','supercifar100','cifar100','tinyimagenet'])
    parser.add_argument("--vit", action="store_true", help="Set this flag if using ViT model results")
    parser.add_argument("--scores-dir", type=str, default="scores_final", help="Directory to save final scores")
    
    args = parser.parse_args()
    
    logger.info(f"Starting score retrieval with arguments: {args}")
    
    dataset = args.dataset
    vit = args.vit
    
    if dataset == 'cifar100' and vit:
        dataset = 'cifar100_modelvit_bbvit_lr0.01'
    elif dataset == 'supercifar100' and vit:
        dataset = 'super_cifar100'
    elif dataset == 'supercifar100' and not vit:
        dataset = 'supercifar'
    elif dataset == 'tinyimagenet':
        dataset = 'tiny-imagenet-200'
    
    # Logic from original script adapted to use args
    ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
    c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
    dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
                'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100', 'super_cifar10':'supercifar10'} # added super_cifar10 just in case as it wasn't in original dict but might be needed given dataset naming patterns, or maybe 'supercifar' covers it. kept original keys mostly.

    if vit:
        results_val = read_results_vit(dataset,'iid_val')
        results_test_iid = read_results_vit(dataset,'iid_test')
        results_ood_1 = read_results_vit(dataset,c100_condition)
        results_ood_2 = read_results_vit(dataset,'ood_nsncs_svhn')
        results_ood_3 = read_results_vit(dataset,ti_condition)
        results_ood_4 = read_results_vit(dataset,'ood_nsncs_lsun_cropped')
        results_ood_5 = read_results_vit(dataset,'ood_nsncs_lsun_resize')
        results_ood_6 = read_results_vit(dataset,'ood_nsncs_isun')
        results_ood_7 = read_results_vit(dataset,'ood_nsncs_textures')
        results_ood_8 = read_results_vit(dataset,'ood_nsncs_places365')
    else:
        results_val = read_results(dataset,'iid_val')
        results_test_iid = read_results(dataset,'iid_test')
        results_ood_1 = read_results(dataset,c100_condition)
        results_ood_2 = read_results(dataset,'ood_nsncs_svhn')
        results_ood_3 = read_results(dataset,ti_condition)
        results_ood_4 = read_results(dataset,'ood_nsncs_lsun_cropped')
        results_ood_5 = read_results(dataset,'ood_nsncs_lsun_resize')
        results_ood_6 = read_results(dataset,'ood_nsncs_isun')
        results_ood_7 = read_results(dataset,'ood_nsncs_textures')
        results_ood_8 = read_results(dataset,'ood_nsncs_places365')

    model_opts=['RW0','RF0','ASHNone']
    best_hyperparameters_list = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.scores_dir):
        logger.info(f"Creating output directory: {args.scores_dir}")
        os.makedirs(args.scores_dir, exist_ok=True)
    else:
        logger.debug(f"Output directory exists: {args.scores_dir}")

    scores_to_process = ['AUGRC','AURC','AUROC_f','FPR@95TPR','ECE','MCE']
    logger.info(f"Processing scores: {scores_to_process}")

    for score_name in scores_to_process:
        logger.debug(f"Processing score: {score_name}")
        cond_1 = results_val['metrics']=='MCD-ECTM_class_avg'
        cond_2 = results_val['metrics']=='MCD-ECTM_class_pred'
        cond_3 = results_val['metrics']=='MCD-ECTM_global'
        # Filter warnings: boolean series combined with bitwise operators on the original df
        # Re-applying logic carefully to match original pandas filtering
        results_val_filtered = results_val[~(cond_1|cond_2|cond_3) ].copy()

        test_iid_scores_mean_list, test_iid_scores_std_list = [], []
        test_ood_1_scores_mean_list, test_ood_1_scores_std_list = [], []
        test_ood_2_scores_mean_list, test_ood_2_scores_std_list = [], []
        test_ood_3_scores_mean_list, test_ood_3_scores_std_list = [], []
        test_ood_4_scores_mean_list, test_ood_4_scores_std_list = [], []
        test_ood_5_scores_mean_list, test_ood_5_scores_std_list = [], []
        test_ood_6_scores_mean_list, test_ood_6_scores_std_list = [], []
        test_ood_7_scores_mean_list, test_ood_7_scores_std_list = [], []
        test_ood_8_scores_mean_list, test_ood_8_scores_std_list = [], []
        
        model_list =['modelvit'] if vit else ['confidnet','devries','dg',]
        mcd_metrics_list = [False] if vit else [True, False] 
        
        for model_name in model_list:
            for mcd_metrics in mcd_metrics_list:
                best_hyperparameters_scores = select_best_hyperparameters(results_val_filtered, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                best_hyperparameters_list.append({'dataset':dataset, 'model':model_name,'mcd_metrics':mcd_metrics,'score_name':score_name,'best_hyperparameters_scores':best_hyperparameters_scores})
                
                scores_test_iid_mean, scores_test_iid_std = select_best_scores(results_test_iid, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_iid_scores_mean_list.append(scores_test_iid_mean)
                test_iid_scores_std_list.append(scores_test_iid_std)
                
                scores_test_ood_1_mean, scores_test_ood_1_std = select_best_scores(results_ood_1, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_1_scores_mean_list.append(scores_test_ood_1_mean)
                test_ood_1_scores_std_list.append(scores_test_ood_1_std)
                
                scores_test_ood_2_mean, scores_test_ood_2_std = select_best_scores(results_ood_2, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_2_scores_mean_list.append(scores_test_ood_2_mean)
                test_ood_2_scores_std_list.append(scores_test_ood_2_std)
                
                scores_test_ood_3_mean, scores_test_ood_3_std  = select_best_scores(results_ood_3, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_3_scores_mean_list.append(scores_test_ood_3_mean)
                test_ood_3_scores_std_list.append(scores_test_ood_3_std)
                
                scores_test_ood_4_mean, scores_test_ood_4_std  = select_best_scores(results_ood_4, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_4_scores_mean_list.append(scores_test_ood_4_mean)
                test_ood_4_scores_std_list.append(scores_test_ood_4_std)
                
                scores_test_ood_5_mean, scores_test_ood_5_std  = select_best_scores(results_ood_5, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_5_scores_mean_list.append(scores_test_ood_5_mean)
                test_ood_5_scores_std_list.append(scores_test_ood_5_std)
                
                scores_test_ood_6_mean, scores_test_ood_6_std  = select_best_scores(results_ood_6, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_6_scores_mean_list.append(scores_test_ood_6_mean)
                test_ood_6_scores_std_list.append(scores_test_ood_6_std)
                
                scores_test_ood_7_mean, scores_test_ood_7_std  = select_best_scores(results_ood_7, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_7_scores_mean_list.append(scores_test_ood_7_mean)
                test_ood_7_scores_std_list.append(scores_test_ood_7_std)
                
                scores_test_ood_8_mean, scores_test_ood_8_std  = select_best_scores(results_ood_8, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
                test_ood_8_scores_mean_list.append(scores_test_ood_8_mean)
                test_ood_8_scores_std_list.append(scores_test_ood_8_std)

        ood_nsncs_dict= {
                            'iid_nsncs_test':test_iid_scores_mean_list,
                            c100_condition:test_ood_1_scores_mean_list,
                            ti_condition:test_ood_3_scores_mean_list,
                            'ood_nsncs_lsun_resize':test_ood_5_scores_mean_list,
                            'ood_nsncs_isun':test_ood_6_scores_mean_list,
                            'ood_nsncs_places365':test_ood_8_scores_mean_list,
                            'ood_nsncs_lsun_cropped':test_ood_4_scores_mean_list,
                            'ood_nsncs_textures':test_ood_7_scores_mean_list,
                            'ood_nsncs_svhn':test_ood_2_scores_mean_list,
                        }

        augrc_list = [pd.concat(ood_nsncs_dict[data_key])[f'{score_name}'].to_frame().rename({f'{score_name}':'_'.join(data_key.split('_')[2:])},axis='columns') for data_key in ood_nsncs_dict]
        augrc_scores_ood = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), augrc_list)
        augrc_scores_ood = augrc_scores_ood.droplevel(level=['network','RankWeight','RankFeat','ASH'])
        augrc_scores_ood = augrc_scores_ood.reset_index()
        augrc_scores_ood = augrc_scores_ood.rename(set_name_dict,axis='columns')
        augrc_scores_ood['methods'] = augrc_scores_ood['methods'].replace(methods_dict)
        augrc_scores_ood['methods'] = augrc_scores_ood['methods'].str.replace('_',' ')
        augrc_scores_ood.columns = augrc_scores_ood.columns.str.replace('_',' ')
        augrc_scores_ood = augrc_scores_ood.set_index(['model','drop out','methods','reward'])

        methods = augrc_scores_ood.index.get_level_values('methods')
        mcd_methods_mask = np.array(['MCD-' in x for x in  methods])
        label_model = 'ViT' if vit else 'Conv'
        
        # Save results
        out_path_false = os.path.join(args.scores_dir, f'scores_{score_name}_MCD-False_{label_model}_{dict_clip.get(dataset, dataset)}.csv')
        out_path_true = os.path.join(args.scores_dir, f'scores_{score_name}_MCD-True_{label_model}_{dict_clip.get(dataset, dataset)}.csv')
        
        logger.info(f"Saving scores to {out_path_false}")
        augrc_scores_ood[~mcd_methods_mask].to_csv(out_path_false)
        logger.info(f"Saving scores to {out_path_true}")
        augrc_scores_ood[mcd_methods_mask].to_csv(out_path_true)

        ood_nsncs_dict_all = {
                                'iid_nsncs_test':test_iid_scores_std_list,
                                c100_condition:test_ood_1_scores_std_list,
                                ti_condition:test_ood_3_scores_std_list,
                                'ood_nsncs_lsun_resize':test_ood_5_scores_std_list,
                                'ood_nsncs_isun':test_ood_6_scores_std_list,
                                'ood_nsncs_places365':test_ood_8_scores_std_list,
                                'ood_nsncs_lsun_cropped':test_ood_4_scores_std_list,
                                'ood_nsncs_textures':test_ood_7_scores_std_list,
                                'ood_nsncs_svhn':test_ood_2_scores_std_list,
                            }

        metric_score_list = [pd.concat(ood_nsncs_dict_all[data_key])[f'{score_name}'].to_frame().rename({f'{score_name}':'_'.join(data_key.split('_')[2:])},axis='columns') for data_key in ood_nsncs_dict_all]
        metric_scores_ood = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), metric_score_list)
        metric_scores_ood = metric_scores_ood.droplevel(level=['network','RankWeight','RankFeat','ASH'])
        metric_scores_ood = metric_scores_ood.reset_index()
        metric_scores_ood = metric_scores_ood.rename(set_name_dict,axis='columns')
        metric_scores_ood['methods'] = metric_scores_ood['methods'].replace(methods_dict)
        metric_scores_ood['methods'] = metric_scores_ood['methods'].str.replace('_',' ')
        metric_scores_ood.columns = metric_scores_ood.columns.str.replace('_',' ')
        metric_scores_ood = metric_scores_ood.set_index(['model','drop out','methods','reward','run'])

        methods = metric_scores_ood.index.get_level_values('methods')
        mcd_methods_mask = np.array(['MCD-' in x for x in  methods])
        
        # Save all scores
        out_path_all_false = os.path.join(args.scores_dir, f'scores_all_{score_name}_MCD-False_{label_model}_{dict_clip.get(dataset, dataset)}.csv')
        out_path_all_true = os.path.join(args.scores_dir, f'scores_all_{score_name}_MCD-True_{label_model}_{dict_clip.get(dataset, dataset)}.csv')

        logger.info(f"Saving all scores to {out_path_all_false}")
        metric_scores_ood[~mcd_methods_mask].to_csv(out_path_all_false)
        logger.info(f"Saving all scores to {out_path_all_true}")
        metric_scores_ood[mcd_methods_mask].to_csv(out_path_all_true)

if __name__ == "__main__":
    main()