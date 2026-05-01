#%%
import pandas as pd
import numpy as np
from loguru import logger
import os
import argparse
from functools import reduce
from src.scores_retrieval_utils import (
    set_name_dict,
    methods_dict
)
#%%
# MCD_flag = False
# BACKBONE = 'Conv'
# SOURCE = 'cifar10'
# df = pd.read_csv(f"scores_final/scores_all_ECE_L2_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv")
#%%
def select_best_hyperparameters(results_val, model_name, score_name, mcd_metrics:bool=False):
    dropout_list = ['do1'] if mcd_metrics else ['do0','do1']
    metrics_flag = 'do1' if mcd_metrics else 'do0'
    best_metrics_list = []
    for dropout_opt in dropout_list:
        cond = (results_val['model']==model_name) & (results_val['drop_out']==dropout_opt)
        results_grouped_val = results_val[cond].groupby(['model','drop_out','reward','calibration','method']).mean().sort_values(by=score_name)
        metrics_name = results_val[results_val['drop_out']==metrics_flag]['method'].unique()
        if mcd_metrics:
            metrics_name = [m for m in metrics_name if 'MCD-' in m ]
        for metric in metrics_name:
            if score_name=='AUROC_f':
                idx_best_metric = results_grouped_val.loc[(model_name,slice(None),slice(None),slice(None),metric)][score_name].idxmax()+(f'{metric}',)
                # best_metrics_list.append(results_grouped_val.loc[(f'{model_name}',)+idx_best_metric].to_frame().T)            
            else:
                idx_best_metric = results_grouped_val.loc[(model_name,slice(None),slice(None),slice(None),metric)][score_name].idxmin()+(f'{metric}',)
            best_metrics_list.append(results_grouped_val.loc[(f'{model_name}',)+idx_best_metric].to_frame().T)
    best_score_per_metric_df = pd.concat(best_metrics_list)
    best_scores_list = []
    for metric in metrics_name:
        if score_name=='AUROC_f':
            idx_best = best_score_per_metric_df.loc[(model_name,slice(None),slice(None),slice(None),metric)][score_name].idxmax()
        else:
            idx_best = best_score_per_metric_df.loc[(model_name,slice(None),slice(None),slice(None),metric)][score_name].idxmin()
        best_scores_list.append(best_score_per_metric_df.loc[(f'{model_name}',)+idx_best+(f'{metric}',)].to_frame().T)
    scores_per_model = pd.concat(best_scores_list).sort_values(by=[score_name])
    return scores_per_model
#%%
def select_best_scores(results_test, best_hyperparameters, model_name, score_name, mcd_metrics:bool=False,):
    dropout_list = ['do1'] if mcd_metrics else ['do0','do1']
    best_metrics_mean_list = []
    best_metrics_list = []
    for dropout_opt in dropout_list:
        cond = (results_test['model']==model_name) & (results_test['drop_out']==dropout_opt)
        results_grouped_test_mean = results_test[cond].groupby(['model','drop_out','reward','calibration','method']).mean()
        best_metrics_mean_list.append(results_grouped_test_mean)
        results_grouped_test = results_test[cond]
        best_metrics_list.append(results_grouped_test)
    scores_test_mean_df = pd.concat(best_metrics_mean_list).loc[best_hyperparameters.index].sort_values(by=score_name)
    scores_test_df = pd.concat(best_metrics_list).set_index(['model','drop_out','reward','calibration','method']).loc[scores_test_mean_df.index]
    scores_test_df = scores_test_df.set_index([scores_test_df.index,'run'])
    return scores_test_mean_df, scores_test_df

#%%
def main():
    parser = argparse.ArgumentParser(description="Retrieve scores from experiment results.")
    parser.add_argument("--dataset", 
                        type=str, 
                        required=True, 
                        help="Name of the dataset (e.g., cifar10, supercifar, cifar100)",
                        choices=['cifar10','supercifar100','cifar100','tinyimagenet'],)
    parser.add_argument("--vit", action="store_true", help="Set this flag if using ViT model results")
    parser.add_argument("--scores-dir", type=str, default="scores_final", help="Directory to save final scores")
    parser.add_argument("--scores-path", type=str, default="scores_final", help="Directory to find scores csv files for processing")

    args = parser.parse_args()
    # args = parser.parse_args(['--dataset','cifar10'])

    logger.info(f"Starting score retrieval with arguments: {args}")

    dataset = args.dataset
    vit = args.vit

    # Logic from original script adapted to use args
    ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
    c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
    dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 
                'supercifar100':'supercifar100','supercifar':'supercifar100',
                'tiny-imagenet-200':'tinyimagenet', 'tinyimagenet':'tinyimagenet',
                'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100', 'super_cifar10':'supercifar10'} 
    # added super_cifar10 just in case as it wasn't in original dict but might be needed given dataset naming patterns, or maybe 'supercifar' covers it. kept original keys mostly.

    vit_condition = 'ViT' if vit else 'Conv'
    load_path = os.path.join(args.scores_path, f'calibration_results_{dataset}_{vit_condition}.csv')
    results_df = pd.read_csv(load_path)
    # Filter to VGG13 results only for Conv backbone, as per original script logic 
    if vit_condition == 'Conv':
        logger.info("Filtering results to VGG13 architecture for Conv backbone.")
        results_df = results_df[results_df['architecture']=='VGG13'].copy()
        results_df = results_df.drop(columns=['architecture'])
    else:
        results_df = results_df.drop(columns=['architecture'])
    #
    best_hyperparameters_list = []

    # Create output directory if it doesn't exist
    if not os.path.exists(args.scores_dir):
        logger.info(f"Creating output directory: {args.scores_dir}")
        os.makedirs(args.scores_dir, exist_ok=True)
    else:
        logger.debug(f"Output directory exists: {args.scores_dir}")

    scores_to_process = ['ece_l1','ece_l2','mce','ece_l1_bound','ece_l2_bound']
    logger.info(f"Processing scores: {scores_to_process}")
    model_list =['modelvit'] if vit else ['confidnet','devries','dg',]
    mcd_metrics_list = [False] if vit else [True, False]
    ood_sets = [
        'iid_test', c100_condition, 'ood_nsncs_svhn', ti_condition,
        'ood_nsncs_lsun_cropped','ood_nsncs_lsun_resize','ood_nsncs_isun',
        'ood_nsncs_textures','ood_nsncs_places365'
    ]

    for score_name in scores_to_process:
        logger.debug(f"Processing score: {score_name}")
        results_val = results_df[results_df['dataset']=='val'].drop('dataset',axis=1)
        cond_1 = results_val['method']=='MCD-ECTM_class_avg'
        cond_2 = results_val['method']=='MCD-ECTM_class_pred'
        cond_3 = results_val['method']=='MCD-ECTM_global'
        # Filter warnings: boolean series combined with bitwise operators on the original df
        # Re-applying logic carefully to match original pandas filtering
        results_val_filtered = results_val[~(cond_1|cond_2|cond_3) ].copy()
        results_ood_avg = {ood:[] for ood in ood_sets}
        results_ood_full = {ood:[] for ood in ood_sets}
        
        for model_name in model_list:
            for mcd_metrics in mcd_metrics_list:
                best_hyperparameters_scores = select_best_hyperparameters(results_val_filtered, model_name, score_name, mcd_metrics=mcd_metrics)
                best_hyperparameters_list.append({'dataset':dataset, 'model':model_name,'mcd_metrics':mcd_metrics,'score_name':score_name,'best_hyperparameters_scores':best_hyperparameters_scores})
                for ood in ood_sets:
                    results_ood_df = results_df[results_df['dataset']==ood].drop('dataset',axis=1)
                    selected_scores_mean, selected_scores_full = select_best_scores(results_ood_df, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics)
                    results_ood_avg[ood].append(selected_scores_mean)
                    results_ood_full[ood].append(selected_scores_full)
        for i, results_ood_best in enumerate([results_ood_avg, results_ood_full]):
            logger.debug(f"Combining scores for {'average' if i==0 else 'full'} results.")
            # logger.debug(f"Score name: {score_name}, model: {model_name}, mcd_metrics: {mcd_metrics}")
            score_list = [pd.concat(results_ood_best[data_key])[f'{score_name}'].to_frame().rename({f'{score_name}':'_'.join(data_key.split('_')[2:] if data_key.startswith('ood_') else data_key.split('_')[1:])},axis='columns') for data_key in results_ood_best]
            scores_combined = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), score_list)
            scores_combined = scores_combined.reset_index()
            scores_combined = scores_combined.rename(set_name_dict,axis='columns')
            scores_combined['method'] = scores_combined['method'].replace(methods_dict)
            scores_combined['method'] = scores_combined['method'].str.replace('_',' ')
            #
            scores_combined = scores_combined.rename({'method':'methods'},axis='columns')
            scores_combined = scores_combined.drop(columns=['calibration'])
            # Do. not include modified scores for now
            # keep_exceptions = {"KPCA RecError global", "PCA RecError global", "MCD-KPCA RecError global", "MCD-PCA RecError global"}

            # mask_keep = (
            #             scores_combined["methods"].isin(keep_exceptions)
            #             | ~scores_combined["methods"].str.contains(r"\bclass\b|\bglobal\b", case=False, regex=True, na=False)
            #             ) 
            # scores_combined = scores_combined[mask_keep]
            
            scores_combined.columns = scores_combined.columns.str.replace('_',' ')
            scores_combined = scores_combined.set_index(['model','drop out','methods','reward',])
            #
            if score_name=='ece_l2':
                scores_combined = np.sqrt(scores_combined)
            #

            methods = scores_combined.index.get_level_values('methods')
            mcd_methods_mask = np.array(['MCD-' in x for x in  methods])
            if i==0:
                path_mcd_false = f'scores_{score_name.upper()}_MCD-False_{vit_condition}_{dict_clip.get(dataset, dataset)}.csv'
                path_mcd_true = f'scores_{score_name.upper()}_MCD-True_{vit_condition}_{dict_clip.get(dataset, dataset)}.csv'
            else:
                path_mcd_false = f'scores_all_{score_name.upper()}_MCD-False_{vit_condition}_{dict_clip.get(dataset, dataset)}.csv'
                path_mcd_true = f'scores_all_{score_name.upper()}_MCD-True_{vit_condition}_{dict_clip.get(dataset, dataset)}.csv'
            # Save results
            out_path_false = os.path.join(args.scores_dir, path_mcd_false)
            out_path_true = os.path.join(args.scores_dir, path_mcd_true)

            

            logger.info(f"Saving scores to {out_path_false}")
            scores_combined[~mcd_methods_mask].to_csv(out_path_false)
            logger.info(f"Saving scores to {out_path_true}")
            scores_combined[mcd_methods_mask].to_csv(out_path_true)

    # keep_exceptions = {"KPCA RecError global", "PCA RecError global", "MCD-KPCA RecError global", "MCD-PCA RecError global"}
    hyperparameters_df = pd.DataFrame(best_hyperparameters_list)
    for mcd in hyperparameters_df['mcd_metrics'].unique():
        hyperparameters_per_mcd = []
        names_list = []
        for score in hyperparameters_df['score_name'].unique():
            subset = hyperparameters_df[(hyperparameters_df['mcd_metrics'] == mcd) & (hyperparameters_df['score_name'] == score)]
            for _, row in subset.iterrows():
                hyperparameters = row['best_hyperparameters_scores'].copy()
                hyperparameters.index.names = ['model','drop_out','reward','calibration','methods']
                hyperparameters = hyperparameters.reset_index()
                hyperparameters['methods'] = hyperparameters['methods'].replace(methods_dict)
                hyperparameters['methods'] = hyperparameters['methods'].str.replace('_',' ')
                # mask_keep = (
                #             hyperparameters["methods"].isin(keep_exceptions)
                #             | ~hyperparameters["methods"].str.contains(r"\bclass\b|\bglobal\b", case=False, regex=True, na=False)
                #             )
                # hyperparameters = hyperparameters[mask_keep]
                hyperparameters = hyperparameters.set_index('methods')[['drop_out','reward','calibration']]
                hyperparameters['reward'] = hyperparameters['reward'].apply(lambda x: float(x.split('rew')[1]))
                if row['model']!='dg':
                    hyperparameters = hyperparameters.drop(columns=['reward'])
                hyperparameters_per_mcd.append(hyperparameters)
                names_list.append((f"{score}",f"{row['model']}"))

        out_path_hyperparameters = os.path.join(args.scores_dir, f'hyperparameters_results_MCD-{mcd}_{vit_condition}_{dataset}.csv')
        compiled_hyperparameters_df = pd.concat(hyperparameters_per_mcd, keys=names_list, axis=1)
        logger.info(f"Saving hyperparameters results to {out_path_hyperparameters}")
        compiled_hyperparameters_df.to_csv(out_path_hyperparameters)

if __name__ == "__main__":
    main()
# %%
# keep_exceptions = {"KPCA RecError global", "PCA RecError global", "MCD-KPCA RecError global", "MCD-PCA RecError global"}
# hyperparameters_df = pd.DataFrame(best_hyperparameters_list)
# for mcd in hyperparameters_df['mcd_metrics'].unique():
#     hyperparameters_per_mcd = []
#     names_list = []
#     for score in hyperparameters_df['score_name'].unique():
#         subset = hyperparameters_df[(hyperparameters_df['mcd_metrics'] == mcd) & (hyperparameters_df['score_name'] == score)]
#         for _, row in subset.iterrows():
#             hyperparameters = row['best_hyperparameters_scores'].copy()
#             hyperparameters.index.names = ['model','drop_out','reward','calibration','methods']
#             hyperparameters = hyperparameters.reset_index()
#             hyperparameters['methods'] = hyperparameters['methods'].replace(methods_dict)
#             hyperparameters['methods'] = hyperparameters['methods'].str.replace('_',' ')
#             mask_keep = (
#                         hyperparameters["methods"].isin(keep_exceptions)
#                         | ~hyperparameters["methods"].str.contains(r"\bclass\b|\bglobal\b", case=False, regex=True, na=False)
#                         )
#             hyperparameters = hyperparameters[mask_keep]
#             hyperparameters = hyperparameters.set_index('methods')[['drop_out','reward','calibration']]
#             hyperparameters['reward'] = hyperparameters['reward'].apply(lambda x: float(x.split('rew')[1]))
#             if row['model']!='dg':
#                 hyperparameters = hyperparameters.drop(columns=['reward'])
#             hyperparameters_per_mcd.append(hyperparameters)
#             names_list.append((f"{score}",f"{row['model']}"))

#%%
# b = hyperparameters_df[(hyperparameters_df['score_name'] == 'ece_l2_bound')&(hyperparameters_df['mcd_metrics'] == False)]['best_hyperparameters_scores'].iloc[0]
# b.index.names = ['model','drop_out','reward','calibration','methods']
# b = b.reset_index()
# # b = b.rename(set_name_dict,axis='columns')
# b['methods'] = b['methods'].replace(methods_dict)
# b['methods'] = b['methods'].str.replace('_',' ')
#
# %%
# keep_exceptions = {"KPCA RecError global", "PCA RecError global", "MCD-KPCA RecError global", "MCD-PCA RecError global"}

# mask_keep = (
#             b["methods"].isin(keep_exceptions)
#             | ~b["methods"].str.contains(r"\bclass\b|\bglobal\b", case=False, regex=True, na=False)
#             ) 
# %%
# b = b[mask_keep]
# %%
