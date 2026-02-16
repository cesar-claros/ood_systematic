
import os
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
from loguru import logger

# Additional imports if needed by the functions
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns

def read_results_vit(dataset,set_name):
    logger.info(f"Reading ViT results for dataset: {dataset}, set: {set_name}")
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/vit/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir() if f'{dataset}_' in j.parts[-1]]
    # print(len(folders_in_exp_dir))
    if 'super' in  dataset:
        lst_idx = [2,3,7,6,8]
    else:
        lst_idx = [1,2,6,5,7]
    res = []
    
    logger.debug(f"Found {len(folders_in_exp_dir)} experiment folders.")
    
    for k in range(len(folders_in_exp_dir)):
        model_description = folders_in_exp_dir[k].parts[-1].split('_')
        files_in_folder = [j for j in folders_in_exp_dir[k].joinpath('analysis').glob(f'*stats*{set_name}.csv')]
        # print(files_in_folder)
        for i in range(len(files_in_folder)):
            exp_description = files_in_folder[i].parts[-1].split('_')
            stats_df = pd.read_csv(files_in_folder[i], index_col=0)
            stats_df[['model','network','drop_out','run','reward']] = [model_description[j] for j in lst_idx]
            stats_df[['RankWeight','RankFeat','ASH']] = [ *exp_description[1:3], exp_description[3]]
            res.append(stats_df)
            
    if not res:
        logger.warning(f"No results found for dataset: {dataset}, set: {set_name}")
        return pd.DataFrame()
        
    results = pd.concat(res).reset_index(names='metrics')
    # print(results['run'])
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    logger.success(f"Successfully read {len(results)} rows for {dataset} - {set_name}")
    return results

def read_results(dataset,set_name):
    logger.info(f"Reading results for dataset: {dataset}, set: {set_name}")
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/{dataset}_paper_sweep/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir()]
    res = []
    
    logger.debug(f"Found {len(folders_in_exp_dir)} experiment folders.")

    for k in range(len(folders_in_exp_dir)):
        model_description = folders_in_exp_dir[k].parts[-1].split('_')
        files_in_folder = [j for j in folders_in_exp_dir[k].joinpath('analysis').glob(f'*stats*{set_name}.csv')]
        for i in range(len(files_in_folder)):
            exp_description = files_in_folder[i].parts[-1].split('_')
            stats_df = pd.read_csv(files_in_folder[i], index_col=0)
            stats_df[['model','network','drop_out','run','reward']] = model_description
            stats_df[['RankWeight','RankFeat','ASH']] = [ *exp_description[1:3], exp_description[3]]
            res.append(stats_df)
            
    if not res:
        logger.warning(f"No results found for dataset: {dataset}, set: {set_name}")
        return pd.DataFrame()

    results = pd.concat(res).reset_index(names='metrics')
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    logger.success(f"Successfully read {len(results)} rows for {dataset} - {set_name}")
    return results

def select_best_hyperparameters(results_val, model_name, score_name, mcd_metrics:bool=False, model_opts=['RW0','RF0','ASHNone']):
    dropout_list = ['do1'] if mcd_metrics else ['do0','do1']
    metrics_flag = 'do1' if mcd_metrics else 'do0'
    best_metrics_list = []
    for dropout_opt in dropout_list:
        cond = (results_val['model']==model_name) & (results_val['RankWeight']==model_opts[0]) & (results_val['RankFeat']==model_opts[1]) & (results_val['ASH']==model_opts[2]) & (results_val['drop_out']==dropout_opt)
        results_grouped_val = results_val[cond].groupby(['model','network','drop_out','reward','RankWeight','RankFeat','ASH','metrics']).mean().sort_values(by=score_name)
        metrics_name = results_val[results_val['drop_out']==metrics_flag]['metrics'].unique()
        if mcd_metrics:
            metrics_name = [m for m in metrics_name if 'MCD-' in m ]
        for metric in metrics_name:
            if score_name=='AUROC_f':
                idx_best_metric = results_grouped_val.loc[(model_name,slice(None),slice(None),slice(None),slice(None),slice(None),slice(None),metric)][score_name].idxmax()+(f'{metric}',)
                # best_metrics_list.append(results_grouped_val.loc[(f'{model_name}',)+idx_best_metric].to_frame().T)            
            else:
                idx_best_metric = results_grouped_val.loc[(model_name,slice(None),slice(None),slice(None),slice(None),slice(None),slice(None),metric)][score_name].idxmin()+(f'{metric}',)
            best_metrics_list.append(results_grouped_val.loc[(f'{model_name}',)+idx_best_metric].to_frame().T)
    best_score_per_metric_df = pd.concat(best_metrics_list)
    best_scores_list = []
    for metric in metrics_name:
        if score_name=='AUROC_f':
            idx_best = best_score_per_metric_df.loc[(model_name,slice(None),slice(None),slice(None),slice(None),slice(None),slice(None),metric)][score_name].idxmax()
        else:
            idx_best = best_score_per_metric_df.loc[(model_name,slice(None),slice(None),slice(None),slice(None),slice(None),slice(None),metric)][score_name].idxmin()
        best_scores_list.append(best_score_per_metric_df.loc[(f'{model_name}',)+idx_best+(f'{metric}',)].to_frame().T)
    scores_per_model = pd.concat(best_scores_list).sort_values(by=[score_name])
    return scores_per_model

def select_best_scores(results_test, best_hyperparameters, model_name, score_name, mcd_metrics:bool=False, model_opts=['RW0','RF0','ASHNone']):
    dropout_list = ['do1'] if mcd_metrics else ['do0','do1']
    best_metrics_mean_list = []
    best_metrics_list = []
    for dropout_opt in dropout_list:
        cond = (results_test['model']==model_name) & (results_test['RankWeight']==model_opts[0]) & (results_test['RankFeat']==model_opts[1]) & (results_test['ASH']==model_opts[2]) & (results_test['drop_out']==dropout_opt)
        results_grouped_test_mean = results_test[cond].groupby(['model','network','drop_out','reward','RankWeight','RankFeat','ASH','metrics']).mean()
        best_metrics_mean_list.append(results_grouped_test_mean)
        results_grouped_test = results_test[cond]
        best_metrics_list.append(results_grouped_test)
    scores_test_mean_df = pd.concat(best_metrics_mean_list).loc[best_hyperparameters.index].sort_values(by=score_name)
    scores_test_df = pd.concat(best_metrics_list).set_index(['model','network','drop_out','reward','RankWeight','RankFeat','ASH','metrics']).loc[scores_test_mean_df.index]
    scores_test_df = scores_test_df.set_index([scores_test_df.index,'run'])
    return scores_test_mean_df, scores_test_df
