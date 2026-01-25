#%%
import os
os.environ["EXPERIMENT_ROOT_DIR"] = '/work/cniel/sw/FD_Shifts/project/experiments'
os.environ["DATASET_ROOT_DIR"] = '/work/cniel/sw/FD_Shifts/project/datasets'
#%%
import numpy as np
import pandas as pd
from pathlib import Path
from functools import reduce
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from src.utils_stats import *
#%%
# dataset = 'cifar10'
# set_name = 'iid_val'
# model_name = 'devries'
# score_name = 'AUGRC'
# mcd_metrics = True
# metrics_flag = 'do1' if mcd_metrics else 'do0'
# dropout_list = ['do1'] if mcd_metrics else ['do0','do1']
#%%
def read_results_vit(dataset,set_name):
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/vit/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir() if f'{dataset}_' in j.parts[-1]]
    # print(len(folders_in_exp_dir))
    if 'super' in  dataset:
        lst_idx = [2,3,7,6,8]
    else:
        lst_idx = [1,2,6,5,7]
    res = []
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
    results = pd.concat(res).reset_index(names='metrics')
    # print(results['run'])
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    return results
#%%
def read_results(dataset,set_name):
    experiment_dir = Path(os.environ["EXPERIMENT_ROOT_DIR"]+f'/{dataset}_paper_sweep/')
    folders_in_exp_dir = [j for j in experiment_dir.iterdir()]
    res = []
    for k in range(len(folders_in_exp_dir)):
        model_description = folders_in_exp_dir[k].parts[-1].split('_')
        files_in_folder = [j for j in folders_in_exp_dir[k].joinpath('analysis').glob(f'*stats*{set_name}.csv')]
        for i in range(len(files_in_folder)):
            exp_description = files_in_folder[i].parts[-1].split('_')
            stats_df = pd.read_csv(files_in_folder[i], index_col=0)
            stats_df[['model','network','drop_out','run','reward']] = model_description
            stats_df[['RankWeight','RankFeat','ASH']] = [ *exp_description[1:3], exp_description[3]]
            res.append(stats_df)
    results = pd.concat(res).reset_index(names='metrics')
    results['run'] = results['run'].str.split(pat='run', expand=True)[1].astype(int)
    return results
#%%
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
#%%
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
#%%
def make_pretty(styler):
    styler.set_caption(f"AUGRC scores for OOD data. Dataset: {dataset}")
    styler.format(precision=3)
    # styler.format_index(lambda v: v.strftime("%A"))
    styler.background_gradient(axis=0, cmap="coolwarm")
    return styler
#%%
def highlight_row(row, target_index, color):
    """
    Highlights a specific row with a given background color.
    """
    if row.name == target_index:
        return [f'background-color: {color}'] * len(row)
    else:
        return [''] * len(row) # Default to no styling
#%%
set_name_dict = {'c100':'cifar100', 
                 'ti':'tinyimagenet',
                 'c10':'cifar10', 
                 'metrics':'methods'}
methods_dict = {'CTM': 'CTM',
                'CTM_class': 'CTM_class',
                'CTM_class_avg': 'CTM_class_avg',
                'CTM_class_avg_mean': 'CTMmean_class_avg',
                'CTM_class_mean': 'CTMmean_class',
                'CTM_class_pred': 'CTM_class_pred',
                'CTM_class_pred_mean': 'CTMmean_class_pred',
                'CTM_global': 'CTM_global',
                'CTM_global_mean': 'CTMmean_global',
                'CTM_mean': 'CTMmean',
                'CTM_oc_mean': 'CTMmeanOC',
                'MCD-CTM': 'MCD-CTM',
                'MCD-CTM_class': 'MCD-CTM_class',
                'MCD-CTM_class_avg': 'MCD-CTM_class_avg',
                'MCD-CTM_class_avg_mean': 'MCD-CTMmean_class_avg',
                'MCD-CTM_class_mean': 'MCD-CTMmean_class',
                'MCD-CTM_class_pred': 'MCD-CTM_class_pred',
                'MCD-CTM_class_pred_mean': 'MCD-CTMmean_class_pred',
                'MCD-CTM_global': 'MCD-CTM_global',
                'MCD-CTM_global_mean': 'MCD-CTMmean_global',
                'MCD-CTM_mean': 'MCD-CTMmean',
                'MCD-CTM_oc_mean': 'MCD-CTMmeanOC',
                'MCD-ECTM': 'MCD-ECTM',
                'MCD-ECTM_class_avg_mean': 'MCD-ECTMmean_class_avg',
                'MCD-ECTM_class_pred_mean': 'MCD-ECTMmean_class_pred',
                'MCD-ECTM_global_mean': 'MCD-ECTMmean_global',
                'MCD-ECTM_mean': 'MCD-ECTMmean',
                'MCD-ECTM_oc_mean': 'MCD-ECTMmeanOC'}
#%%
best_methods_dict = {}
best_methods_scores_dict = {}
best_methods_ranked_dict = {}
#%%
dataset = 'cifar10'
# dataset = 'supercifar'
# dataset = 'cifar100'
# dataset = 'tiny-imagenet-200'
# dataset = 'cifar10'
# dataset = 'super_cifar100'
# dataset = 'cifar100_modelvit_bbvit_lr0.01'
# dataset = 'tiny-imagenet-200'
#%%
ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
            'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100'}

#%%
vit = False
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
    # 'ood_nsncs_ti' if (dataset=='cifar10' or dataset=='cifar100') else 'ood_sncs_c10':test_ood_3_scores_mean_list,
    # 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10':test_ood_1_scores_mean_list,

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
    
#%%
# results_val['set_name'] = 'val'
# results_test_iid['set_name'] = 'test'
# results_ood_1['set_name'] = 'ood_sncs_c100' if dataset=='cifar10' else 'ood_sncs_c10'
# results_ood_2['set_name'] = 'ood_nsncs_svhn'
# results_ood_3['set_name'] = 'ood_sncs_c10' if dataset=='tiny-imagenet-200' else 'ood_nsncs_ti'
# results_ood_7['set_name'] = 'ood_nsncs_textures'
#%%
# pd.concat([results_val, results_test_iid, results_ood_1, results_ood_2, results_ood_3, results_ood_7])
# pd.concat([results_val, results_test_iid, results_?ood_1, results_ood_2, results_ood_3, results_ood_7]).to_csv(f'{dataset}_raw.csv')

#%%
model_opts=['RW0','RF0','ASHNone']
# score_name = 'AUGRC'
# score_name = 'AURC'
# score_name = 'AUROC_f'
# score_name = 'FPR@95TPR'
best_hyperparameters_list = []
for score_name in ['AUGRC','AURC','AUROC_f','FPR@95TPR']:
    cond_1 = results_val['metrics']=='MCD-ECTM_class_avg'
    cond_2 = results_val['metrics']=='MCD-ECTM_class_pred'
    cond_3 = results_val['metrics']=='MCD-ECTM_global'
    results_val = results_val[~(cond_1|cond_2|cond_3) ]
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
    # for model_name in ['modelvit']:
        for mcd_metrics in mcd_metrics_list:
            # model_name = 'dg'
            # mcd_metrics = True
            best_hyperparameters_scores = select_best_hyperparameters(results_val, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            best_hyperparameters_list.append({'dataset':dataset, 'model':model_name,'mcd_metrics':mcd_metrics,'score_name':score_name,'best_hyperparameters_scores':best_hyperparameters_scores})
            # break
            # l_.append(best_hyperparameters_scores)
            # print(best_hyperparameters_scores.shape)
            #
            scores_test_iid_mean, scores_test_iid_std = select_best_scores(results_test_iid, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_iid_scores_mean_list.append(scores_test_iid_mean)
            test_iid_scores_std_list.append(scores_test_iid_std)
            #
            scores_test_ood_1_mean, scores_test_ood_1_std = select_best_scores(results_ood_1, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_1_scores_mean_list.append(scores_test_ood_1_mean)
            test_ood_1_scores_std_list.append(scores_test_ood_1_std)
            #
            scores_test_ood_2_mean, scores_test_ood_2_std = select_best_scores(results_ood_2, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_2_scores_mean_list.append(scores_test_ood_2_mean)
            test_ood_2_scores_std_list.append(scores_test_ood_2_std)
            #
            scores_test_ood_3_mean, scores_test_ood_3_std  = select_best_scores(results_ood_3, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_3_scores_mean_list.append(scores_test_ood_3_mean)
            test_ood_3_scores_std_list.append(scores_test_ood_3_std)
            #
            scores_test_ood_4_mean, scores_test_ood_4_std  = select_best_scores(results_ood_4, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_4_scores_mean_list.append(scores_test_ood_4_mean)
            test_ood_4_scores_std_list.append(scores_test_ood_4_std)
            #
            scores_test_ood_5_mean, scores_test_ood_5_std  = select_best_scores(results_ood_5, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_5_scores_mean_list.append(scores_test_ood_5_mean)
            test_ood_5_scores_std_list.append(scores_test_ood_5_std)
            #
            scores_test_ood_6_mean, scores_test_ood_6_std  = select_best_scores(results_ood_6, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_6_scores_mean_list.append(scores_test_ood_6_mean)
            test_ood_6_scores_std_list.append(scores_test_ood_6_std)
            #
            scores_test_ood_7_mean, scores_test_ood_7_std  = select_best_scores(results_ood_7, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_7_scores_mean_list.append(scores_test_ood_7_mean)
            test_ood_7_scores_std_list.append(scores_test_ood_7_std)
            #
            scores_test_ood_8_mean, scores_test_ood_8_std  = select_best_scores(results_ood_8, best_hyperparameters_scores, model_name, score_name, mcd_metrics=mcd_metrics, model_opts=model_opts)
            test_ood_8_scores_mean_list.append(scores_test_ood_8_mean)
            test_ood_8_scores_std_list.append(scores_test_ood_8_std)
        # break  
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

    # break
    augrc_list = [pd.concat(ood_nsncs_dict[data_key])[f'{score_name}'].to_frame().rename({f'{score_name}':'_'.join(data_key.split('_')[2:])},axis='columns') for data_key in ood_nsncs_dict]
    augrc_scores_ood = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), augrc_list)
    augrc_scores_ood = augrc_scores_ood.droplevel(level=['network','RankWeight','RankFeat','ASH'])
    augrc_scores_ood = augrc_scores_ood.reset_index()
    augrc_scores_ood = augrc_scores_ood.rename(set_name_dict,axis='columns')
    augrc_scores_ood['methods'] = augrc_scores_ood['methods'].replace(methods_dict)
    augrc_scores_ood['methods'] = augrc_scores_ood['methods'].str.replace('_',' ')
    augrc_scores_ood.columns = augrc_scores_ood.columns.str.replace('_',' ')
    augrc_scores_ood = augrc_scores_ood.set_index(['model','drop out','methods','reward'])
    # augrc_scores_ood = augrc_scores_ood.drop(columns='reward')
    # mean_scores_ood = pd.concat([pd.concat(ood_nsncs_dict[data_key]) for data_key in ood_nsncs_dict]).groupby(level=[0,1,2,3,4,5,6,7]).mean().sort_values(by=['AUGRC'])
    # metrics = augrc_scores_ood.index.get_level_values('metrics')
    # mcd_metrics_mask = np.array(['MCD-' in x for x in  metrics])
    methods = augrc_scores_ood.index.get_level_values('methods')
    mcd_methods_mask = np.array(['MCD-' in x for x in  methods])
    label_model = 'ViT' if vit else 'Conv'
    augrc_scores_ood[~mcd_methods_mask].to_csv(f'scores_final/scores_{score_name}_MCD-False_{label_model}_{dict_clip[dataset]}.csv')
    augrc_scores_ood[mcd_methods_mask].to_csv(f'scores_final/scores_{score_name}_MCD-True_{label_model}_{dict_clip[dataset]}.csv')

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
    # augrc_scores_ood = augrc_scores_ood.drop(columns='reward')
    # mean_scores_ood = pd.concat([pd.concat(ood_nsncs_dict[data_key]) for data_key in ood_nsncs_dict]).groupby(level=[0,1,2,3,4,5,6,7]).mean().sort_values(by=['AUGRC'])
    # metrics = augrc_scores_ood.index.get_level_values('metrics')
    # mcd_metrics_mask = np.array(['MCD-' in x for x in  metrics])
    methods = metric_scores_ood.index.get_level_values('methods')
    mcd_methods_mask = np.array(['MCD-' in x for x in  methods])
    label_model = 'ViT' if vit else 'Conv'
    metric_scores_ood[~mcd_methods_mask].to_csv(f'scores_final/scores_all_{score_name}_MCD-False_{label_model}_{dict_clip[dataset]}.csv')
    metric_scores_ood[mcd_methods_mask].to_csv(f'scores_final/scores_all_{score_name}_MCD-True_{label_model}_{dict_clip[dataset]}.csv')