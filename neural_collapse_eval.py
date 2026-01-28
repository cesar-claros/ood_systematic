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
# configs = pd.read_csv('configs_exp/configs_cifar10_iid_train.txt',sep=None)
# dataset = 'cifar10'
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
                    do_enabled = False
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
                                'architecture' : cf.model.network.backbone,
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
#%%
df_metrics.to_csv('nc_metrics.csv')
df_metrics_global.to_csv('nc_metrics_global.csv')
df_metrics_class_pred.to_csv('nc_metrics_class_pred.csv')
#%%
df_metrics = pd.read_csv('nc_metrics.csv',index_col=0)
df_metrics_global = pd.read_csv('nc_metrics_global.csv',index_col=0)
df_metrics_class_pred = pd.read_csv('nc_metrics_class_pred.csv',index_col=0)
#%%
df_metrics['projection'] = 'None'
df_metrics_global['projection'] = 'Global'
df_metrics_class_pred['projection'] = 'Class pred'
#%%
# df_metrics = df_metrics[df_metrics['study']!='dg']
# df_metrics_global = df_metrics_global[df_metrics_global['study']!='dg']
# df_metrics_class_pred = df_metrics_class_pred[df_metrics_class_pred['study']!='dg']
#%%
# Rename Backbone
df_metrics.loc[df_metrics['study']=='vit','architecture'] = df_metrics.loc[df_metrics['study']=='vit','study']
df_metrics.loc[df_metrics['study']=='dg','architecture'] = 'vgg13'
df_metrics_global.loc[df_metrics_global['study']=='vit','architecture'] = df_metrics_global.loc[df_metrics_global['study']=='vit','study']
df_metrics_global.loc[df_metrics_global['study']=='dg','architecture'] = 'vgg13'
df_metrics_class_pred.loc[df_metrics_class_pred['study']=='vit','architecture'] = df_metrics_class_pred.loc[df_metrics_class_pred['study']=='vit','study']
df_metrics_class_pred.loc[df_metrics_class_pred['study']=='dg','architecture'] = 'vgg13'
#%%
# Filter rewards that are not used
idx_drop_cifar10 = df_metrics.loc[(df_metrics['study']=='dg')&(df_metrics['dataset']=='cifar10')&~(df_metrics['reward'].isin([2.2,3.0,10.0]))].index
idx_drop_supercifar = df_metrics.loc[(df_metrics['study']=='dg')&(df_metrics['dataset']=='supercifar')&~(df_metrics['reward'].isin([2.2,3.0,10.0,20.0]))].index
idx_drop_cifar100 = df_metrics.loc[(df_metrics['study']=='dg')&(df_metrics['dataset']=='cifar100')&~(df_metrics['reward'].isin([6.0,10.0,15.0,20.0]))].index
idx_drop_tinyimagenet = df_metrics.loc[(df_metrics['study']=='dg')&(df_metrics['dataset']=='tinyimagenet')&~(df_metrics['reward'].isin([15.0,20.0]))].index
idx_drop = np.concatenate([idx_drop_cifar10,idx_drop_supercifar,idx_drop_cifar100,idx_drop_tinyimagenet])
#%%
df_metrics = df_metrics.drop(index=idx_drop )
df_metrics_global = df_metrics_global.drop(index=idx_drop )
df_metrics_class_pred = df_metrics_class_pred.drop(index=idx_drop )
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
#%%
# X = df_metrics[df_metrics['architecture']=='vgg13'].drop(['lr','dropout','reward','run','study','dataset','architecture','projection'],axis=1)
X = df_metrics[(df_metrics['architecture']=='vgg13')&(df_metrics['study']!='dg')].drop(['lr','dropout','reward','run','study','dataset','architecture','projection'],axis=1)
#%%
tsne = TSNE(n_components=2, perplexity=5, learning_rate=100, random_state=42)
X_tsne = tsne.fit_transform(X)
#%%
# 3. Visualize the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8)
plt.title('t-SNE Visualization of Neural Collapse Metrics (ViT)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
# plt.colorbar(scatter, ticks=range(len(np.unique(y))), label='Cluster')
plt.grid(True)
plt.show()
#%%


#%%
# dataset = 'cifar10'
# dataset = 'supercifar'
# dataset = 'cifar100'
dataset = 'tiny-imagenet-200'
# dataset = 'cifar10'
# dataset = 'super_cifar100'
# dataset = 'cifar100_modelvit_bbvit_lr0.01'
# dataset = 'tiny-imagenet-200'

#%%
best_ood_list = []
for dataset in ['cifar10','supercifar','cifar100','tiny-imagenet-200']:

    ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
    c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
    dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
                'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100'}

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

    res_list = [results_ood_1,results_ood_2,results_ood_3,results_ood_4,
                results_ood_5,results_ood_6,results_ood_7,results_ood_8]
    ood_list = []
    for model in ['dg', 'devries', 'confidnet']:
        for run in [1,2,3,4,5]:
            rewards_opts = res_list[0]['reward'].unique() if model=='dg' else ['rew2.2']
            ranked_scores = []
            for rew in rewards_opts:
                for res in res_list:
                    r = res[(res['model']==model)&(res['run']==run)&(res['drop_out']=='do0')&(res['reward']==rew)].set_index('metrics').sort_values(by='AUGRC')[['AUGRC','AURC']]
                    mask_global = r.index.str.contains('global', na=False)
                    mask_class = r.index.str.contains('class', na=False)
                    ranked_scores.append(r[~(mask_global|mask_class)])
                if dataset=='cifar10' or dataset=='supercifar' or dataset=='cifar100':
                    near = pd.concat([ranked_scores[0],ranked_scores[2]]).groupby(level='metrics').mean().rank()
                    mid = pd.concat([ranked_scores[3],ranked_scores[4],ranked_scores[5],ranked_scores[1]]).groupby(level='metrics').mean().rank()
                    far = pd.concat([ranked_scores[6],ranked_scores[7]]).groupby(level='metrics').mean().rank()
                elif dataset=='tiny-imagenet-200':
                    near = pd.concat([ranked_scores[0],ranked_scores[2],ranked_scores[3],ranked_scores[4],ranked_scores[5]]).groupby(level='metrics').mean().rank()
                    mid = pd.concat([ranked_scores[6],ranked_scores[7]]).groupby(level='metrics').mean().rank()
                    far = pd.concat([ranked_scores[1]]).groupby(level='metrics').mean().rank()

                ood_method_name = pd.concat([mid]).groupby(level='metrics').mean().mean(axis=1).idxmin()
                # ood_method_name = pd.concat([near,mid,far]).groupby(level='metrics').mean().idxmin()
                ood_list.append({'dataset':dataset,'architecture':'vgg13','dropout':False,'study':model,'run':run,'reward':rew,'ood':ood_method_name,})
    best_ood_list.append(pd.DataFrame(ood_list))
#%%
best_ood_df = pd.concat(best_ood_list)
best_ood_df['ood_method'] = best_ood_df['ood'].str.split('_').str[0]
best_ood_df['reward'] = best_ood_df['reward'].str.split('rew').str[1].astype(float)
best_ood_df['dataset'] = best_ood_df['dataset'].replace({'tiny-imagenet-200':'tinyimagenet'})
#%%
merged_df = pd.merge(left=best_ood_df,right=df_metrics,left_on=['dataset','architecture','study','run','dropout','reward'],right_on=['dataset','architecture','study','run','dropout','reward'])
#%%
X_ = merged_df[merged_df['study']!='dg'].drop(['lr','dropout','reward','run','study','dataset','architecture','projection','ood'],axis=1)
#%%
import seaborn as sns
sns.pairplot(X_, hue='ood_method')
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


#%%
drop_list_weights = ['lr','dropout','reward','run','study','equiangular_uc','equinorm_uc','max_equiangular_uc','self_duality','var_collapse']
drop_list_means = ['lr','dropout','reward','run','study','equiangular_wc','equinorm_wc','max_equiangular_wc',]
group_list = ['projection','architecture','dataset']
df_metrics_weights_grouped = df_metrics.drop(drop_list_weights,axis=1).groupby(group_list)
df_metrics_grouped = df_metrics.drop(drop_list_means,axis=1).groupby(group_list)
df_metrics_global_grouped = df_metrics_global.drop(drop_list_means,axis=1).groupby(group_list)
df_metrics_class_pred_grouped = df_metrics_class_pred.drop(drop_list_means,axis=1).groupby(group_list)

#%%
cmap_name = 'pink'
#%%
df_mean_weights = df_metrics_weights_grouped.mean().droplevel(0).iloc[[0,2,1,3,4,6,5,7],[1,0,2]]
# df_mean_weights.round(4).reset_index().to_csv('nc_metrics_weights_mean.csv')
df_mean_weights.round(4).reset_index().to_markdown(index=False)
# print(df_mean_weights.style.format(precision=4).background_gradient(cmap=cmap_name).to_latex(convert_css=True))
#%%
df_mean = df_metrics_grouped.mean().iloc[[0,2,1,3,4,6,5,7],[2,1,3,0,4]]
# df_mean.round(4).reset_index().to_csv('nc_metrics_mean.csv')
# print(df_mean.round(4).reset_index().to_markdown(index=False))
# print(df_mean.style.format(precision=4).background_gradient(cmap='cividis').to_latex(convert_css=True))
# print(df_mean.round(4).reset_index().to_markdown(index=False))
#%%
df_global_mean = df_metrics_global_grouped.mean().iloc[[0,2,1,3,4,6,5,7],[2,1,3,0,4]]
# df_global_mean.round(4).reset_index().to_csv('nc_metrics_global_mean.csv')
# print(df_global_mean.round(4).reset_index().to_markdown(index=False))
# print(df_global_mean.style.format(precision=4).background_gradient(cmap='cividis').to_latex(convert_css=True))
#%%
df_class_pred_mean = df_metrics_class_pred_grouped.mean().iloc[[0,2,1,3,4,6,5,7],[2,1,3,0,4]]
# df_class_pred_mean.round(4).reset_index().to_csv('nc_metrics_class_pred_mean.csv')
# print(df_class_pred_mean.round(4).reset_index().to_markdown(index=False))
# print(df_class_pred_mean.style.format(precision=4).background_gradient(cmap='cividis').to_latex(convert_css=True))
#%%
df_concat = pd.concat([df_mean,df_global_mean,df_class_pred_mean])
#%%
df_concat.loc[(slice(None),'vgg13',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].style.format(precision=4).background_gradient(cmap=cmap_name)
# print(df_concat.loc[(slice(None),'vgg13',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].style.format(precision=4).background_gradient(cmap=cmap_name).to_latex(convert_css=True))
print(df_concat.loc[(slice(None),'vgg13',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].round(4).reset_index().to_markdown(index=False))
#%%
df_concat.loc[(slice(None),'vit',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].style.format(precision=4).background_gradient(cmap=cmap_name)
# print(df_concat.loc[(slice(None),'vit',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].style.format(precision=4).background_gradient(cmap=cmap_name).to_latex(convert_css=True))
print(df_concat.loc[(slice(None),'vit',slice(None))].swaplevel().loc[(['cifar10','supercifar','cifar100','tinyimagenet'],slice(None))].round(4).reset_index().to_markdown(index=False))
#%%)
df_metrics_grouped.std().iloc[[0,2,1,3,4,6,5,7]].style.background_gradient().format(precision=3)
#%%
path = args.model_path
rank_weight_opt = args.rank_weight
rank_feat_opt = args.rank_feature
ash_method_opt = args.ash # 'ash_s@90'
use_cuda_opt = args.use_cuda
temperature_scale_opt = args.temperature_scale
#%%
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
    do_enabled = False
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
model_evaluations = {}