#%%
import os
import joblib
from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Tuple, List, Union, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _sigmoid_calibration
from sklearn.metrics import log_loss
import pandas as pd
from src.calibration_metric import ece
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
import argparse
from src.utils_calibration import *

#%%
# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')
#%%

#%%
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "supercifar100", "tinyimagenet"],
        help="Dataset choice."
    )
    p.add_argument(
        "--vit", 
        action="store_true", 
        help="Set this flag if using ViT model results"
        )
    p.add_argument(
        "--scores-dir", 
        type=str, 
        default="scores_final", 
        help="Directory to save final scores"
        )
    args = p.parse_args()

    vit = args.vit
    vit_cond = "ViT" if vit else "Conv"

    dataset_in = args.dataset  # original CLI name (for output filename)
    data_dict_path = os.path.join(args.scores_dir, f"calibration_results_{dataset_in}_{vit_cond}.csv")
     
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.scores_dir):
        logger.info(f"Creating output directory: {args.scores_dir}")
        os.makedirs(args.scores_dir, exist_ok=True)
    else:
        logger.debug(f"Output directory exists: {args.scores_dir}")

    # if os.path.exists(data_dict_path):
    #     logger.info(f"Results already exist at {data_dict_path}. Skipping.")
    #     return

    dataset = resolve_dataset_name(dataset_in, vit)

    # Logic from original script adapted to use args
    ti_condition = 'ood_nsncs_ti' if (
        dataset == 'cifar10' or dataset.split('_')[0] == 'cifar100' or dataset == 'super_cifar100' or dataset == 'supercifar'
    ) else 'ood_sncs_c10'

    c100_condition = 'ood_sncs_c100' if (dataset == 'cifar10' or dataset == 'tiny-imagenet-200') else 'ood_sncs_c10'

    ood_sets = [
        'iid_test', c100_condition, 'ood_nsncs_svhn', ti_condition,
        'ood_nsncs_lsun_cropped','ood_nsncs_lsun_resize','ood_nsncs_isun',
        'ood_nsncs_textures','ood_nsncs_places365'
    ]

    # Load data
    results_ood = {}
    if vit:
        results_val = read_confids_vit(dataset, 'iid_val')
        for ood in ood_sets:
            results_ood[ood] = read_confids_vit(dataset, ood)
        model_names = ['modelvit']
        range_models = range(-1,4)
        keep_exceptions = ["KPCA_RecError_global", "PCA_RecError_global",]
    else:
        results_val = read_confids(dataset, 'iid_val')
        for ood in ood_sets:
            results_ood[ood] = read_confids(dataset, ood)
        model_names = ['confidnet','devries','dg']
        range_models = range(5)
        keep_exceptions = ["KPCA_RecError_global", "PCA_RecError_global", "MCD-KPCA_RecError_global", "MCD-PCA_RecError_global"]
    
    # Do not include modified scores for now
    
    mask_keep =  [x for x in list(results_val.columns) if ('global' not in x) and ('class' not in x) ]
    mask_keep.extend(keep_exceptions)
    results_val = results_val[mask_keep]
    # End    

    calibration_methods = ['Sigmoid','Isotonic','Beta']
    data_dict = []
    for model in model_names:
        rewards = results_val[results_val['model']==model]['reward'].unique()
        networks = results_val[results_val['model']==model]['network'].unique()
        # rewards = ['rew2.2', 'rew10', 'rew3', 'rew6'] if model=='dg' else ['rew2.2']
        for rew in rewards:
            for do in ['do0','do1']:
                for run in range_models:
                    for net in networks:
                        condition = (results_val['model']==model) &\
                                        (results_val['drop_out']==do) &\
                                        (results_val['reward']==rew) &\
                                        (results_val['run']==run+1) &\
                                        (results_val['network']==net)
                        val_df = results_val[condition]
                        mcd_condition = 2 if do=='do1' else 1
                        assert len(val_df['filepath'].unique())<=mcd_condition, f"Expected at most {mcd_condition} unique filepath for model={model}, do={do}, rew={rew}, run={run+1}, but found {len(val_df['filepath'].unique())}"  
                        # Looping when mcd_confids and non_mcd_confids exist for same (model, do, rew, run) - 
                        for val_path in val_df['filepath'].unique():
                            val = val_df[val_df['filepath']==val_path]
                            logger.info(f"Processing model={model}, net={net}, do={do}, rew={rew}, run={run+1} with {len(val)} validation samples.")
                            # --- Pre-extract arrays to avoid pandas overhead/pickling costs ---
                            val_labels = 1 - val['residuals'].to_numpy().astype(int)

                            val = val.dropna(axis=1).drop(
                                ['residuals','model','network','drop_out','run','reward','RankWeight','RankFeat','ASH','filepath'],
                                axis=1
                            )
                            
                            # Pull val scores once
                            val_scores_by_method = {c: val[c].to_numpy(dtype=float, copy=False) for c in val.columns}
                            # Check for non-finite values in any of the score columns before parallel processing
                            for c in val.columns:
                                arr = val_scores_by_method[c]
                                if not np.isfinite(arr).all():
                                    bad = np.sum(~np.isfinite(arr))
                                    logger.warning(f"Non-finite in val column {c}: {bad} / {arr.size}")

                            # Pull OOD scores/labels once for this (model, do, rew, run)
                            ood_pre = {}
                            for ood in ood_sets:
                                ood_all_df = results_ood[ood]
                                cond_ood = (
                                    (ood_all_df['model'] == model) &
                                    (ood_all_df['drop_out'] == do) &
                                    (ood_all_df['reward'] == rew) &
                                    (ood_all_df['run'] == run + 1) &
                                    (ood_all_df['network'] == net)
                                )
                                ood_df = ood_all_df[cond_ood]
                                assert len(ood_df['filepath'].unique()) <= mcd_condition, (
                                    f"Expected at most {mcd_condition} unique filepath for OOD dataset={ood}, "
                                    f"model={model}, do={do}, rew={rew}, run={run+1}, but found {len(ood_df['filepath'].unique())}"
                                )
                                ood_df = ood_df[list(val.columns)+['residuals']]  # only keep relevant cols + residuals for labels
                                ood_df = ood_df.dropna(axis=0)  # drop rows with NaNs in any of the score columns
                                ood_labels = 1 - ood_df['residuals'].to_numpy().astype(int)

                                # only keep the same methods we're calibrating on (handles missing cols safely)
                                ood_scores_by_method = {
                                    c: ood_df[c].to_numpy(dtype=float, copy=False)
                                    for c in val.columns
                                    if c in ood_df.columns
                                }

                                ood_pre[ood] = (ood_labels, ood_scores_by_method)

                            def run_one(csf_name: str, cal_method: str):
                                rows = []
                                s_val_raw = val_scores_by_method[csf_name]
                                s_val, y_val = finite_xy(s_val_raw, val_labels)
                                # If too few points after filtering, skip safely
                                if s_val.size < s_val_raw.size:  # arbitrary threshold to ensure enough data for calibration; adjust as needed
                                    logger.warning(f"Skipping calibration for model={model}, net={net}, do={do}, rew={rew}, run={run+1}, method={csf_name}, cal_method={cal_method} due to insufficient valid samples after filtering: {s_val.size} < 900")
                                    return []
                                # Fit
                                calibrator = Calibrator(method=cal_method.lower(), cv=5, random_state=42)
                                calibrator.fit(s_val, y_val)

                                # Val eval
                                # p_val = calibrator.predict_proba(s_val)
                                p_val_cv, y_val_cv = calibrator.get_cv_probs_labels()
                                k1_val, bar_p_val =  extract_k_and_pood([p_val_cv], [y_val_cv], bound_type='l1')
                                k2_val, bar_p2_val =  extract_k_and_pood([p_val_cv], [y_val_cv], bound_type='l2')
                                mask_id = (y_val_cv == 1)
                                mask_ood = (y_val_cv == 0)
                                n_id, n_ood = np.sum(mask_id), np.sum(mask_ood)
                                alpha = n_ood / n_id
                                if 'vgg' in net:
                                    architecture = 'VGG13'
                                elif 'vit' in net: 
                                    architecture = 'ViT'
                                elif 'resnet' in net:
                                    architecture = 'ResNet18'
                                else:
                                    architecture = 'Unknown'
                                rows.append({
                                    'dataset': 'val',
                                    'method': csf_name,
                                    'calibration': cal_method,
                                    'model': model,
                                    'architecture': architecture,
                                    'drop_out': do,
                                    'reward': rew,
                                    'run': run+1,
                                    'ece_l1': ece(p_val_cv, y_val_cv, mode='l1'),
                                    'ece_l2': ece(p_val_cv, y_val_cv, mode='l2'),
                                    'mce': ece(p_val_cv, y_val_cv, mode='inf'),
                                    'ece_l1_bound': calc_l1_bounds_from_mixed([p_val_cv], [y_val_cv], fixed_alpha=None)[0],
                                    'ece_l2_bound': calc_l2_bounds_from_mixed([p_val_cv], [y_val_cv], fixed_alpha=None)[0],
                                    'k1': k1_val[0],
                                    'bar_p_ood': bar_p_val[0],
                                    'k2': k2_val[0],
                                    'bar_p2_ood': bar_p2_val[0],
                                    'alpha': alpha,
                                })

                                # OOD evals
                                for ood in ood_sets:
                                    ood_labels_raw, ood_scores_map = ood_pre[ood]
                                    s_ood_raw = ood_scores_map.get(csf_name)
                                    if s_ood_raw is None:
                                        continue

                                    s_ood, y_ood = finite_xy(s_ood_raw, ood_labels_raw)
                                    if s_ood.size == 0:
                                        continue
                                    # ood_labels, ood_scores_map = ood_pre[ood]
                                    # s_ood = ood_scores_map.get(csf_name)
                                    # if s_ood is None:
                                        # continue

                                    p_ood = calibrator.predict_proba(s_ood)
                                    k1, bar_p_ood =  extract_k_and_pood([p_ood], [y_ood], bound_type='l1')
                                    k2, bar_p2_ood =  extract_k_and_pood([p_ood], [y_ood], bound_type='l2')
                                    mask_id = (y_ood == 1)
                                    mask_ood = (y_ood == 0)
                                    n_id, n_ood = np.sum(mask_id), np.sum(mask_ood)
                                    alpha = n_ood / n_id
                                    # if 'vgg' in net:
                                    #     architecture = 'VGG13'
                                    # elif 'vit' in net: 
                                    #     architecture = 'ViT'
                                    # elif 'resnet' in net:
                                    #     architecture = 'ResNet18'
                                    # else:
                                    #     architecture = 'Unknown'
                                    rows.append({
                                        'dataset': ood,
                                        'method': csf_name,
                                        'calibration': cal_method,
                                        'model': model,
                                        'architecture': architecture,
                                        'drop_out': do,
                                        'reward': rew,
                                        'run': run+1,
                                        'ece_l1': ece(p_ood, y_ood, mode='l1'),
                                        'ece_l2': ece(p_ood, y_ood, mode='l2'),
                                        'mce': ece(p_ood, y_ood, mode='inf'),
                                        'ece_l1_bound': calc_l1_bounds_from_mixed([p_ood], [y_ood], fixed_alpha=None)[0],
                                        'ece_l2_bound': calc_l2_bounds_from_mixed([p_ood], [y_ood], fixed_alpha=None)[0],
                                        'k1': k1[0],
                                        'bar_p_ood': bar_p_ood[0],
                                        'k2': k2[0],
                                        'bar_p2_ood': bar_p2_ood[0],
                                        'alpha': alpha,
                                    })

                                return rows

                            tasks = list(product(val.columns, calibration_methods))

                            # Start with threads (usually good for sklearn/numpy + avoids pickling huge arrays)
                            all_rows = Parallel(n_jobs=-1, prefer="threads")(
                                delayed(run_one)(csf_name, cal_method) for csf_name, cal_method in tasks
                            )

                            # Flatten into your existing list
                            for rows in all_rows:
                                data_dict.extend(rows)


    # ... keep the rest of your logic exactly as-is ...
    # At the end:
    data_dict_df = pd.DataFrame(data_dict)
    logger.info(f"Saving calibration results to {data_dict_path} with {len(data_dict_df)} rows.")
    # outpath = os.path.join(args.scores_dir, os.path.basename(data_dict_path))
    data_dict_df.to_csv(data_dict_path, index=False)

if __name__ == "__main__":
    main()

# ==========================================
# 4. Main Execution and Data Generation
# ==========================================
#%%
# if __name__ == "__main__":

#     for vit in [False, True]:
#         vit_cond = 'ViT' if vit else 'Conv'
#         for dataset in ['cifar10', 'cifar100', 'supercifar100','tinyimagenet']:
#             data_dict_path = f'calibration_results_{dataset}_{vit_cond}.csv'
#             if os.path.exists(data_dict_path):
#                 logger.info(f"Results for dataset={dataset}, vit={vit_cond} already exist at {data_dict_path}. Skipping data generation.")
#                 continue
#             # Dataset naming logic adapted from original script to handle ViT vs Conv and various dataset naming patterns
#             if dataset == 'cifar100' and vit:
#                 dataset = 'cifar100_modelvit_bbvit_lr0.01'
#             elif dataset == 'supercifar100' and vit:
#                 dataset = 'super_cifar100'
#             elif dataset == 'supercifar100' and not vit:
#                 dataset = 'supercifar'
#             elif dataset == 'tinyimagenet':
#                 dataset = 'tiny-imagenet-200'

#             # Logic from original script adapted to use args
#             ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
#             c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
#             dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
#                         'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100', 'super_cifar10':'supercifar10'} # added super_cifar10 just in case as it wasn't in original dict but might be needed given dataset naming patterns, or maybe 'supercifar' covers it. kept original keys mostly.
#             ood_sets = ['iid_test',c100_condition,'ood_nsncs_svhn',ti_condition,'ood_nsncs_lsun_cropped','ood_nsncs_lsun_resize','ood_nsncs_isun','ood_nsncs_textures','ood_nsncs_places365']
#             results_ood = {}    
#             if vit:
#                 results_val = read_confids_vit(dataset,'iid_val')
#                 # results_test_iid = read_confids_vit(dataset)
#                 for ood in ood_sets:
#                     results_ood[ood] = read_confids_vit(dataset,ood)
#             else:
#                 results_val = read_confids(dataset,'iid_val')
#                 # results_test_iid = read_confids(dataset,'iid_test')
#                 for ood in ood_sets:
#                     results_ood[ood] = read_confids(dataset,ood)

#             calibration_methods = ['Sigmoid','Isotonic','Beta']
#             data_dict = []
#             for model in ['confidnet','devries','dg']:
#                 rewards = results_val[results_val['model']==model]['reward'].unique()
#                 # rewards = ['rew2.2', 'rew10', 'rew3', 'rew6'] if model=='dg' else ['rew2.2']
#                 for rew in rewards:
#                     for do in ['do0','do1']:
#                         for run in range(5):
#                             condition = (results_val['model']==model) &\
#                                             (results_val['drop_out']==do) &\
#                                             (results_val['reward']==rew) &\
#                                             (results_val['run']==run+1)
#                             val_df = results_val[condition]
#                             mcd_condition = 2 if do=='do1' else 1
#                             assert len(val_df['filepath'].unique())==mcd_condition, f"Expected exactly {mcd_condition} unique filepath for model={model}, do={do}, rew={rew}, run={run+1}, but found {len(val_df['filepath'].unique())}"  
#                             # Looping when mcd_confids and non_mcd_confids exist for same (model, do, rew, run) - 
#                             for val_path in val_df['filepath'].unique():
#                                 val = val_df[val_df['filepath']==val_path]
#                                 logger.info(f"Processing model={model}, do={do}, rew={rew}, run={run+1} with {len(val)} validation samples.")
#                                 # --- Pre-extract arrays to avoid pandas overhead/pickling costs ---
#                                 val_labels = 1 - val['residuals'].to_numpy().astype(int)

#                                 val = val.dropna(axis=1).drop(
#                                     ['residuals','model','network','drop_out','run','reward','RankWeight','RankFeat','ASH','filepath'],
#                                     axis=1
#                                 )

#                                 # Pull val scores once
#                                 val_scores_by_method = {c: val[c].to_numpy(dtype=float, copy=False) for c in val.columns}

#                                 # Pull OOD scores/labels once for this (model, do, rew, run)
#                                 ood_pre = {}
#                                 for ood in ood_sets:
#                                     ood_all_df = results_ood[ood]
#                                     cond_ood = (
#                                         (ood_all_df['model'] == model) &
#                                         (ood_all_df['drop_out'] == do) &
#                                         (ood_all_df['reward'] == rew) &
#                                         (ood_all_df['run'] == run + 1)
#                                     )
#                                     ood_df = ood_all_df[cond_ood]
#                                     assert len(ood_df['filepath'].unique()) == mcd_condition, (
#                                         f"Expected exactly {mcd_condition} unique filepath for OOD dataset={ood}, "
#                                         f"model={model}, do={do}, rew={rew}, run={run+1}, but found {len(ood_df['filepath'].unique())}"
#                                     )
#                                     ood_df = ood_df[list(val.columns)+['residuals']]  # only keep relevant cols + residuals for labels
#                                     ood_df = ood_df.dropna(axis=0)  # drop rows with NaNs in any of the score columns
#                                     ood_labels = 1 - ood_df['residuals'].to_numpy().astype(int)

#                                     # only keep the same methods we're calibrating on (handles missing cols safely)
#                                     ood_scores_by_method = {
#                                         c: ood_df[c].to_numpy(dtype=float, copy=False)
#                                         for c in val.columns
#                                         if c in ood_df.columns
#                                     }

#                                     ood_pre[ood] = (ood_labels, ood_scores_by_method)

#                                 calibration_methods = ['Sigmoid', 'Isotonic', 'Beta']

#                                 def run_one(csf_name: str, cal_method: str):
#                                     rows = []

#                                     # Fit
#                                     calibrator = Calibrator(method=cal_method.lower(), cv=5, random_state=42)
#                                     s_val = val_scores_by_method[csf_name]
#                                     calibrator.fit(s_val, val_labels)

#                                     # Val eval
#                                     p_val = calibrator.predict_proba(s_val)
#                                     rows.append({
#                                         'dataset': 'val',
#                                         'method': csf_name,
#                                         'calibration': cal_method,
#                                         'model': model,
#                                         'drop_out': do,
#                                         'reward': rew,
#                                         'run': run+1,
#                                         'ece_l1': ece(p_val, val_labels, mode='l1'),
#                                         'ece_l2': ece(p_val, val_labels, mode='l2'),
#                                         'mce': ece(p_val, val_labels, mode='inf'),
#                                         'ece_l1_bound': calc_l1_bounds_from_mixed([p_val], [val_labels], fixed_alpha=None)[0],
#                                         'ece_l2_bound': calc_l2_bounds_from_mixed([p_val], [val_labels], fixed_alpha=None)[0],
#                                     })

#                                     # OOD evals
#                                     for ood in ood_sets:
#                                         ood_labels, ood_scores_map = ood_pre[ood]
#                                         s_ood = ood_scores_map.get(csf_name)
#                                         if s_ood is None:
#                                             continue

#                                         p_ood = calibrator.predict_proba(s_ood)
#                                         rows.append({
#                                             'dataset': ood,
#                                             'method': csf_name,
#                                             'calibration': cal_method,
#                                             'model': model,
#                                             'drop_out': do,
#                                             'reward': rew,
#                                             'run': run+1,
#                                             'ece_l1': ece(p_ood, ood_labels, mode='l1'),
#                                             'ece_l2': ece(p_ood, ood_labels, mode='l2'),
#                                             'mce': ece(p_ood, ood_labels, mode='inf'),
#                                             'ece_l1_bound': calc_l1_bounds_from_mixed([p_ood], [ood_labels], fixed_alpha=None)[0],
#                                             'ece_l2_bound': calc_l2_bounds_from_mixed([p_ood], [ood_labels], fixed_alpha=None)[0],
#                                         })

#                                     return rows

#                                 tasks = list(product(val.columns, calibration_methods))

#                                 # Start with threads (usually good for sklearn/numpy + avoids pickling huge arrays)
#                                 all_rows = Parallel(n_jobs=-1, prefer="threads")(
#                                     delayed(run_one)(csf_name, cal_method) for csf_name, cal_method in tasks
#                                 )

#                                 # Flatten into your existing list
#                                 for rows in all_rows:
#                                     data_dict.extend(rows)
                                    
#             data_dict_df = pd.DataFrame(data_dict)
#             logger.info(f"Saving calibration results to {data_dict_path} with {len(data_dict_df)} rows.")
#             data_dict_df.to_csv(data_dict_path, index=False)
#%%
# from joblib import Parallel, delayed
# from itertools import product

# after you compute `val_labels` and create `val` (columns only)

                            # val_labels = 1 - val['residuals'].to_numpy().astype(int)

                            # val = val.dropna(axis=1).drop(
                            #     ['residuals','model','network','drop_out','run','reward','RankWeight','RankFeat','ASH','filepath'],
                            #     axis=1
                            # )

                            # # Pull val scores once
                            # val_scores_by_method = {c: val[c].to_numpy(dtype=float, copy=False) for c in val.columns}

                            # # Pull OOD scores/labels once for this (model, do, rew, run)
                            # ood_pre = {}
                            # for ood in ood_sets:
                            #     ood_all_df = results_ood[ood]
                            #     cond_ood = (
                            #         (ood_all_df['model'] == model) &
                            #         (ood_all_df['drop_out'] == do) &
                            #         (ood_all_df['reward'] == rew) &
                            #         (ood_all_df['run'] == run + 1)
                            #     )
                            #     ood_df = ood_all_df[cond_ood]
                            #     assert len(ood_df['filepath'].unique()) == 1, (
                            #         f"Expected exactly one unique filepath for OOD dataset={ood}, "
                            #         f"model={model}, do={do}, rew={rew}, run={run+1}, but found {len(ood_df['filepath'].unique())}"
                            #     )

                            #     ood_labels = 1 - ood_df['residuals'].to_numpy().astype(int)

                            #     # only keep the same methods we're calibrating on (handles missing cols safely)
                            #     ood_scores_by_method = {
                            #         c: ood_df[c].to_numpy(dtype=float, copy=False)
                            #         for c in val.columns
                            #         if c in ood_df.columns
                            #     }

                            #     ood_pre[ood] = (ood_labels, ood_scores_by_method)

                            # calibration_methods = ['Sigmoid', 'Isotonic', 'Beta']

                            # def run_one(csf_name: str, cal_method: str):
                            #     rows = []

                            #     # Fit
                            #     calibrator = Calibrator(method=cal_method.lower(), cv=5, random_state=42)
                            #     s_val = val_scores_by_method[csf_name]
                            #     calibrator.fit(s_val, val_labels)

                            #     # Val eval
                            #     p_val = calibrator.predict_proba(s_val)
                            #     rows.append({
                            #         'dataset': 'val',
                            #         'method': csf_name,
                            #         'calibration': cal_method,
                            #         'model': model,
                            #         'drop_out': do,
                            #         'reward': rew,
                            #         'run': run+1,
                            #         'ece_l1': ece(p_val, val_labels, mode='l1'),
                            #         'ece_l2': ece(p_val, val_labels, mode='l2'),
                            #         'mce': ece(p_val, val_labels, mode='inf'),
                            #         'ece_l1_bound': calc_l1_bounds_from_mixed([p_val], [val_labels], fixed_alpha=None)[0],
                            #         'ece_l2_bound': calc_l2_bounds_from_mixed([p_val], [val_labels], fixed_alpha=None)[0],
                            #     })

                            #     # OOD evals
                            #     for ood in ood_sets:
                            #         ood_labels, ood_scores_map = ood_pre[ood]
                            #         s_ood = ood_scores_map.get(csf_name)
                            #         if s_ood is None:
                            #             continue

                            #         p_ood = calibrator.predict_proba(s_ood)
                            #         rows.append({
                            #             'dataset': ood,
                            #             'method': csf_name,
                            #             'calibration': cal_method,
                            #             'model': model,
                            #             'drop_out': do,
                            #             'reward': rew,
                            #             'run': run+1,
                            #             'ece_l1': ece(p_ood, ood_labels, mode='l1'),
                            #             'ece_l2': ece(p_ood, ood_labels, mode='l2'),
                            #             'mce': ece(p_ood, ood_labels, mode='inf'),
                            #             'ece_l1_bound': calc_l1_bounds_from_mixed([p_ood], [ood_labels], fixed_alpha=None)[0],
                            #             'ece_l2_bound': calc_l2_bounds_from_mixed([p_ood], [ood_labels], fixed_alpha=None)[0],
                            #         })

                            #     return rows

                            # tasks = list(product(val.columns, calibration_methods))

                            # # Start with threads (usually good for sklearn/numpy + avoids pickling huge arrays)
                            # all_rows = Parallel(n_jobs=-1, prefer="threads")(
                            #     delayed(run_one)(csf_name, cal_method) for csf_name, cal_method in tasks
                            # )

                            # # Flatten into your existing list
                            # for rows in all_rows:
                            #     data_dict.extend(rows)

#%%

# for vit in [False, True]:
#     for dataset in ['cifar10', 'cifar100', 'supercifar100','tinyimagenet']:

#         if dataset == 'cifar100' and vit:
#             dataset = 'cifar100_modelvit_bbvit_lr0.01'
#         elif dataset == 'supercifar100' and vit:
#             dataset = 'super_cifar100'
#         elif dataset == 'supercifar100' and not vit:
#             dataset = 'supercifar'
#         elif dataset == 'tinyimagenet':
#             dataset = 'tiny-imagenet-200'

#         # Logic from original script adapted to use args
#         ti_condition = 'ood_nsncs_ti' if (dataset=='cifar10' or dataset.split('_')[0]=='cifar100' or dataset=='super_cifar100' or dataset=='supercifar') else 'ood_sncs_c10'
#         c100_condition = 'ood_sncs_c100' if (dataset=='cifar10' or dataset=='tiny-imagenet-200') else 'ood_sncs_c10'
#         dict_clip = {'cifar10':'cifar10', 'cifar100':'cifar100', 'supercifar':'supercifar100','tiny-imagenet-200':'tinyimagenet',
#                     'super_cifar100':'supercifar100','cifar100_modelvit_bbvit_lr0.01':'cifar100', 'super_cifar10':'supercifar10'} # added super_cifar10 just in case as it wasn't in original dict but might be needed given dataset naming patterns, or maybe 'supercifar' covers it. kept original keys mostly.
#         ood_sets = ['iid_test',c100_condition,'ood_nsncs_svhn',ti_condition,'ood_nsncs_lsun_cropped','ood_nsncs_lsun_resize','ood_nsncs_isun','ood_nsncs_textures','ood_nsncs_places365']
#         results_ood = {}    
#         if vit:
#             results_val = read_confids_vit(dataset,'iid_val')
#             # results_test_iid = read_confids_vit(dataset)
#             for ood in ood_sets:
#                 results_ood[ood] = read_confids_vit(dataset,ood)
#         else:
#             results_val = read_confids(dataset,'iid_val')
#             # results_test_iid = read_confids(dataset,'iid_test')
#             for ood in ood_sets:
#                 results_ood[ood] = read_confids(dataset,ood)

#         calibration_methods = ['Sigmoid','Isotonic','Beta']
#         data_dict = []
#         for model in ['confidnet','devries','dg']:
#             rewards = ['rew2.2', 'rew10', 'rew3', 'rew6'] if model=='dg' else ['rew2.2']
#             for rew in rewards:
#                 for do in ['do0','do1']:
#                     for run in range(5):
#                         condition = (results_val['model']==model) &\
#                                         (results_val['drop_out']==do) &\
#                                         (results_val['reward']==rew) &\
#                                         (results_val['run']==run+1)
#                         val = results_val[condition]
#                         assert len(val['filepath'].unique())==1, f"Expected exactly one unique filepath for model={model}, do={do}, rew={rew}, run={run+1}, but found {len(val['filepath'].unique())}"  
#                         # logger.info(f"Processing model={model}, do={do}, rew={rew}, run={run+1} with {len(val)} validation samples.")
#                         val_labels = 1 - val['residuals'].values.astype(int)  # Assuming 'residuals' is 0 for correct and 1 for incorrect predictions
#                         val = val.dropna(axis=1).drop(['residuals','model','network','drop_out','run','reward','RankWeight','RankFeat','ASH','filepath'],axis=1)
#                         for csf_name in tqdm(   val.columns, desc=f"Calibrating {model} {do} {rew} run{run+1}", leave=False):
#                             val_scores = val[csf_name].values
#                             for cal_method in tqdm(calibration_methods, desc=f"Calibrating {model} {do} {rew} run{run+1}", leave=False):
#                                 calibrator = Calibrator(method=cal_method.lower(), cv=5, random_state=42)
#                                 calibrator.fit(val_scores, val_labels)
#                                 probs_val = calibrator.predict_proba(val_scores)
#                                 data_dict.append({
#                                         'dataset': 'val',
#                                         'method': csf_name,
#                                         'calibration': cal_method,
#                                         'model': model,
#                                         'drop_out': do,
#                                         'reward': rew,
#                                         'run': run+1,
#                                         'ece_l1': ece(probs_val, val_labels, mode='l1'),
#                                         'ece_l2': ece(probs_val, val_labels, mode='l2'),
#                                         'mce': ece(probs_val, val_labels, mode='inf'),
#                                         'ece_l1_bound' : calc_l1_bounds_from_mixed([probs_val], [val_labels], fixed_alpha=None)[0],
#                                         'ece_l2_bound' : calc_l2_bounds_from_mixed([probs_val], [val_labels], fixed_alpha=None)[0],
#                                     })
#                                 for ood in tqdm(ood_sets, desc=f"Processing OOD datasets for model={model}, do={do}, rew={rew}, run={run+1}", leave=False):
#                                     ood_all_df = results_ood[ood]
#                                     condition = (ood_all_df['model']==model) &\
#                                                 (ood_all_df['drop_out']==do) &\
#                                                 (ood_all_df['reward']==rew) &\
#                                                 (ood_all_df['run']==run+1)
#                                     ood_df = ood_all_df[condition]
#                                     assert len(ood_df['filepath'].unique())==1, f"Expected exactly one unique filepath for OOD dataset={ood}, model={model}, do={do}, rew={rew}, run={run+1}, but found {len(ood_df['filepath'].unique())}" 
#                                     ood_scores = ood_df[csf_name].values  # Assuming 'residuals' is 0 for correct and 1 for incorrect predictions
#                                     ood_labels = 1 - ood_df['residuals'].values.astype(int)  # Assuming 'residuals' is 0 for correct and 1 for incorrect predictions
#                                     probs_ood = calibrator.predict_proba(ood_scores)
#                                     data_dict.append({
#                                         'dataset': ood,
#                                         'method': csf_name,
#                                         'calibration': cal_method,
#                                         'model': model,
#                                         'drop_out': do,
#                                         'reward': rew,
#                                         'run': run+1,
#                                         'ece_l1': ece(probs_ood, ood_labels, mode='l1'),
#                                         'ece_l2': ece(probs_ood, ood_labels, mode='l2'),
#                                         'mce': ece(probs_ood, ood_labels, mode='inf'),
#                                         'ece_l1_bound' : calc_l1_bounds_from_mixed([probs_ood], [ood_labels], fixed_alpha=None)[0],
#                                         'ece_l2_bound' : calc_l2_bounds_from_mixed([probs_ood], [ood_labels], fixed_alpha=None)[0], 
#                                     })
        
#         data_dict_df = pd.DataFrame(data_dict)
#         vit_cond = 'ViT' if vit else 'Conv'
#         data_dict_df.to_csv(f'calibration_results_{dataset}_{vit_cond}.csv', index=False)
#%%
# Pick the scores based on the score in the validation set
# data_dict_df[(data_dict_df['method']=='MLS')&(data_dict_df['dataset']=='val')].set_index(['dataset','method','model','drop_out','reward']).groupby(['calibration']).mean().sort_values('ece_l1')
#%%
# val.dropna(axis=1).drop(['residuals','model','network','drop_out','run','reward','RankWeight','RankFeat','ASH'],axis=1)
#%%
# data_dict[method]['cifar100']['probs'][0]
# data_dict[method]['cifar100']['labels'][0]
#%%
# plot_route_1_severity_stats(data_dict)
#%%
# plot_route_2_alpha_stats(data_dict, target_dataset=ood_sets[1], max_alpha=5.0)
#%%
# plot_route_3_pareto_stats(data_dict, target_dataset=ood_sets[1])
#%%
# plot_route_4_variance_stats(data_dict, method='Sigmoid')
#%%
# n_models = 5
# n_val_samples = 1000
# n_test_id_hits = 500
# n_test_ood = 500

# calibration_methods = ['sigmoid', 'isotonic', 'beta']
# ood_dataset_names = ['CIFAR100 (Near)', 'SuperCIFAR (Mid)', 'SVHN (Far)']

# id_probs_store = {m: np.zeros((n_models, n_test_id_hits)) for m in calibration_methods}
# ood_probs_store = {m: {ds: np.zeros((n_models, n_test_ood)) for ds in ood_dataset_names} for m in calibration_methods}

# print("Generating simulated data, calibrating, and plotting...")

# for model_idx in range(n_models):
#     rng = np.random.default_rng(100 + model_idx)
    
#     # Validation Set (mixed hits/misses)
#     val_labels = rng.binomial(1, 0.7, size=n_val_samples)
#     val_scores = np.where(val_labels == 1, rng.normal(15, 5, n_val_samples), rng.normal(-5, 8, n_val_samples))
    
#     # Test Set ID Hits
#     test_id_scores = rng.normal(15, 5, n_test_id_hits)
    
#     # Test Set OOD Samples (varying severity)
#     test_ood_scores_dict = {
#         'CIFAR100 (Near)': rng.normal(-2, 7, n_test_ood),
#         'SuperCIFAR (Mid)': rng.normal(-5, 8, n_test_ood),
#         'SVHN (Far)': np.concatenate([rng.normal(-8, 5, 400), rng.normal(18, 2, 100)]) # Spiky/dangerous OOD
#     }

#     for method in calibration_methods:
#         # [cite_start]Fit 5-fold CV ensembled calibrator [cite: 1, 14, 22]
#         calibrator = Calibrator(method=method, cv=5, random_state=42)
#         calibrator.fit(val_scores, val_labels)
        
#         # Predict and store for ID Hits
#         id_probs_store[method][model_idx, :] = calibrator.predict_proba(test_id_scores)
        
#         # Predict and store for each OOD dataset
#         for ds_name, ood_scores in test_ood_scores_dict.items():
#             ood_probs_store[method][ds_name][model_idx, :] = calibrator.predict_proba(ood_scores)

# --- Execute Plotting Routes ---
# Route 1: Shift Severity
# plot_route_1_severity_stats(id_probs_store, ood_probs_store, alpha=1.0)

# Route 2: Alpha-Robustness (evaluated on the hardest shift)
# plot_route_2_alpha_stats(id_probs_store, ood_probs_store, target_dataset='SVHN (Far)', max_alpha=5.0)

# Route 3: Pareto Trade-off
# plot_route_3_pareto_stats(id_probs_store, ood_probs_store, target_dataset='SVHN (Far)')

# Route 4: Variance Gap (checking Isotonic regression's vulnerability to spikes)
# plot_route_4_variance_stats(id_probs_store, ood_probs_store, method='isotonic', alpha=1.0)
# %%
