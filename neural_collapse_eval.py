#%%
import os
import argparse
import pandas as pd
import torch
from loguru import logger
from fd_shifts.utils import exp_utils
from fd_shifts.models import get_model
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
from torch.nn import functional as F
from src import scores_methods
import numpy as np
from src.utils import get_study_name, is_dropout_enabled, get_conf, extract_char_after_substring

# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')

def main():
    parser = argparse.ArgumentParser(description="Neural Collapse Evaluation")
    parser.add_argument("--output-dir", type=str, default="neural_collapse_metrics", help="Directory to save output metrics")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

    # File paths
    nc_metrics_path = os.path.join(output_dir, 'nc_metrics.csv')
    nc_metrics_global_path = os.path.join(output_dir, 'nc_metrics_global.csv')
    nc_metrics_class_pred_path = os.path.join(output_dir, 'nc_metrics_class_pred.csv')

    # Check if files exist to skip computation
    if os.path.exists(nc_metrics_path) and os.path.exists(nc_metrics_global_path) and os.path.exists(nc_metrics_class_pred_path):
        logger.info("Found existing metric files. Loading them...")
        df_metrics = pd.read_csv(nc_metrics_path, index_col=0)
        df_metrics_global = pd.read_csv(nc_metrics_global_path, index_col=0)
        df_metrics_class_pred = pd.read_csv(nc_metrics_class_pred_path, index_col=0)
    else:
        logger.info("Metric files not found. Starting computation...")
        nc_metrics_list = []
        nc_metrics_global_list = []
        nc_metrics_class_pred_list = []

        datasets = ['cifar10','supercifar','cifar100','tinyimagenet']
        
        for dataset in datasets:
            config_file = f'configs_exp/configs_{dataset}_iid_train.txt'
            if not os.path.exists(config_file):
                logger.warning(f"Config file {config_file} not found. Skipping dataset {dataset}.")
                continue
            
            logger.info(f"Processing dataset: {dataset}")
            
            with open(config_file, 'r') as f:
                for k, cfg_line in enumerate(f):
                    cfg_line_split = cfg_line.split().copy()
                    
                    if len(cfg_line_split) < 7:
                         continue

                    path = cfg_line_split[1]
                    rank_weight_opt = False if 'no' in cfg_line_split[2] else True
                    rank_feat_opt = False if 'no' in cfg_line_split[3] else True
                    ash_method_opt = cfg_line_split[4]
                    use_cuda_opt = False if 'no' in cfg_line_split[5] else True
                    temperature_scale_opt = False if 'no' in cfg_line_split[6] else True

                    if rank_weight_opt: continue
                    if rank_feat_opt: continue
                    if ash_method_opt != 'None': continue

                    cuda_available = torch.cuda.is_available()
                    if not (cuda_available and use_cuda_opt):
                        use_cuda_opt = False
                    else:
                        pass 

                    if ash_method_opt == 'None':
                        ash_method_opt = None

                    study_name = get_study_name(path)
                    do_enabled = is_dropout_enabled(path)
                    
                    try:
                        cf = get_conf(path, study_name)
                        ckpt_path = exp_utils._get_path_to_best_ckpt(cf.exp.dir, 'last', cf.test.selection_mode)
                    except Exception as e:
                        logger.warning(f"Failed to load config or checkpoint for {path}: {e}")
                        continue

                    if 'super' in path:
                        cf.eval.query_studies.noise_study = ['corrupt_cifar100']
                        cf.eval.query_studies.new_class_study = ['cifar10', 'svhn', 'tinyimagenet_resize']
                        if do_enabled:
                            cf.model.avg_pool = False
                    if 'vit' in path:
                        cf.data.num_workers = 12

                    # Load module
                    try:
                        # Ensure we get the correct model class
                        # Note: get_model returns the class, not instance
                        ModelClass = get_model(cf.model.name)
                        module = ModelClass(cf)
                        module.load_only_state_dict(ckpt_path, device='cpu')
                    except Exception as e:
                        logger.error(f"Error loading model from {ckpt_path}: {e}")
                        continue

                    if study_name == 'confidnet':
                        module.backbone.encoder.disable_dropout()
                        module.network.encoder.disable_dropout()
                    elif (study_name == 'devries') or (study_name == 'dg'):
                        module.model.encoder.disable_dropout()
                    elif study_name == 'vit':
                        module.disable_dropout()
                    
                    # Batch size adjustments
                    if do_enabled and use_cuda_opt:
                        if (study_name=='devries' or study_name=='dg'):
                            cf.trainer.batch_size //= 2
               
                        elif (study_name=='confidnet'):
                            cf.trainer.batch_size //= 4
           
                    if study_name=='vit':
                         if use_cuda_opt and not do_enabled:
                            cf.trainer.batch_size //= 2
                         elif use_cuda_opt and do_enabled:
                            cf.trainer.batch_size //= 2
                            confids_test = cf.eval.confidence_measures.test
                            cf.eval.confidence_measures.test = [i for i in confids_test if 'mcd' not in i]
                         elif not use_cuda_opt:
                            cf.trainer.batch_size = 128
                    
                    logger.info(f"Processing: {study_name}, DO={do_enabled}, Path={path}")

                    # Load datasets
                    datamodule = FDShiftsDataLoader(cf)
                    datamodule.setup()
                    
                    # Instantiate model wrapper
                    # Assuming scores_methods is imported and available
                    try:                        
                        # Compute evaluations
                        model_opts = f'_RW{int(rank_weight_opt)}_RF{int(rank_feat_opt)}_ASH{str(ash_method_opt)}'
                        
                        run_val = int(extract_char_after_substring(path,'run')) if 'run' in path else 0
                        rew_val = float(extract_char_after_substring(path,'rew')) if 'rew' in path else 0.0
                        lr_val = extract_char_after_substring(path,'lr') if 'lr' in path else '0.0'

                        model_cfgs = {
                            'dataset': dataset,
                            'architecture' : 'VGG13' if study_name!='vit' else 'ViT',
                            'study': study_name,
                            'dropout': do_enabled,
                            'run': run_val, 
                            'reward': rew_val, 
                            'lr': lr_val,
                        }

                        # Function to process and append metrics
                        def process_nc(params_prefix, target_list):
                            nc_eval = scores_methods.NeuralCollapseMetrics(module, study_name, cf)
                            nc_eval.load_params(filename=params_prefix + model_opts)
                            metrics = {k: nc_eval.nc_metrics[k].item() for k in nc_eval.nc_metrics}
                            metrics.update(model_cfgs)
                            target_list.append(metrics)

                        process_nc('NeuralCollapse_params', nc_metrics_list)
                        process_nc('NeuralCollapse_global_params', nc_metrics_global_list)
                        process_nc('NeuralCollapse_class_pred_params', nc_metrics_class_pred_list)
                        
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        continue

        df_metrics = pd.DataFrame(nc_metrics_list)
        df_metrics_global = pd.DataFrame(nc_metrics_global_list)
        df_metrics_class_pred = pd.DataFrame(nc_metrics_class_pred_list)

        df_metrics.to_csv(nc_metrics_path)
        df_metrics_global.to_csv(nc_metrics_global_path)
        df_metrics_class_pred.to_csv(nc_metrics_class_pred_path)
        logger.success(f"Saved raw metrics to {output_dir}")

    # Post-processing
    logger.info("Computing grouped averages...")
    drop_cols = ['lr','reward','run','study']
    # Ensure these columns exist before dropping
    drop_cols = [c for c in drop_cols if c in df_metrics.columns]
    
    means_cols = ['equiangular_uc','equinorm_uc','max_equiangular_uc','self_duality','var_collapse','cdnv_score','bias_collapse','M_etf_diff','wM_etf_diff']
    weights_cols = ['equiangular_wc','equinorm_wc','max_equiangular_wc','w_etf_diff']
    group_list = ['dropout','architecture','dataset']
    
    # Filter columns that actually exist in the dataframes
    means_cols = [c for c in means_cols if c in df_metrics.columns]
    weights_cols = [c for c in weights_cols if c in df_metrics.columns]

    def safe_groupby(df, val_cols, grp_cols, name):
        if df.empty:
            logger.warning(f"Dataframe for {name} is empty. Skipping groupby.")
            return
        
        # Verify columns exist
        available_val_cols = [c for c in val_cols if c in df.columns]
        available_grp_cols = [c for c in grp_cols if c in df.columns]
        
        if not available_val_cols or not available_grp_cols:
             logger.warning(f"Missing columns for {name}. skipping.")
             return

        try:
            grouped = df.drop(drop_cols, axis=1, errors='ignore')[available_val_cols + available_grp_cols].groupby(available_grp_cols).mean()
            out_path = os.path.join(output_dir, f'{name}.csv')
            grouped.to_csv(out_path)
            logger.info(f"Saved {name} to {out_path}")
        except Exception as e:
            logger.error(f"Error grouping {name}: {e}")

    safe_groupby(df_metrics, weights_cols, group_list, 'nc_metrics_weights_grouped')
    safe_groupby(df_metrics, means_cols, group_list, 'nc_metrics_grouped')
    safe_groupby(df_metrics_global, means_cols, group_list, 'nc_metrics_global_grouped')
    safe_groupby(df_metrics_class_pred, means_cols, group_list, 'nc_metrics_class_pred_grouped')

if __name__ == "__main__":
    main()

