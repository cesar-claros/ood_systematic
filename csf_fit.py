#%%
import torch
from fd_shifts.utils import exp_utils
from fd_shifts.models import get_model
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
import argparse
import pandas as pd
from torch.nn import functional as F
from src import utils
from src import csf_pipeline
from src.trained_module import TrainedModule
from src.csfs.temperature_scaling import TemperatureScaling
from fd_shifts import logger
#%%
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Fit CSFs on validation/training splits")
    # Add an argument
    parser.add_argument('--model_path', type=str, required=True, help="Path of folder where experiment is found")
    parser.add_argument('--rank_weight', required=True, action=argparse.BooleanOptionalAction, help="Adding RankWeight functionality to model")
    parser.add_argument('--rank_feature', required=True, action=argparse.BooleanOptionalAction, help="Adding RankFeature functionality to model")
    parser.add_argument('--ash', type=str, required=True, help="Adding ASH functionality to model", default='None')
    parser.add_argument('--use_cuda', required=True, action=argparse.BooleanOptionalAction, help="Adding RankFeature functionality to model")
    parser.add_argument('--temperature_scale', required=True, action=argparse.BooleanOptionalAction, help="Carry operations out using temperature scaling")
    csf_group = parser.add_mutually_exclusive_group()
    csf_group.add_argument('--csfs', type=str, default=None,
                           help="Comma-separated CSF families to fit (default: all). E.g. 'KernelPCA,Mahalanobis'.")
    csf_group.add_argument('--skip-csfs', dest='skip_csfs', type=str, default=None,
                           help="Comma-separated CSF families to skip (default: none). E.g. 'KernelPCA'.")
    parser.add_argument('--projections', type=str, default='plain,global,class,class_pred',
                        help=("Comma-separated projection modes to fit "
                              "(default: 'plain,global,class,class_pred'). 'plain' = CSFs operating "
                              "on raw features/logits (no PF); 'global', 'class', 'class_pred' = CSFs "
                              "operating on ProjectionFiltering outputs. Use 'none' to skip all "
                              "(only the raw Temperature is fit)."))
    # Parse the arguments
    args = parser.parse_args()
    path = args.model_path
    rank_weight_opt = args.rank_weight
    rank_feat_opt = args.rank_feature
    ash_method_opt = args.ash # 'ash_s@90'
    use_cuda_opt = args.use_cuda
    temperature_scale_opt = args.temperature_scale
    csfs_arg = [s.strip() for s in args.csfs.split(',')] if args.csfs else None
    skip_csfs_arg = [s.strip() for s in args.skip_csfs.split(',')] if args.skip_csfs else None
    active = csf_pipeline.build_active(csfs=csfs_arg, skip_csfs=skip_csfs_arg)
    logger.info(f"Active CSF families: {sorted(active)}")
    if args.projections.strip().lower() in ('none', ''):
        projections = set()
    else:
        projections = {s.strip() for s in args.projections.split(',') if s.strip()}
    logger.info(f"Projections to fit: {sorted(projections) if projections else 'none'}")

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
            if 'vgg' in path:    
                print("Disabling average pooling for VGG-13 supercifar experiments with dropout enabled...")
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
            if 'tiny' in path:
                new_batch_size = cf.trainer.batch_size//8
            else:    
                new_batch_size = cf.trainer.batch_size//2
            logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
            cf.trainer.batch_size = new_batch_size
        elif (study_name=='confidnet'):
            if 'tiny' in path:
                new_batch_size = cf.trainer.batch_size//16
            else:    
                new_batch_size = cf.trainer.batch_size//4
            logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
            cf.trainer.batch_size = new_batch_size
    
    if study_name=='vit' and use_cuda_opt and not do_enabled:
        new_batch_size = cf.trainer.batch_size*2
        logger.info(f'Changing the batch size from {cf.trainer.batch_size} to {new_batch_size}...')
        cf.trainer.batch_size = new_batch_size
    if study_name=='vit' and use_cuda_opt and do_enabled:
        new_batch_size = cf.trainer.batch_size*2
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
    model = TrainedModule(module, study_name, cf,
                                        rank_weight=rank_weight_opt, 
                                        rank_feat=rank_feat_opt, 
                                        ash_method=ash_method_opt, 
                                        use_cuda=use_cuda_opt)
    # Compute evaluations
    model_opts = f'_RW{int(rank_weight_opt)}_RF{int(rank_feat_opt)}_ASH{str(ash_method_opt)}'
    model_evaluations = {}
    iid_datasets = ['val','train']
    for set_name in iid_datasets:
        # try:
        #     print(f'Loading data from {set_name} dataset...')
        #     model_eval = utils.load_data(cf, filename=set_name)     
        # except:
        logger.info(f'Evaluating model with {set_name} dataset...')
        model_eval = utils.compute_model_evaluations(model, datamodule, set_name=set_name)
        # utils.save_data(cf, model_eval, filename=set_name)
        # Compute (or load) temperature scale.
        if set_name == 'val':
            temperature_scale = csf_pipeline._load_or_fit_temperature(
                cf, model_opts, suffix=None,
                logits=model_eval['logits'], labels=model_eval['labels'],
            )
            if do_enabled:
                temperature_scale_dist = csf_pipeline._load_or_fit_temperature(
                    cf, model_opts, suffix='distribution',
                    logits=model_eval['logits_dist'].mean(dim=2), labels=model_eval['labels'],
                )
        model_eval['softmax'] = F.softmax(model_eval['logits'], dim=1, dtype=torch.float64)
        model_eval['softmax_scaled'] = temperature_scale.get_scaled_softmax(model_eval['logits'])
        model_eval['correct'] = (model_eval['softmax'].max(dim=1).indices == model_eval['labels']).long()
        if do_enabled:
            model_eval['softmax_dist'] = F.softmax(model_eval['logits_dist'], dim=1, dtype=torch.float64)
            model_eval['softmax_scaled_dist'] = temperature_scale_dist.get_scaled_softmax(model_eval['logits_dist'])
            model_eval['correct_mcd'] = (model_eval['softmax_dist'].mean(dim=2).max(dim=1).indices == model_eval['labels']).long()
        model_evaluations.update({set_name:model_eval})
    # Fit (or load) ProjectionFiltering and projection-specific Temperature variants.
    csf_pipeline.fit_projections(cf, module, study_name, model_evaluations, do_enabled,
                                 model_opts, temperature_scale_opt, projections)
    # Compute score methods
    csf_pipeline.run_score_methods(cf, module, study_name, model_evaluations, do_enabled, model_opts=model_opts, temp_scaled=temperature_scale_opt, active=active, projections=projections)

    eval_name = 'iid_val'
    # Evaluate score methods and fucntions
    csf_pipeline.compute_metrics(module, study_name, cf, model_evaluations, eval_name, do_enabled, model_opts=model_opts, n_bins=20, temp_scaled=temperature_scale_opt, active=active, projections=projections)

if __name__ == "__main__":
    main()