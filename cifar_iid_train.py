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
#%%
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    # Add an argument
    parser.add_argument('--model_path', type=str, required=True, help="Path of folder where experiment is found")
    parser.add_argument('--rank_weight', required=True, action=argparse.BooleanOptionalAction, help="Adding RankWeight functionality to model")
    parser.add_argument('--rank_feature', required=True, action=argparse.BooleanOptionalAction, help="Adding RankFeature functionality to model")
    parser.add_argument('--ash', type=str, required=True, help="Adding ASH functionality to model", default='None')
    parser.add_argument('--use_cuda', required=True, action=argparse.BooleanOptionalAction, help="Adding RankFeature functionality to model")
    parser.add_argument('--temperature_scale', required=True, action=argparse.BooleanOptionalAction, help="Carry operations out using temperature scaling")
    # parser.add_argument('--temperature_scale', required=True, action=argparse.BooleanOptionalAction, help="Carry operations out using temperature scaling")
    # parser.add_argument('--test_mode', required=True, type=str, help="Sets considered for evaluation", default='None', choices=['val','iid_test','noise_study'])
    # Parse the arguments
    args = parser.parse_args()
    path = args.model_path
    rank_weight_opt = args.rank_weight
    rank_feat_opt = args.rank_feature
    ash_method_opt = args.ash # 'ash_s@90'
    use_cuda_opt = args.use_cuda
    temperature_scale_opt = args.temperature_scale

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
    model = scores_methods.TrainedModule(module, study_name, cf, 
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
        # Compute temperature scale
        if set_name == 'val':
            temperature_scale = scores_methods.TemperatureScaling(cf)
            temperature_scale.compute_temperature(model_eval['logits'], model_eval['labels'])
            temperature_scale.save_params(filename='Temperature_params'+model_opts)
            if do_enabled:
                temperature_scale_dist = scores_methods.TemperatureScaling(cf)
                temperature_scale_dist.compute_temperature(model_eval['logits_dist'].mean(dim=2), model_eval['labels'])
                temperature_scale_dist.save_params(filename='Temperature_distribution_params'+model_opts)
        model_eval['softmax'] = F.softmax(model_eval['logits'], dim=1, dtype=torch.float64)
        model_eval['softmax_scaled'] = temperature_scale.get_scaled_softmax(model_eval['logits'])
        model_eval['correct'] = (model_eval['softmax'].max(dim=1).indices == model_eval['labels']).long()
        if do_enabled:
            model_eval['softmax_dist'] = F.softmax(model_eval['logits_dist'], dim=1, dtype=torch.float64)
            model_eval['softmax_scaled_dist'] = temperature_scale_dist.get_scaled_softmax(model_eval['logits_dist'])
            model_eval['correct_mcd'] = (model_eval['softmax_dist'].mean(dim=2).max(dim=1).indices == model_eval['labels']).long()
        model_evaluations.update({set_name:model_eval})
    # Compute score methods
    utils_funcs.run_score_methods(cf, module, study_name, model_evaluations, do_enabled, model_opts=model_opts, temp_scaled=temperature_scale_opt)
    
    eval_name = 'iid_val'
    # Evaluate score methods and fucntions
    utils_funcs.compute_metrics(module, study_name, cf, model_evaluations, eval_name, do_enabled, model_opts=model_opts, n_bins=20, temp_scaled=temperature_scale_opt)

if __name__ == "__main__":
    main()