
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, List
from fd_shifts import configs
import pytorch_lightning as pl
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import os
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast
from torch.utils.data import DataLoader, Subset
from fd_shifts import logger
import torchvision.transforms as transforms
import torchvision
import pandas as pd
#%%
ArrayType = torch.Tensor
T = TypeVar(
    "T", Callable[[ArrayType], ArrayType], Callable[[ArrayType, ArrayType], ArrayType]
)
#%%
def _update(d, u):
    for k, v in u.items():
        if k=='defaults':
            continue
        if isinstance(v, Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
#%%
def get_study_name(path:str)->str:
    if 'confidnet' in path:
        study_name = 'confidnet'
    elif 'devries' in path:
        study_name = 'devries'
    elif 'dg' in path:
        study_name = 'dg'
    elif 'vit' in path:
        study_name = 'vit'
    else:
        raise NotImplementedError
    return study_name
#%%
def is_dropout_enabled(path:str)->bool:
    if 'do1' in path:
        do_enabled = True
    elif 'do0' in path:
        do_enabled = False
    else:
        raise NotImplementedError
    return do_enabled
#%%

def get_conf(path:str, study_name:str)->configs.Config:
    if study_name=='dg':
        study_name = 'deepgamblers'
    configs.init()
    cfg_ = configs.Config.with_defaults(study=study_name)
    cfg_.trainer.optimizer = configs.SGD()
    if study_name=='vit':
        cfg_.trainer.lr_scheduler = configs.LinearWarmupCosineAnnealingLR()
    else:
        cfg_.trainer.lr_scheduler = configs.CosineAnnealingLR()
    cfg_.__pydantic_validate_values__()
    path_file = '/work/cniel/sw/FD_Shifts/project/experiments/'+path
    cfg_1 = OmegaConf.load(path_file + '/hydra/config.yaml')
    cfg_2 = OmegaConf.load(path_file + '/hydra/hydra.yaml')
    cfg_1.exp.work_dir = cfg_2.hydra.runtime.cwd
    base1_ = _update(DictConfig(cfg_), cfg_1)
    cfg_1 = OmegaConf.to_object(base1_)
    cf: configs.Config = cast(configs.Config, cfg_1)
    return cf
# %%
def get_model_and_last_layer(module: type[pl.LightningModule],
                                study_name:str, return_model=True):
    if study_name == 'confidnet':
        model = module.backbone
    elif (study_name == 'devries') or (study_name == 'dg') or (study_name == 'vit') :
        model = module.model
    else:
        raise NotImplementedError

    if study_name == 'vit':
        classifier_state = model.head.state_dict()
    else:
        classifier_state = model.classifier.state_dict()

    if "module.weight" in classifier_state:
        w = classifier_state["module.weight"]
        b = classifier_state["module.bias"]
    elif "fc.weight" in classifier_state:
        w = classifier_state["fc.weight"]
        b = classifier_state["fc.bias"]
    elif "fc2.weight" in classifier_state:
        w = classifier_state["fc2.weight"]
        b = classifier_state["fc2.bias"]
    elif "weight" in classifier_state:
        w = classifier_state["weight"]
        b = classifier_state["bias"]    
    else:
        print(list(classifier_state.keys()))
        raise RuntimeError("No classifier weights found")
    if return_model:
        return model, w, b
    else:
        return w, b
#%%
def save_data(cf, data:ArrayType, path:str|None=None, filename:str='data'):
    # assert self.M is not None, 'M has not been computed'
    # assert self.gamma is not None, 'gamma has not been computed'
    # params_dict = {
    #             'M': self.M,
    #             'gamma': self.gamma,
    #             }
    if path is None:
        if os.path.exists(f'{cf.exp.dir}/eval'):
            path = f'{cf.exp.dir}/eval/{filename}.pt'
        else:
            os.mkdir(f'{cf.exp.dir}/eval')
            path = f'{cf.exp.dir}/eval/{filename}.pt'
    else:
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        path = f'{path}/{filename}.pt'
    torch.save(data, path)

#%%
def load_data(cf, path:str|None=None, filename:str='data') -> ArrayType:
    if path is None:
        path = f'{cf.exp.dir}/eval/{filename}.pt'
    else:
        path = f'{path}/{filename}.pt'
    assert os.path.exists(path), f'Specified path {path} does not exist..'
    data = torch.load(path)
    return data
    # self.M = params_dict['M']
    # self.gamma = params_dict['gamma']
#%%
def sampling_corrupt_cifar(n_intensities:int=5, n_corruptions:int=15, n_test:int = 10000, sampling_rate:float=0.2):
    indices_list = []
    samples_per_intensity_level = n_corruptions*n_test
    for i in range(n_intensities):
        subset_size = int((samples_per_intensity_level)*sampling_rate)
        indices = torch.randperm(samples_per_intensity_level)[:subset_size]+samples_per_intensity_level*i
        indices_list.append(indices)
    subset_indices = torch.cat(indices_list)
    return subset_indices
#%%
def compute_model_evaluations(model, datamodule, set_name:str) :
    # model_evaluations = {}
    if model.study_name=='vit':
        resize_img = (384,384)
    elif model.dataset=='tiny-imagenet-200':
        resize_img = (64,64)
    else:
        resize_img = (32,32)
    set_name = set_name.split('_')
    if set_name[0]=='train':
        dataloaders = datamodule.train_dataloader()
    elif set_name[0]=='val':
        dataloaders = datamodule.val_dataloader()
    elif set_name[0] == 'test' : # test_n
        test_set = int(set_name[1])
        if test_set <= 5:
            dataloaders = datamodule.test_dataloader()[test_set]
            if len(set_name)>2:
                if set_name[2] == 'sampled':
                    sampling_rate = int(set_name[3])
                    batch_size = dataloaders.batch_size
                    num_workers = dataloaders.num_workers
                    if 'corrupt' in dataloaders.dataset.root:
                        logger.info(f'Sampling {sampling_rate}% of the original dataset size...')
                        subset_indices = sampling_corrupt_cifar(sampling_rate=sampling_rate/100)
                        data_subset = Subset(dataloaders.dataset, subset_indices)
                        dataloaders = DataLoader(data_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    else:
                        raise Exception("Subsampling only defined for Corrupt CIFAR...")
        elif test_set == 6: # LSUN cropped
            transform = transforms.Compose([
                                        transforms.Resize(resize_img),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            dataset = torchvision.datasets.ImageFolder(datamodule.data_root_dir.joinpath('LSUN'), transform=transform)
            dataloaders = DataLoader(dataset, batch_size=datamodule.batch_size, num_workers= datamodule.num_workers, shuffle=False)
        elif test_set == 7: # LSUN resize
            transform = transforms.Compose([
                                        transforms.Resize(resize_img),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            dataset = torchvision.datasets.ImageFolder(datamodule.data_root_dir.joinpath('LSUN_resize'), transform=transform)
            dataloaders = DataLoader(dataset, batch_size=datamodule.batch_size, num_workers= datamodule.num_workers, shuffle=False)
        elif test_set == 8: # iSUN
            transform = transforms.Compose([
                                        transforms.Resize(resize_img),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            dataset = torchvision.datasets.ImageFolder(datamodule.data_root_dir.joinpath('iSUN'), transform=transform)
            dataloaders = DataLoader(dataset, batch_size=datamodule.batch_size, num_workers= datamodule.num_workers, shuffle=False)
        elif test_set == 9: # Textures
            transform = transforms.Compose([
                                        transforms.Resize(resize_img),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            dataset = torchvision.datasets.ImageFolder(datamodule.data_root_dir.joinpath('dtd','images'), transform=transform)
            dataloaders = DataLoader(dataset, batch_size=datamodule.batch_size, num_workers= datamodule.num_workers, shuffle=False)
        elif test_set == 10: # Places365
            transform = transforms.Compose([
                                        transforms.Resize(resize_img),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
            dataset = torchvision.datasets.ImageFolder(datamodule.data_root_dir.joinpath('places365'), transform=transform)
            dataloaders = DataLoader(dataset, batch_size=datamodule.batch_size, num_workers= datamodule.num_workers, shuffle=False)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # Compute model evaluations
    results_list_set = [ model(batch,i) for i,batch in enumerate(tqdm(dataloaders, position=0, leave=True)) ]
    # results_list_set.append(  )
    key_set = {tuple(d.keys()) for d in results_list_set}
    # Check if all dictionaries in results_list have the same keys
    assert len(key_set)==1, 'Dictionaries in the results list have different keys'
    keys_names = list(*key_set)
    # Concatenate all evaluations 
    results_concat = {}
    for k in keys_names:
        try:
            arr_ = torch.concat([d[k] for d in results_list_set])
        except:
            arr_ = None
        results_concat[k] = arr_
    
    return results_concat

#%%
# def best_params_entropy(confid_func, softmax, residuals, neg_sign:bool=False):
#     param_grid = {
#                     'M': np.arange(1, n_classes+1, n_classes//10),
#                     'gamma': np.arange(0.1, 2.0, 0.1), }
#     grid = ParameterGrid(param_grid)
#     scores_entropy_val = []
#     for params in grid:
#         # print(params['gamma'])
#         if params['gamma'] == 1.0:
#             continue
#         confidence = confid_func(softmax, gamma=params['gamma'], M=params['M'])
#         if neg_sign:
#             confidence = -1*confidence 
#         stats_val_ = RiskCoverageStats(confids =  confidence, residuals = residuals)
#         scores_entropy_val.append( {'augrc' : stats_val_.augrc,
#                                     'aurc' : stats_val_.aurc,    
#                                     'M': params['M'],
#                                     'gamma' : params['gamma']} )
#     ent_df = pd.DataFrame(scores_entropy_val)
#     return ent_df
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
def extract_char_after_substring(main_string, search_string):
    """
    Finds a specific string within a main string and extracts the character
    immediately following it.

    Args:
        main_string (str): The string to search within.
        search_string (str): The string to search for.

    Returns:
        str: The character immediately after the search_string, or None if
             the search_string is not found or is at the end of the main_string.
    """
    index = main_string.find(search_string)

    if index != -1:  # If the search_string is found
        # Calculate the index of the character after the search_string
        char_after_index = index + len(search_string)
        
        # Check if there is a character at that index (i.e., not at the end)
        if char_after_index < len(main_string):
            return main_string[char_after_index:].split(search_string)[0].split('_')[0]
        else:
            return None  # Search string is at the end of the main string
    else:
        return None  # Search string not found