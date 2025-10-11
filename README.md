# A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts
Convolutional Models       |  Transformer Models
:-------------------------:|:-------------------------:
<img src="https://github.com/cesar-claros/ood_systematic/blob/main/figs/top_cliques_Conv_False_%5B'AUGRC'%2C%20'AURC'%5D_1.jpeg">  |  <img src="https://github.com/cesar-claros/ood_systematic/blob/main/figs/top_cliques_ViT_False_%5B'AUGRC'%2C%20'AURC'%5D_1.jpeg">


This project relies heavily on FD-Shifts [https://github.com/IML-DKFZ/fd-shifts]. Make sure that you install the forked version of FD-Shifts [https://github.com/cesar-claros/fd-shifts-0.1.1/] that we modified to include TinyImagenent as a source dataset, and also include the required libraries to train and test all the ID/OOD detectection methods shown in the paper. To clone this version execute the following line:
```bash
pip install git+https://github.com/cesar-claros/fd-shifts-0.1.1.git
```

Alternatively, you can install FD-Shifts v0.1.1 [https://github.com/IML-DKFZ/fd-shifts] and ```bayesian-optimization==3.1.0```, ```faiss-cpu==1.9.0```, ```MedPy```, ```tinyimagenet==0.9.9```, and ```torch_pca==1.0.0```, before executing our code. However, support for TinyImagenet as source dataset is not included in this version.

### Verify TinyImagenet as source dataset
Once FD-Shifts [https://github.com/cesar-claros/fd-shifts-0.1.1/] has been installed, you can check if the experiments that use TinyImagenet are included by executing the following line:
```bash
fd_shifts list 
```
You should be able to see the experiments that use TinyImagenet in the output
```bash
...
fd-shifts/tiny-imagenet-200_paper_sweep/devries_bbvgg13_do0_run1_rew2.2
fd-shifts/tiny-imagenet-200_paper_sweep/devries_bbvgg13_do0_run2_rew2.2
fd-shifts/tiny-imagenet-200_paper_sweep/devries_bbvgg13_do0_run3_rew2.2
fd-shifts/tiny-imagenet-200_paper_sweep/devries_bbvgg13_do0_run4_rew2.2
...
```

### Train models associated to TinyImagenet
You can train all models the models that use TinyImagenet by executing  

## Usage
### Clone code
To  clone the code into ```project``` folder, execute the following line:
```bash
git clone https://github.com/cesar-claros/ood_systematic.git project
```

### Data folder requirements
Follow instructions for data folder structure according to [https://github.com/IML-DKFZ/fd-shifts/blob/v0.1.1/docs/datasets.md]. Additionally, download the extra OOD datasets used in this work from the following link:
- [OOD datasets](https://zenodo.org/records/17317862)
into ```$DATASET_ROOT_DIR```. This file contains datasets commonly used for OOD detection evaluation, which include Textures, Places365, iSUN, LSUN, and LSUN resize.

### Download trained models' weights
Download the model weights following instructions described in the FD-Shifts project [https://github.com/IML-DKFZ/fd-shifts]. In addition to the available trained models, we trained a set of models that use TinyImagenet. These trained models can be downloaded from be following link:
- [TinyImagenet](https://zenodo.org/records/17316185)

### Train Confidence Score Functions
To train all Confidence Score Functions (CSFs) for a given trained model, execute the following line: 
```bash
python cifar_iid_train.py --model_path=$model_path $rank_weight_opt $rank_feature_opt --ash=$ash $cuda_opt $temp_opt
```
where ```$model_path``` is the name of the trained model, and ```$rank_weight_opt```, ```$rank_feature_opt``` and ```$ash``` enable RankWeight, RankFeat and/or ASH at inference time. ```$cuda_opt``` enables GPU to be used during training and ```$temp_out``` determines if Temperature scaling will be used for all logits or not.  
For example, if you want to train the CFS for first run of the Deep Gamblers model trained on CIFAR-10 with no RankWeight, RankFeat and/or ASH at inference time and with GPU and Temperature scaling, execute the following line:
```bash
python cifar_iid_train.py --model_path=cifar10_paper_sweep/dg_bbvgg13_do0_run1_rew2.2 --no-rank_weight --no-rank_feature --ash=None --use_cuda --temperature_scale
```
### Test Confidence Score Functions
Once the CFSs are trained, you can evaluate them executing the following line:
```bash
python cifar_test.py --model_path=$model_path $rank_weight_opt $rank_feature_opt --ash=$ash $cuda_opt $temp_opt --test_mode=$test_opt
```
where ```--test_mode=$test_opt``` indicates the dataset that will be used to evaluate the trained CFS. Available options are ```iid_test```, ```ood_sncs_c100```, ```ood_nsncs_svhn```, ```ood_nsncs_ti```, ```ood_nsncs_ti```, ```ood_nsncs_lsun_cropped```, ```ood_nsncs_lsun_resize```, ```ood_nsncs_isun```, ```ood_nsncs_textures```, ```ood_nsncs_places365```.
For example, if you want to test the trained CFS for first run of the Deep Gamblers model trained on CIFAR-10 with no RankWeight, RankFeat and/or ASH at inference time and with GPU and Temperature scaling, using the SVHN dataset, execute the following line:
```bash
python cifar_test.py --model_path=cifar10_paper_sweep/dg_bbvgg13_do0_run1_rew2.2 --no-rank_weight --no-rank_feature --ash=None --use_cuda --temperature_scale --test_mode=ood_nsncs_svhn
```
