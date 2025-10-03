# A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts
This project relies heavily on FD-Shifts [https://github.com/IML-DKFZ/fd-shifts]. Make sure that you installed FD-Shifts v0.1.1, before executing our code

## Usage
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
python cifar_iid_train.py --model_path=cifar10_paper_sweep/dg_bbvgg13_do0_run1_rew2.2 --no-rank_weight --no-rank_feature --ash=None --use_cuda --temperature_scale --test_mode=ood_nsncs_svhn
```
