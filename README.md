# A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts
Convolutional Models       |  Transformer Models
:-------------------------:|:-------------------------:
<img src="https://github.com/cesar-claros/ood_systematic/blob/main/figs/top_cliques_Conv_False_RC.jpeg">  |  <img src="https://github.com/cesar-claros/ood_systematic/blob/main/figs/top_cliques_ViT_False_RC.jpeg">

## Environment Reproduction
This project relies heavily on FD-Shifts [https://github.com/IML-DKFZ/fd-shifts]. Make sure that you install the forked version of FD-Shifts [https://github.com/cesar-claros/fd-shifts-0.1.1/] that we modified to include TinyImagenent as a source dataset, and also include the required libraries to train and test all the ID/OOD detectection methods shown in the paper. To clone this version execute the following line:
```bash
pip install git+https://github.com/cesar-claros/fd-shifts-0.1.1.git
```

Alternatively, you can install FD-Shifts v0.1.1 [https://github.com/IML-DKFZ/fd-shifts] and ```bayesian-optimization==3.1.0```, ```faiss-cpu==1.9.0```, ```MedPy```, ```tinyimagenet==0.9.9```, and ```torch_pca==1.0.0```, before executing our code. However, support for TinyImagenet as source dataset is not included in this version.


A better alternative is to reproduce the environment used in this project by pulling the following Docker image:
```bash
docker pull cesarclaros/systematic_analysis_ood:cuda11.7
```
This container has all the required depedencies to execute the experiments and our code.


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
```bash
fd_shifts launch --dataset=tiny-imagenet-200
```


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

## Evaluation and Analysis Scripts

The following scripts are refactored to support command-line arguments and logging.

### 1. Data Retrieval
**`retrieve_scores.py`**: Aggregates scores from experiment directories.
```bash
python retrieve_scores.py --dataset cifar10 --output-dir ./scores
```
Arguments:
- `--dataset`: Dataset name (e.g., `cifar10`, `cifar100`, `tinyimagenet`).
- `--output-dir`: Output directory for CSV files.

### 2. CLIP-based Metrics
**`clip_uncertainty.py`**: Computes uncertainty metrics using CLIP features.
```bash
python clip_uncertainty.py --dataset cifar10 --output-dir ./clip_scores
```
Arguments:
- `--dataset`: Dataset name.
- `--output-dir`: Directory to save joblib results.

**`clip_proximity.py`**: Computes proximity metrics using CLIP features.
```bash
python clip_proximity.py --iid_dataset cifar10 --output-dir ./clip_scores
```
Arguments:
- `--iid_dataset`: In-distribution dataset name.
- `--output-dir`: Directory to save output JSON.

**`clip_clustering.py`**: Performs clustering analysis on CLIP features.
```bash
python clip_clustering.py --dataset cifar10 --n-clusters 3 --input-dir ./clip_scores --output-dir ./cluster_results --latex
```
Arguments:
- `--dataset`: Dataset name.
- `--n-clusters`: Number of clusters (default: 3).
- `--input-dir`: Directory containing input JSON files.
- `--output-dir`: Directory for output CSV/Latex.
- `--latex`: Flag to generate LaTeX table format.

### 3. Statistical Evaluation
**`stats_eval.py`**: Performs full statistical evaluation and generates plots.
```bash
python stats_eval.py --mcd --backbone ViT --metric-group RC --output-dir ./ood_eval_outputs
```
Arguments:
- `--mcd`: Enable MCD (Monte Carlo Dropout) entries.
- `--backbone`: Model backbone (`Conv` or `ViT`).
- `--metric-group`: Metric group (`RC` for AUGRC/AURC, `ROC` for AUROC/FPR).
- `--output-dir`: Output directory.

**`stats_eval_demo.py`**: A step-by-step demonstration of the evaluation pipeline saving intermediate results.
```bash
python stats_eval_demo.py --source cifar10 --backbone Conv --metric AURC --group 0 --methods "MSP" "MaxLogit" --output-dir ./demo_outputs
```
Arguments:
- `--source`: Source dataset.
- `--backbone`: Model backbone.
- `--metric`: Specific metric to analyze.
- `--group`: Subgroup filter (e.g., '0' for test).
- `--methods`: List of methods to filter (substring match).
- `--output-dir`: Output directory.

### 4. Neural Collapse Analysis
**`neural_collapse_eval.py`**: Computes and aggregates neural collapse metrics.
```bash
python neural_collapse_eval.py --output-dir neural_collapse_metrics
```
Arguments:
- `--output-dir`: Directory to save metric CSVs.
