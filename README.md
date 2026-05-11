# A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts

This repository contains the code and analysis pipeline for *A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts*. The paper studies how the geometry of a learned representation (measured via Neural Collapse metrics) governs which Confidence Score Function (CSF) is competitive for OOD detection on a given trained classifier, and proposes a per-CSF logistic predictor that recommends a competitive shortlist of detectors for an unseen model without OOD validation data.

The repository covers the full pipeline from FD-Shifts pretrained checkpoints through CSF training and evaluation, statistical analysis, Neural Collapse computation, CLIP-based OOD stratification, and the cross-architecture predictor used in Section 4.4. Reproducing every paper figure end to end takes one to two weeks of wall-clock time on a small GPU cluster (Appendix C). For reproducibility purposes, the heaviest intermediate artifacts are also published as zip archives (see [Quick reproduction with prebuilt archives](#quick-reproduction-with-prebuilt-archives)).

### Note on shared artifacts with our AISTATS 2026 paper

Two scripts in this repository are shared with our AISTATS 2026 Calibration Workshop companion paper *Bounding Worst-Case Calibration Error in OOD Detection Under Distribution Shift* ([OpenReview](https://openreview.net/forum?id=aZnAoyzssI)):

- **`calibration_ood.py`, `recompute_metric.py`, and the `scores_calibration/` directory.** These are AISTATS-only artifacts. They are listed in the repository layout for completeness, but the pipeline does not invoke them and does not read from `scores_calibration/`. You can ignore them entirely if you only want to reproduce this paper.
- **`stats_eval.py`** is also used by the AISTATS paper (with `--metric-group CE` or `--metric-group CE_BOUND`) to compute calibration top cliques. The paper only uses `--metric-group RC` (AURC / AUGRC), which is the only mode whose outputs feed Figures 1 and 3.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Environment setup](#environment-setup)
3. [Data folder requirements](#data-folder-requirements)
4. [Trained model checkpoints](#trained-model-checkpoints)
5. [Quick reproduction with prebuilt archives](#quick-reproduction-with-prebuilt-archives)
6. [Full pipeline reproduction](#full-pipeline-reproduction)
7. [Expected folder structure after the full pipeline](#expected-folder-structure-after-the-full-pipeline)
8. [Mapping paper figures and tables to scripts](#mapping-paper-figures-and-tables-to-scripts)
9. [Citing FD-Shifts and external datasets](#citing-fd-shifts-and-external-datasets)

---

## Repository layout

```
ood_systematic/
|-- src/                              # supporting modules (CSF implementations, RC stats, calibration)
|-- nc_csf_predictivity/              # NC-based predictor sub-package (Section 4.4 / Appendix H)
|   |-- data/                         # build/harmonize/split the per-cell label matrices
|   |-- evaluation/                   # baselines, regret, splits, and figure scripts
|   |-- ablations/calibration_features_clique.py  # the headline predictor
|   |-- stats/                        # Holm-Wilcoxon tests
|   `-- outputs/                      # generated parquets and figures (regenerable)
|
|-- cifar_iid_train.py                # Stage 1: train CSFs on the validation split
|-- cifar_test.py                     # Stage 2: evaluate CSFs on each OOD dataset
|-- retrieve_scores.py                # Stage 3: aggregate per-cell scores into CSVs (run twice; once with --fix-config)
|-- neural_collapse_eval.py           # Stage 4: compute Papyan NC metrics
|-- clip_proximity.py                 # Stage 5a: CLIP feature distances per dataset
|-- clip_clustering.py                # Stage 5b: k-means near/mid/far stratification
|-- clip_clustering_all_backbones.py  # Stage 5c: stratification across CLIP backbones
|-- clip_robustness.py                # Stage 5d: clustering robustness analysis
|-- generate_openood_grouping.py      # Stage 5e: OpenOOD-style binary grouping
|-- stats_eval.py                     # Stage 6a: top-clique pipeline (Figures 1, 3) — also used by AISTATS 2026 with --metric-group CE / CE_BOUND
|-- stats_eval_demo.py                # Stage 6b: worked example for Appendix E
|-- mantel_analysis.py                # Stage 6c: Mantel test (Appendix G)
|-- projection_filtering_analysis.py  # Stage 6d: paired AUGRC tables (Appendix F)
|-- projection_clique_analysis.py     # Stage 6e: clique substitution (Appendix F)
|-- calibration_ood.py                # AISTATS 2026 only — not part of this pipeline
|-- recompute_metric.py               # AISTATS 2026 only — not part of this pipeline
`-- README.md
```


---

## Environment setup

This project depends on a forked version of FD-Shifts that adds TinyImageNet support and several CSF implementations. The recommended setup is via Docker:

```bash
docker pull cesarclaros/systematic_analysis_ood:cuda11.7
```

The container ships every dependency required by the pipeline (PyTorch, FD-Shifts, scikit-learn, pyarrow, CLIP via `open_clip`, etc.).

If you prefer a local install, clone the modified FD-Shifts:

```bash
pip install git+https://github.com/cesar-claros/fd-shifts-0.1.1.git
```

Alternatively, install the upstream FD-Shifts v0.1.1 plus the additional packages used here:

```bash
pip install fd-shifts==0.1.1
pip install bayesian-optimization==3.1.0 faiss-cpu==1.9.0 MedPy tinyimagenet==0.9.9 torch_pca==1.0.0
```

Note that this upstream version does not include TinyImageNet as a source dataset.

### Verify TinyImageNet experiments are registered

After installing the forked FD-Shifts, confirm the TinyImageNet experiments are visible:

```bash
fd_shifts list
```

The output should include entries such as:

```
fd-shifts/tiny-imagenet-200_paper_sweep/devries_bbvgg13_do0_run1_rew2.2
```

### Environment variables

The pipeline reads two environment variables (set in `code/.env`):

```bash
EXPERIMENT_ROOT_DIR=/abs/path/to/fd_shifts_experiments     # checkpoint and score outputs
DATASET_ROOT_DIR=/abs/path/to/datasets                     # raw datasets (CIFAR-10/100, TinyImageNet, OOD sets)
```

---

## Data folder requirements

Follow the FD-Shifts dataset instructions:
[https://github.com/IML-DKFZ/fd-shifts/blob/v0.1.1/docs/datasets.md](https://github.com/IML-DKFZ/fd-shifts/blob/v0.1.1/docs/datasets.md)

In addition, download the OOD datasets used in this work into `$DATASET_ROOT_DIR`:

- OOD datasets (Textures, Places365, iSUN, LSUN, LSUN-resize): https://zenodo.org/records/17317862

---

## Trained model checkpoints

The headline classifiers come from FD-Shifts:
[https://github.com/IML-DKFZ/fd-shifts](https://github.com/IML-DKFZ/fd-shifts) (release v0.1.1).

FD-Shifts does not ship TinyImageNet checkpoints, so two extra model pools must be downloaded separately to reproduce all paper results:

- **TinyImageNet-trained VGG-13 / ViT classifiers** (the `tiny-imagenet-200_paper_sweep/*` experiments registered by our forked FD-Shifts; not part of the upstream FD-Shifts release): https://zenodo.org/records/17316185
- **ResNet-18 cross-architecture pool** used in Section 4.4 (4 datasets x 3 paradigms x 1 seed, 56 checkpoints; about 2.5 hours per training on a T4): https://zenodo.org/records/19712370

Place the unzipped checkpoints under `$EXPERIMENT_ROOT_DIR` so that FD-Shifts can find them via `fd_shifts list`.

---

## Quick reproduction with prebuilt archives

Five folders dominate the disk footprint and are expensive to regenerate (Stage 3, Stage 4, and Stage 5 outputs). To reproduce only the Stage 6 statistical analyses and the Stage 7 NC-predictor figures (which take minutes to tens of minutes), download and unzip the following archives at the repository root:

| Archive | Folder produced | Size | Cost to regenerate from scratch |
| --- | --- | --- | --- |
| `clip_scores.zip` | `clip_scores/` | 128 KB | hours (CLIP feature extraction over four datasets) |
| `clip_robustness.zip` | `clip_robustness/` | 1.6 MB | hours (CLIP across three encoders) |
| `neural_collapse_metrics.zip` | `neural_collapse_metrics/` | 312 KB | tens of minutes (NC metric extraction over 376 checkpoints) |
| `scores_risk.zip` | `scores_risk/` | 70 MB | one to two weeks (Stage 1, 2, 3 over the FD-Shifts pool) |
| `scores_risk_resnet18.zip` | `scores_risk_resnet18/` | 18 MB | days (Stage 1, 2, 3 over the ResNet-18 pool) |
| `scores_calibration.zip` | `scores_calibration/` | 114 MB | AISTATS 2026 companion paper only — not needed for this pipeline |

Download links (Google Drive):

- `clip_scores.zip`: https://drive.google.com/file/d/1vl-cg2ffmfsU5qbscrIkCowqdX57HBH_/view?usp=sharing
- `clip_robustness.zip`: https://drive.google.com/file/d/1FcNKsd36jNwbbMLyQGeNLGeLU4_T06Kl/view?usp=sharing
- `neural_collapse_metrics.zip`: https://drive.google.com/file/d/1y8tmMrVqxYRer4x9vyEcuGelpX2VkNed/view?usp=sharing
- `scores_risk.zip`: https://drive.google.com/file/d/13n8O49xMEQ4F6cOsQcyqC-estuBdPFSJ/view?usp=sharing
- `scores_risk_resnet18.zip`: https://drive.google.com/file/d/1xrgOJ7UggZiAnAr_4ud6c7r2qbXpOBR0/view?usp=sharing
- `scores_calibration.zip`: https://drive.google.com/file/d/1nCueWkz7BfbnF9DTtJzxoTl-8Q-C-bVB/view?usp=sharing (AISTATS 2026 companion paper only)

After unzipping, the Stage 6 and Stage 7 commands listed below regenerate every paper figure in under one hour on a single CPU machine.

---

## Full pipeline reproduction

The complete pipeline has eight stages, executed in order. Stages 1 to 4 operate per checkpoint and per dataset; we recommend dispatching them in parallel via your cluster's job scheduler (12 CPU workers and two GPU types in parallel were used in the paper; see Appendix C).

### Stage 1: Train Confidence Score Functions on the validation split

For each FD-Shifts checkpoint, fit the trainable CSFs (Mahalanobis covariance, Temperature scaling, NeCo PCA, KPCA Nystrom basis, ConfidNet head, etc.) on the validation set:

```bash
python cifar_iid_train.py \
    --model_path=cifar10_paper_sweep/dg_bbvgg13_do0_run1_rew2.2 \
    --no-rank_weight --no-rank_feature --ash=None \
    --use_cuda --temperature_scale
```

Arguments:

- `--model_path` is the FD-Shifts experiment ID (without the `fd-shifts/` prefix).
- `--rank_weight` / `--no-rank_weight` and `--rank_feature` / `--no-rank_feature` toggle RankWeight and RankFeat.
- `--ash` selects an ASH activation-shaping variant (`None`, `s`, `p`, or `b`).
- `--use_cuda` enables GPU.
- `--temperature_scale` applies temperature scaling to logits.

Repeat for every checkpoint in the FD-Shifts release plus the TinyImageNet and ResNet-18 checkpoints (376 total; see Appendix C for the per-source counts).

### Stage 2: Evaluate Confidence Score Functions on each OOD dataset

For each (checkpoint, OOD dataset) pair:

```bash
python cifar_test.py \
    --model_path=cifar10_paper_sweep/dg_bbvgg13_do0_run1_rew2.2 \
    --no-rank_weight --no-rank_feature --ash=None \
    --use_cuda --temperature_scale \
    --test_mode=ood_nsncs_svhn
```

Test modes correspond to the available datasets:

```
iid_test
ood_sncs_c100
ood_nsncs_svhn
ood_nsncs_ti
ood_nsncs_lsun_cropped
ood_nsncs_lsun_resize
ood_nsncs_isun
ood_nsncs_textures
ood_nsncs_places365
```

### Stage 3: Aggregate scores into per-cell CSVs

`retrieve_scores.py` is run twice per (dataset, backbone) combination: once to produce the default per-cell aggregates (read by Stage 6 statistical analyses), and once with `--fix-config` to produce the per-cell hyperparameter-locked variants (read by the projection-filtering analyses and the NC-predictor data prep).

VGG-13 and ViT (FD-Shifts pool):

```bash
# Default aggregates (consumed by stats_eval.py and mantel_analysis.py)
for src in cifar10 cifar100 supercifar100 tinyimagenet; do
  python retrieve_scores.py --dataset $src --scores-dir scores_risk
  python retrieve_scores.py --dataset $src --vit --scores-dir scores_risk
done

# Hyperparameter-locked _fix-config variants (consumed by projection_*_analysis.py
# and nc_csf_predictivity/data/build_dataset.py)
for src in cifar10 cifar100 supercifar100 tinyimagenet; do
  python retrieve_scores.py --dataset $src --fix-config --scores-dir scores_risk
  python retrieve_scores.py --dataset $src --vit --fix-config --scores-dir scores_risk
done
```

ResNet-18 cross-architecture pool (only the `--fix-config` variants are needed, and only for the Conv naming convention):

```bash
for src in cifar10 cifar100 supercifar100 tinyimagenet; do
  python retrieve_scores.py --dataset $src --fix-config \
    --network bbresnet18 --scores-dir scores_risk_resnet18
done
```

Arguments:

- `--dataset` is the source dataset name.
- `--vit` selects the ViT pool (default is the Conv backbone family, which covers VGG-13 and ResNet-18).
- `--scores-dir` is the destination directory.
- `--fix-config` skips the cross-(dropout, reward) hyperparameter selection and retains every (drop_out, reward, metric) slice. Output filenames get the `_fix-config` suffix.
- `--network` (optional) restricts the rows kept from disk to a specific backbone label (e.g., `bbresnet18`). Use this to keep ResNet-18 separate from VGG-13.

### Stage 4: Compute Neural Collapse metrics

```bash
python neural_collapse_eval.py --output-dir neural_collapse_metrics
```

This walks every checkpoint, extracts penultimate-layer activations and classifier weights, and writes the eight Papyan NC metrics into per-source CSVs in `neural_collapse_metrics/`.

### Stage 5: CLIP-based OOD stratification

CLIP feature extraction is the most expensive non-checkpoint stage; budget about ten minutes per (dataset, encoder) pair on a GPU.

```bash
python clip_proximity.py --iid_dataset cifar10 --output-dir clip_scores
python clip_proximity.py --iid_dataset cifar100 --output-dir clip_scores
python clip_proximity.py --iid_dataset supercifar100 --output-dir clip_scores
python clip_proximity.py --iid_dataset tinyimagenet --output-dir clip_scores

python clip_clustering.py --dataset cifar10 --n-clusters 3 --input-dir clip_scores --output-dir clip_scores --latex
python clip_clustering.py --dataset cifar100 --n-clusters 3 --input-dir clip_scores --output-dir clip_scores --latex
python clip_clustering.py --dataset supercifar100 --n-clusters 3 --input-dir clip_scores --output-dir clip_scores --latex
python clip_clustering.py --dataset tinyimagenet --n-clusters 3 --input-dir clip_scores --output-dir clip_scores --latex
```

For Appendix D.1 (CLIP backbone robustness):

```bash
python clip_clustering_all_backbones.py --output-dir clip_scores
python clip_robustness.py --output-dir clip_robustness
```

For Appendix D.2 (OpenOOD binary grouping comparison):

```bash
python generate_openood_grouping.py --input-dir clip_scores --output-dir clip_scores_openood
```

### Stage 6: Statistical analyses (top cliques, Mantel, projection filtering)

The following analyses run on CPU in minutes once Stages 3 and 4 are complete.

```bash
# Figures 1 and 3 (top-clique panels)
python stats_eval.py --all-paradigms --metric-group RC --filter-methods --output-dir ood_eval_outputs
python stats_eval.py --all-paradigms --metric-group RC --filter-methods \
    --clip-dir clip_scores_openood --output-dir ood_eval_outputs

# Appendix E worked example heatmap
python stats_eval_demo.py --source cifar10 --backbone Conv --metric AURC --group 0 \
    --methods Confidence GEN MSR CTM fDBD Energy --output-dir ood_eval_outputs

# Appendix F paired-AUGRC tables and clique substitutions
python projection_filtering_analysis.py --output-dir projection_analysis_outputs
python projection_clique_analysis.py --output-dir projection_clique_outputs

# Appendix G Mantel test
python mantel_analysis.py --output-dir mantel_outputs_papyan
```

### Stage 7: NC-based predictor (Section 4.4 and Appendix H)

The predictor sub-package lives under `nc_csf_predictivity/`. The full chain (data prep, splits, baselines, headline predictor, then heatmap and regret plots) is:

```bash
python -m nc_csf_predictivity.data.build_dataset
python -m nc_csf_predictivity.data.harmonize
python -m nc_csf_predictivity.data.oracle_regret
python -m nc_csf_predictivity.data.cliques_track1
python -m nc_csf_predictivity.data.cliques_resnet18

python -m nc_csf_predictivity.evaluation.splits
python -m nc_csf_predictivity.evaluation.baselines

python -m nc_csf_predictivity.ablations.calibration_features_clique

python -m nc_csf_predictivity.evaluation.coefficients_heatmap_clique
python -m nc_csf_predictivity.evaluation.regret_by_side_clique_bc
```

Wall-clock for Stage 7 is about thirty minutes to one hour on a single CPU machine: the bulk is the per-CSF L2 LogisticRegressionCV (Cs=50, cv=5, class-balanced) over twenty CSFs, three feature configurations (`source`, `n_classes`, `none`), and two splits (`xarch`, `lopo`).

---

## Expected folder structure after the full pipeline

After Stages 1 to 7 complete, the repository should look like:

```
ood_systematic/
|-- src/
|-- nc_csf_predictivity/
|   `-- outputs/
|       |-- splits/
|       |-- track1/
|       |   |-- dataset/long_harmonized.parquet, oracle.parquet
|       |   |-- xarch/baselines/aggregate.parquet
|       |   `-- lopo/baselines/aggregate.parquet
|       |-- ablations/calib_cliques/track1/{xarch,lopo}/{source,n_classes,none}/
|       |   |-- preds.parquet
|       |   `-- coefficients.parquet
|       |-- clique_bc/track1/{xarch,lopo}/{source,n_classes,none}/stats.json
|       `-- figures/
|           |-- clique_coefficients_heatmap_xarch_{source,n_classes,none}.{pdf,png}
|           |-- clique_coefficients_heatmap_lopo_{source,n_classes,none}.{pdf,png}
|           `-- regret_by_side_clique_bc_{xarch,lopo}.{pdf,png}
|
|-- scores_risk/                      # Stage 3 outputs (about 70 MB)
|   |-- scores_AUGRC_MCD-False_Conv_cifar10.csv               # default aggregates
|   |-- scores_all_AUGRC_MCD-False_Conv_cifar10.csv
|   |-- scores_AUGRC_MCD-False_Conv_cifar10_fix-config.csv    # --fix-config aggregates
|   |-- ...                           # AURC, AUROC_f, FPR@95TPR, ECE, MCE; per source x backbone x MCD
|
|-- scores_risk_resnet18/             # ResNet-18 pool, fix-config only (about 18 MB)
|   `-- scores_*_MCD-*_Conv_<source>_fix-config.csv
|
|-- scores_calibration/               # AISTATS 2026 companion-paper artifact only.
|   |                                 # listed here for reference (about 114 MB).
|   |-- calibration_results_<source>_<backbone>.csv      # per-checkpoint reliability data
|   |-- scores_<ECE_*|MCE>_MCD-*_<backbone>_<source>.csv
|   |-- scores_all_<ECE_*|MCE>_MCD-*_<backbone>_<source>.csv
|   `-- hyperparameters_results_MCD-*_<backbone>_<source>.csv
|
|-- neural_collapse_metrics/          # Stage 4 outputs (about 312 KB)
|   |-- nc_metrics_Conv.csv
|   `-- nc_metrics_ViT.csv
|
|-- clip_scores/                      # Stage 5a/b outputs (about 128 KB)
|   |-- clip_distances_<source>.csv
|   `-- clip_proximity_<source>.json
|
|-- clip_scores_openood/              # Stage 5e outputs (about 12 KB)
|   `-- clip_distances_<source>.csv   # cifar10, cifar100, supercifar100 only
|
|-- clip_robustness/                  # Stage 5c/d outputs (about 1.6 MB)
|   |-- average_ranks_with_groups.csv
|   |-- cross_method_agreement.csv
|   `-- dendrogram_<source>_<encoder>.{pdf,jpeg}
|
|-- ood_eval_outputs/                 # Stage 6a/b outputs (about 7 MB)
|   |-- top_cliques_all_paradigms_False_RC.{pdf,jpeg}
|   |-- top_cliques_all_paradigms_False_RC_clip_scores_openood.{pdf,jpeg}
|   `-- 6_0_posthoc_heatmap.{pdf,png}
|
|-- mantel_outputs_papyan/            # Stage 6c outputs (about 400 KB)
|   |-- mantel_summary_*.csv
|   |-- mantel_per_metric_*.csv
|   `-- mantel_loo_*.csv
|
|-- projection_analysis_outputs/      # Stage 6d outputs (about 20 KB)
|   `-- projection_paired_augrc_<backbone>.csv
|
|-- projection_clique_outputs/        # Stage 6e outputs (about 32 KB)
|   `-- projection_clique_substitutions_<backbone>.csv
|
|-- *.py                              # the pipeline scripts
`-- README.md
```

Total disk footprint after the full pipeline: about 230 MB excluding any FD-Shifts checkpoint or dataset storage.

---

## Mapping paper figures and tables to scripts

| Paper artifact | Script that produces it | Output location |
| --- | --- | --- |
| Figure 1 (top-clique panels) | `stats_eval.py --all-paradigms --metric-group RC` (1) | `ood_eval_outputs/top_cliques_all_paradigms_False_RC.pdf` |
| Figure 2 (cross-architecture regret) | `nc_csf_predictivity.evaluation.regret_by_side_clique_bc` | `nc_csf_predictivity/outputs/figures/regret_by_side_clique_bc_xarch.pdf` |
| Figure 3 (OpenOOD comparison) | `stats_eval.py --clip-dir clip_scores_openood --metric-group RC` (1) | `ood_eval_outputs/top_cliques_all_paradigms_False_RC_clip_scores_openood.pdf` |
| Appendix A hyperparameter tables | `recompute_metric.py` | `scores_calibration/hyperparameters_results_*.csv` |
| Appendix B NC tables | `neural_collapse_eval.py` | `neural_collapse_metrics/nc_metrics_*.csv` |
| Appendix D Table 6 (OOD grouping) | `clip_clustering.py --latex` | `clip_scores/clip_distances_*.csv` plus printed LaTeX |
| Appendix D Table 7 (CLIP distances) | `clip_proximity.py` + `clip_clustering.py` | `clip_scores/` |
| Appendix D.1 robustness tables | `clip_robustness.py`, `clip_clustering_all_backbones.py` | `clip_robustness/` |
| Appendix E worked example | `stats_eval_demo.py` | `ood_eval_outputs/6_0_posthoc_heatmap.pdf` |
| Appendix F Tables 15, 16 | `projection_filtering_analysis.py` | `projection_analysis_outputs/` |
| Appendix F clique substitutions | `projection_clique_analysis.py` | `projection_clique_outputs/` |
| Appendix G Tables 18, 19 (Mantel) | `mantel_analysis.py` | `mantel_outputs_papyan/` |
| Appendix H coefficient heatmaps | `nc_csf_predictivity.evaluation.coefficients_heatmap_clique` | `nc_csf_predictivity/outputs/figures/clique_coefficients_heatmap_xarch_{source,n_classes,none}.pdf` |

(1) `stats_eval.py` is shared with our AISTATS 2026 companion paper, which uses it with `--metric-group CE` or `--metric-group CE_BOUND` to compute calibration top cliques. Only the `--metric-group RC` invocations (AURC / AUGRC) feed this paper.

---

## Citing FD-Shifts, our framework, and external datasets

Please cite the FD-Shifts paper if you use their framework and the released checkpoints:

```bibtex
@article{jaeger2022call,
  title={A Call to Reflect on Evaluation Practices for Failure Detection in Image Classification},
  author={Jaeger, Paul F. and Lueth, Carsten T. and Klein, Lukas and Bungert, Till J.},
  journal={ICLR},
  year={2023}
}

@article{traub2024overcoming,
  title={Overcoming Common Flaws in the Evaluation of Selective Classification Systems},
  author={Traub, Jonathan and Bungert, Till J. and L\"uth, Carsten T. and Baumgartner, Michael and Maier-Hein, Klaus H. and Maier-Hein, Lena and Jaeger, Paul F.},
  journal={arXiv preprint arXiv:2407.01032},
  year={2024}
}
```

Please refer to their github page for more information on the FD-Shifts framework: [https://github.com/fd-shifts/fd-shifts.pytorch](https://github.com/fd-shifts/fd-shifts.pytorch)

Please cite our paper if you use our framework and released checkpoints for ResNet-18 models and TinyImageNet models

```bibtex
@misc{claros-olivares2026systematic,
      title={A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts}, 
      author={Claudio César Claros Olivares and Austin J. Brockmeier},
      year={2026},
      eprint={2511.11934},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.11934}, 
}
```

The image-classification benchmarks and OOD evaluation sets used here retain their original licenses; see the paper's "Licenses for existing assets" checklist entry for the full list of dataset citations (CIFAR-10/100, SuperCIFAR-100, TinyImageNet, iSUN, LSUN, SVHN, Places365, Textures/DTD).
