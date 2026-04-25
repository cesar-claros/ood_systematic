#%%
"""Compute per-configuration head-side logit diagnostics.

Walks the same ``configs_exp/configs_{dataset}_iid_train.txt`` grid as
``neural_collapse_eval.py``. For each config we:

 1. Load the checkpoint and run a forward pass on the validation split.
 2. Load T* from ``{cf.exp.dir}/params/Temperature_params{model_opts}.pt``
    if it exists; otherwise fit on val logits and save (so downstream
    CSF scoring remains consistent).
 3. Compute raw and scaled softmax/logit diagnostics via
    ``src.head_diagnostics.compute_head_diagnostics``.

Emits ``head_logit_metrics/head_logit_metrics.csv`` keyed on
(architecture, study, dataset, dropout, run, reward, lr, model_opts) so it
joins directly against ``neural_collapse_metrics/nc_metrics.csv``.
"""
import os
import argparse

import pandas as pd
import torch
from loguru import logger

from fd_shifts.utils import exp_utils
from fd_shifts.models import get_model
from fd_shifts.loaders.data_loader import FDShiftsDataLoader

from src import scores_methods, utils
from src.utils import get_study_name, is_dropout_enabled, get_conf, extract_char_after_substring
from src.head_diagnostics import compute_head_diagnostics

os.environ.setdefault("EXPERIMENT_ROOT_DIR", "/work/cniel/sw/FD_Shifts/project/experiments")
os.environ.setdefault("DATASET_ROOT_DIR",    "/work/cniel/sw/FD_Shifts/project/datasets")


def _load_or_fit_temperature(cf, model_opts: str, model_eval: dict,
                             do_enabled: bool) -> tuple[object, object | None]:
    ts = scores_methods.TemperatureScaling(cf)
    t_path = os.path.join(cf.exp.dir, "params", f"Temperature_params{model_opts}.pt")
    if os.path.exists(t_path):
        ts.load_params(filename=f"Temperature_params{model_opts}")
    else:
        logger.info(f"Temperature_params not found at {t_path}; fitting on val.")
        ts.compute_temperature(model_eval["logits"], model_eval["labels"])
        ts.save_params(filename=f"Temperature_params{model_opts}")

    ts_dist = None
    if do_enabled:
        ts_dist = scores_methods.TemperatureScaling(cf)
        td_path = os.path.join(cf.exp.dir, "params",
                               f"Temperature_distribution_params{model_opts}.pt")
        if os.path.exists(td_path):
            ts_dist.load_params(filename=f"Temperature_distribution_params{model_opts}")
        else:
            logger.info(f"Temperature_distribution_params not found at {td_path}; fitting on val.")
            ts_dist.compute_temperature(
                model_eval["logits_dist"].mean(dim=2), model_eval["labels"]
            )
            ts_dist.save_params(filename=f"Temperature_distribution_params{model_opts}")
    return ts, ts_dist


def main():
    parser = argparse.ArgumentParser(description="Head-side logit diagnostics")
    parser.add_argument("--output-dir", type=str, default="head_logit_metrics",
                        help="Directory to write head_logit_metrics.csv")
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "head_logit_metrics.csv")
    logger.info(f"Output: {out_csv}")

    rows: list[dict] = []
    datasets = ["cifar10", "supercifar", "cifar100", "tinyimagenet"]

    for dataset in datasets:
        config_file = f"configs_exp/configs_{dataset}_iid_train.txt"
        if not os.path.exists(config_file):
            logger.warning(f"Config file {config_file} not found. Skipping {dataset}.")
            continue
        logger.info(f"Processing dataset: {dataset}")

        with open(config_file, "r") as f:
            for cfg_line in f:
                parts = cfg_line.split()
                if len(parts) < 7:
                    continue
                path = parts[1]
                rank_weight_opt = "no" not in parts[2]
                rank_feat_opt   = "no" not in parts[3]
                ash_method_opt  = parts[4]
                use_cuda_opt    = "no" not in parts[5]

                # Only the clean config matches nc_metrics.csv's join keys.
                if rank_weight_opt or rank_feat_opt or ash_method_opt != "None":
                    continue

                if not (torch.cuda.is_available() and use_cuda_opt):
                    use_cuda_opt = False
                ash_val = None if ash_method_opt == "None" else ash_method_opt

                study_name = get_study_name(path)
                do_enabled = is_dropout_enabled(path)

                try:
                    cf = get_conf(path, study_name)
                    ckpt_path = exp_utils._get_path_to_best_ckpt(
                        cf.exp.dir, "last", cf.test.selection_mode
                    )
                except Exception as e:
                    logger.warning(f"Failed to load config/ckpt for {path}: {e}")
                    continue

                if "super" in path:
                    cf.eval.query_studies.noise_study = ["corrupt_cifar100"]
                    cf.eval.query_studies.new_class_study = [
                        "cifar10", "svhn", "tinyimagenet_resize",
                    ]
                    if do_enabled:
                        cf.model.avg_pool = False
                if "vit" in path:
                    cf.data.num_workers = 12

                try:
                    ModelClass = get_model(cf.model.name)
                    module = ModelClass(cf)
                    module.load_only_state_dict(ckpt_path, device="cpu")
                except Exception as e:
                    logger.error(f"Error loading model from {ckpt_path}: {e}")
                    continue

                if study_name == "confidnet":
                    module.backbone.encoder.disable_dropout()
                    module.network.encoder.disable_dropout()
                elif study_name in ("devries", "dg"):
                    module.model.encoder.disable_dropout()
                elif study_name == "vit":
                    module.disable_dropout()

                if do_enabled and use_cuda_opt:
                    if study_name in ("devries", "dg"):
                        cf.trainer.batch_size //= 2
                    elif study_name == "confidnet":
                        cf.trainer.batch_size //= 4
                if study_name == "vit":
                    if use_cuda_opt and not do_enabled:
                        cf.trainer.batch_size //= 2
                    elif use_cuda_opt and do_enabled:
                        cf.trainer.batch_size //= 2
                        cf.eval.confidence_measures.test = [
                            i for i in cf.eval.confidence_measures.test if "mcd" not in i
                        ]
                    elif not use_cuda_opt:
                        cf.trainer.batch_size = 128

                logger.info(f"Processing: {study_name}, DO={do_enabled}, path={path}")

                datamodule = FDShiftsDataLoader(cf)
                datamodule.setup()

                try:
                    model = scores_methods.TrainedModule(
                        module, study_name, cf,
                        rank_weight=rank_weight_opt,
                        rank_feat=rank_feat_opt,
                        ash_method=ash_val,
                        use_cuda=use_cuda_opt,
                    )
                    model_opts = f"_RW{int(rank_weight_opt)}_RF{int(rank_feat_opt)}_ASH{str(ash_val)}"

                    logger.info("Evaluating on train split...")
                    model_eval = utils.compute_model_evaluations(model, datamodule, set_name="train")

                    ts, ts_dist = _load_or_fit_temperature(cf, model_opts, model_eval, do_enabled)

                    diags = compute_head_diagnostics(
                        logits=model_eval["logits"],
                        labels=model_eval["labels"],
                        temperature=ts.temperature,
                        logits_dist=model_eval.get("logits_dist") if do_enabled else None,
                    )
                    if do_enabled and ts_dist is not None:
                        diags["temperature_mcd"] = float(ts_dist.temperature)

                    run_val = int(extract_char_after_substring(path, "run")) if "run" in path else 0
                    rew_val = float(extract_char_after_substring(path, "rew")) if "rew" in path else 0.0
                    lr_val  = extract_char_after_substring(path, "lr") if "lr" in path else "0.0"

                    rows.append({
                        "dataset":      dataset,
                        "architecture": "VGG13" if study_name != "vit" else "ViT",
                        "study":        study_name,
                        "dropout":      do_enabled,
                        "run":          run_val,
                        "reward":       rew_val,
                        "lr":           lr_val,
                        "model_opts":   model_opts,
                        **diags,
                    })
                    # Incremental write so a long HPC run leaves partial results.
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    continue

    if not rows:
        logger.warning("No rows computed; nothing written.")
        return
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.success(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
