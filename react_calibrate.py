"""Calibrate the ReAct clipping threshold for one checkpoint.

ReAct (Sun et al., 2021) clips activations at the p-th percentile of the ID
activation distribution. This pipeline applies activation surgery at the
pre-last-maxpool feature map (the same hook ASH uses in
`TrainedModule.forward_features`), so the threshold is computed there, over
the validation split. The value prints as the LAST stdout line so a runner
can capture it:

    t=$(python react_calibrate.py --model_path <exp> --use_cuda | tail -1)
    python csf_fit.py --model_path <exp> ... --ash "react@$t"

CNN backbones only (the pilot scope); the ViT hook point differs.
"""

import argparse

import torch
from fd_shifts import logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
from fd_shifts.models import get_model
from fd_shifts.utils import exp_utils

from src import utils
from src.trained_module import TrainedModule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the ReAct clipping threshold on validation data")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--percentile', type=float, default=90.0,
                        help="ID activation percentile (ReAct paper: 90)")
    parser.add_argument('--max_batches', type=int, default=20,
                        help="validation batches to sample")
    parser.add_argument('--use_cuda', action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()

    path = args.model_path
    study_name = utils.get_study_name(path)
    if study_name == 'vit':
        raise NotImplementedError(
            "react_calibrate supports CNN backbones only (pilot scope)")
    cf = utils.get_conf(path, study_name)
    do_enabled = utils.is_dropout_enabled(path)
    # Mirror csf_fit.py's SuperCIFAR handling: without avg_pool=False the
    # VGG head shape differs from the dropout-enabled checkpoints and the
    # state dict fails to load.
    if 'super' in path:
        cf.eval.query_studies.noise_study = ['corrupt_cifar100']
        cf.eval.query_studies.new_class_study = ['cifar10', 'svhn',
                                                 'tinyimagenet_resize']
        if do_enabled and 'vgg' in path:
            logger.info("Disabling average pooling for VGG-13 supercifar "
                        "experiments with dropout enabled...")
            cf.model.avg_pool = False
    ckpt_path = exp_utils._get_path_to_best_ckpt(
        cf.exp.dir, 'last', cf.test.selection_mode)
    module = get_model(cf.model.name)(cf)
    module.load_only_state_dict(ckpt_path, device='cpu')
    if study_name == 'confidnet':
        module.backbone.encoder.disable_dropout()
        module.network.encoder.disable_dropout()
    else:
        module.model.encoder.disable_dropout()

    datamodule = FDShiftsDataLoader(cf)
    datamodule.setup()
    model = TrainedModule(module, study_name, cf, rank_weight=False,
                          rank_feat=False, ash_method=None,
                          use_cuda=args.use_cuda)
    hook = model.model.encoder.features[:int(model.maxpool_layers_name[-1])]

    samples = []
    loader = datamodule.val_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= args.max_batches:
                break
            act = hook(x.to(model.device))
            samples.append(act.flatten().cpu())
    values = torch.cat(samples).float()
    threshold = torch.quantile(
        values[torch.randperm(len(values))[:2_000_000]], args.percentile / 100)
    logger.info(f"ReAct threshold p{args.percentile:g} over "
                f"{len(values):,} activations: {threshold:.6g}")
    print(f"{threshold:.6g}")


if __name__ == "__main__":
    main()
