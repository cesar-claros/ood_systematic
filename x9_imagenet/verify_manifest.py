"""Verify the candidate checkpoint roster against the installed timm registry
and write the authoritative model manifest.

Two levels:
  default      registry + pretrained-cfg check only (seconds, no downloads):
               tag exists, num_classes==1000, eval input 224x224; records
               mean/std/crop_pct/interpolation per checkpoint.
  --instantiate  additionally builds each surviving model with pretrained
               weights (downloads to HF_HOME; ~10-20 GB total across the
               pool), records parameter count and penultimate feature dim,
               and forward-checks a zeros batch -> (1, 1000) logits.

Run inside the x9 container (documentation/imagenet_scale_plan.md):
  singularity exec x9_imagenet.sif python x9_imagenet/verify_manifest.py
  singularity exec --nv x9_imagenet.sif \
      env HF_HOME=$DATASET_ROOT_DIR/hf python x9_imagenet/verify_manifest.py --instantiate

Output: x9_imagenet/manifest_verified.csv (tracked; the reproducibility
artifact naming the exact pool).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
from loguru import logger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from candidates import CANDIDATES  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "manifest_verified.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instantiate", action="store_true",
                    help="download weights, record params/feature dim, "
                         "forward-check each model")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any candidate is missing/rejected")
    args = ap.parse_args()

    import timm
    import torch

    logger.info(f"timm {timm.__version__}, torch {torch.__version__}, "
                f"{len(CANDIDATES)} candidates")
    registry = set(timm.list_pretrained())

    rows = []
    for family, tag, note in CANDIDATES:
        rec = {"family": family, "tag": tag, "note": note, "status": "ok",
               "input_size": None, "crop_pct": None, "interpolation": None,
               "mean": None, "std": None, "params_m": None,
               "num_features": None}
        if tag not in registry:
            rec["status"] = "missing_from_registry"
            rows.append(rec)
            continue
        try:
            cfg = timm.get_pretrained_cfg(tag)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"cfg_error:{e}"
            rows.append(rec)
            continue
        n_cls = getattr(cfg, "num_classes", None)
        in_size = tuple(getattr(cfg, "input_size", ()) or ())
        if n_cls != 1000:
            rec["status"] = f"rejected_num_classes_{n_cls}"
        elif len(in_size) != 3 or in_size[1] != 224 or in_size[2] != 224:
            rec["status"] = f"rejected_input_size_{in_size}"
        rec.update({
            "input_size": "x".join(map(str, in_size)),
            "crop_pct": getattr(cfg, "crop_pct", None),
            "interpolation": getattr(cfg, "interpolation", None),
            "mean": ",".join(f"{v:g}" for v in getattr(cfg, "mean", ())),
            "std": ",".join(f"{v:g}" for v in getattr(cfg, "std", ())),
        })
        rows.append(rec)

    df = pd.DataFrame(rows)
    ok = df["status"] == "ok"
    logger.info(f"registry check: {int(ok.sum())} ok / "
                f"{int((~ok).sum())} dropped")
    for _, r in df[~ok].iterrows():
        logger.warning(f"  {r['tag']}: {r['status']}")

    if args.instantiate:
        for i, r in df[ok].iterrows():
            tag = r["tag"]
            try:
                model = timm.create_model(tag, pretrained=True)
                model.eval()
                with torch.no_grad():
                    out = model(torch.zeros(1, 3, 224, 224))
                assert out.shape == (1, 1000), out.shape
                df.at[i, "params_m"] = round(
                    sum(p.numel() for p in model.parameters()) / 1e6, 1)
                df.at[i, "num_features"] = int(model.num_features)
                logger.info(f"  {tag}: OK "
                            f"({df.at[i, 'params_m']}M params, "
                            f"d={df.at[i, 'num_features']})")
                del model
            except Exception as e:  # noqa: BLE001
                df.at[i, "status"] = f"instantiate_error:{e}"
                logger.error(f"  {tag}: {e}")
        ok = df["status"] == "ok"

    df.to_csv(OUT, index=False)
    fam = df[ok].groupby("family").size()
    logger.info(f"wrote {OUT}")
    logger.info("verified pool per family:\n" + fam.to_string())
    logger.info(f"TOTAL verified: {int(ok.sum())}")
    if args.strict and int((~ok).sum()):
        raise SystemExit("strict mode: some candidates failed")


if __name__ == "__main__":
    main()
