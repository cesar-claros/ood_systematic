"""Within-shortlist selection policies: how does a practitioner pick?

The set-regret convention credits a shortlist with its best member. This
script measures what realizable selection policies cost, for the
regime-free shortlists shown in the responses (NC+source and NC only,
heads trained on the 280 VGG-13 models), on all three targets:

  best    : the reporting convention (min AUGRC over members per row)
  id-pick : ONE member chosen per model with no OOD data at all, the
            member with the lowest ID failure-detection AUGRC (the
            model's iid rows), then evaluated on every OOD row
  random  : expected regret of deploying a uniformly drawn member
            (per-row mean over members)
  worst   : the adversarial bracket (max over members)

Gates: the 'best' pooled numbers must reproduce 37_regimefree_tables.md.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/shortlist_pick_policies.py
Output: nc_csf_predictivity/outputs/38_shortlist_pick_policies.md
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[2]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

from pool_a_analysis import OUT_ROOT, pool_cliques_for  # noqa: E402
from calibration_features_clique import (  # noqa: E402
    NC_PRIMARY,
    add_model_id,
)
from input_ablation_grid import (  # noqa: E402
    REGIMES,
    ssl_shortlists,
    stored_shortlists,
)

CONFIGS = ["source_nr", "none_nr"]
EXPECTED_BEST = {   # pooled rows of 37_regimefree_tables.md
    ("ResNet18", "source_nr"): (1.06, 0.96, 0.64),
    ("ResNet18", "none_nr"): (1.23, 1.16, 0.80),
    ("ViT", "source_nr"): (1.76, 2.34, 2.30),
    ("ViT", "none_nr"): (1.35, 2.24, 3.94),
    ("SSL", "source_nr"): (4.96, 3.99, 5.07),
    ("SSL", "none_nr"): (1.59, 1.38, 0.59),
}


def policy_regrets(ood: pd.DataFrame, iid: pd.Series,
                   shortlists: pd.DataFrame) -> pd.DataFrame:
    members = (shortlists.groupby(["model_id", "regime"])["csf"]
               .apply(set).rename("members"))
    recs = []
    for (mid, _, regime), g in ood.groupby(
            ["model_id", "eval_dataset", "regime"]):
        vals = g.set_index("csf")["augrc"]
        oracle, worst_all = float(vals.min()), float(vals.max())
        mem = members.get((mid, regime), set()) & set(vals.index)
        if not mem:
            r = worst_all - oracle
            recs.append({"regime": regime, "best": r, "id_pick": r,
                         "random": r, "worst_member": r})
            continue
        mvals = vals.loc[list(mem)]
        id_avail = iid.loc[[(mid, c) for c in mem
                            if (mid, c) in iid.index]]
        if len(id_avail):
            pick = id_avail.idxmin()[1]
            id_regret = float(vals.loc[pick]) - oracle
        else:
            id_regret = float(mvals.mean()) - oracle
        recs.append({
            "regime": regime,
            "best": float(mvals.min()) - oracle,
            "id_pick": id_regret,
            "random": float(mvals.mean()) - oracle,
            "worst_member": float(mvals.max()) - oracle,
        })
    df = pd.DataFrame(recs)
    return df.groupby("regime").mean().clip(lower=0)


def main() -> None:
    lh = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    lh = add_model_id(lh)

    # SSL shortlists refit exactly as in regimefree_tables.py
    cliques = pool_cliques_for(("VGG13",), lh)
    lh_std = lh.copy()
    for arch, sub in lh_std.groupby("architecture"):
        for c in NC_PRIMARY:
            lh_std.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label", aggfunc="first")
                  .reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]
    vgg_models = (lh_std[lh_std["architecture"] == "VGG13"]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    tr_marginal = pd.DataFrame(
        [{**m.to_dict(), "regime": r} for _, m in vgg_models.iterrows()
         for r in REGIMES]).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward",
                        "regime"], how="inner")
    models_df = pd.read_parquet(OUT_ROOT / "pool_a" / "models_pool_a.parquet")
    pool_long = pd.read_parquet(OUT_ROOT / "pool_a" / "long_pool_a.parquet")
    pool_long["model_id"] = (pool_long["paradigm"] + "|"
                             + pool_long["source"] + "|"
                             + pool_long["run"].astype(str))

    targets = {}
    for arch, split, fold in [("ResNet18", "xarch", "vgg13_to_resnet18"),
                              ("ViT", "lopo", "lopo_modelvit")]:
        sub = lh[lh["architecture"] == arch]
        ood = sub[sub["regime"].isin(REGIMES)][
            ["model_id", "eval_dataset", "regime", "csf", "augrc"]]
        iid = (sub[sub["regime"] == "test"]
               .groupby(["model_id", "csf"])["augrc"].min())
        sls = {c: stored_shortlists(split, c, fold) for c in CONFIGS}
        targets[arch] = (ood, iid, sls)
    ood = pool_long[pool_long["regime"].isin(REGIMES)][
        ["model_id", "eval_dataset", "regime", "csf", "augrc"]]
    iid = (pool_long[pool_long["regime"] == "test"]
           .groupby(["model_id", "csf"])["augrc"].min())
    sls = {c: ssl_shortlists(c, models_df, tr_marginal, tr_marginal,
                             csf_cols) for c in CONFIGS}
    targets["SSL"] = (ood, iid, sls)

    results, bad = {}, []
    for tgt, (ood, iid, sls) in targets.items():
        for config in CONFIGS:
            res = policy_regrets(ood, iid, sls[config])
            results[(tgt, config)] = res
            got = tuple(round(float(res.loc[r, "best"]), 2)
                        for r in REGIMES)
            exp = EXPECTED_BEST[(tgt, config)]
            if any(abs(g - e) > 0.02 for g, e in zip(got, exp)):
                bad.append((tgt, config, exp, got))
            logger.info(f"{tgt}/{config}: best {got}, id-pick "
                        + str(tuple(round(float(res.loc[r, 'id_pick']), 2)
                                    for r in REGIMES)))
    if bad:
        for b in bad:
            logger.error(f"gate mismatch: {b}")
        raise SystemExit("Gates FAILED; report not written.")
    logger.info("Gates PASSED (best-member pooled matches "
                "37_regimefree_tables)")

    lines = [
        "# Within-shortlist selection policies (regime-free shortlists)\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/"
        "shortlist_pick_policies.py`. Policies per (model, regime) "
        "shortlist: best member (reporting convention); id-pick = the one "
        "member with the lowest ID failure-detection AUGRC for that model "
        "(no OOD data used); random = expected regret of a uniformly "
        "drawn member; worst member. Pooled mean regret per regime; "
        "'best' gates on 37_regimefree_tables.md.\n\n",
        "| Target | Config | Regime | best | id-pick | random | worst "
        "member |\n|---|---|---|---|---|---|---|\n"]
    for tgt in ["ResNet18", "ViT", "SSL"]:
        for config in CONFIGS:
            label = "NC+source" if config == "source_nr" else "NC only"
            res = results[(tgt, config)]
            for regime in REGIMES:
                row = res.loc[regime]
                lines.append(
                    f"| {tgt} | {label} | {regime} | {row['best']:.2f} | "
                    f"{row['id_pick']:.2f} | {row['random']:.2f} | "
                    f"{row['worst_member']:.2f} |\n")
    out = OUT_ROOT / "38_shortlist_pick_policies.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}")
    print("".join(lines))


if __name__ == "__main__":
    main()
