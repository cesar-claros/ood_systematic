"""G1/G2 gate: compare the single-model pilot run against OpenOOD v1.5.

References (OpenOOD v1.5 report, arXiv 2306.09301, Table 1; ImageNet-1k
standard benchmark, ResNet-50, fetched 2026-08-06): near-OOD = mean AUROC
over {SSB-hard, NINCO}, far-OOD = mean over {iNaturalist, Textures,
OpenImage-O}; ID accuracy 76.18.

Hard gates (exit nonzero on failure):
  - accuracy within 0.5 points of 76.18;
  - MSR (their MSP) and Energy (their EBO) near/far AUROC within 2.0 points.
    Tolerance covers the known protocol deltas: our scores use the fitted
    temperature where OpenOOD uses T=1, and our detector fits use the
    100/class fit draw where OpenOOD fits on larger train subsets.
Report-only comparisons: ViM and Maha (larger legitimate protocol gaps),
plus the three extra sets (imagenet_o, sun, places) OpenOOD's table lacks.

Run after the pilot model:
  python x9_imagenet/g1_gate.py --out-dir $DATASET_ROOT_DIR/x9_outputs
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd
from loguru import logger

NEAR = ["ssb_hard", "ninco"]
FAR = ["inaturalist", "texture", "openimage_o"]
EXTRA = ["imagenet_o", "sun", "places"]
REF = {  # csf -> (near AUROC, far AUROC), OpenOOD v1.5 Table 1, ResNet-50
    "MSR": (76.02, 85.23),
    "Energy": (75.89, 89.47),
    "ViM": (72.08, 92.68),
    "Maha": (55.44, 74.25),
}
HARD = {"MSR", "Energy"}
TOL_AUROC = 2.0
ACC_REF, TOL_ACC = 76.18, 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="resnet50.tv_in1k")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    rows = pd.read_parquet(out / f"{args.tag}_rows.parquet")
    model = pd.read_parquet(out / f"{args.tag}_model.parquet").iloc[0]

    failures, lines = [], [f"# G1/G2 gate: {args.tag}\n\n"]
    acc = 100 * float(model["acc"])
    ok = abs(acc - ACC_REF) <= TOL_ACC
    lines.append(f"Accuracy: {acc:.2f} vs {ACC_REF} (tol {TOL_ACC}) "
                 f"{'PASS' if ok else 'FAIL'}\n\n")
    if not ok:
        failures.append(f"accuracy {acc:.2f}")

    def mean_auroc(csf: str, sets: list[str]) -> float:
        sub = rows[(rows["csf"] == csf) & rows["eval_dataset"].isin(sets)]
        assert len(sub) == len(sets), (csf, sets, sub["eval_dataset"].tolist())
        return 100 * float(sub["auroc"].mean())

    lines.append("| CSF | near ours | near ref | far ours | far ref | gate |\n"
                 "|---|---|---|---|---|---|\n")
    for name, (ref_n, ref_f) in REF.items():
        near, far = mean_auroc(name, NEAR), mean_auroc(name, FAR)
        hard = name in HARD
        ok = (abs(near - ref_n) <= TOL_AUROC
              and abs(far - ref_f) <= TOL_AUROC)
        verdict = ("PASS" if ok else "FAIL") if hard else \
            (f"report-only ({'within' if ok else 'outside'} {TOL_AUROC})")
        if hard and not ok:
            failures.append(f"{name} near {near:.2f}/{ref_n} "
                            f"far {far:.2f}/{ref_f}")
        lines.append(f"| {name} | {near:.2f} | {ref_n} | {far:.2f} | "
                     f"{ref_f} | {verdict} |\n")

    lines.append("\nExtra sets (no OpenOOD reference), AUROC per CSF:\n\n")
    ex = rows[rows["eval_dataset"].isin(EXTRA)].pivot_table(
        index="csf", columns="eval_dataset", values="auroc")
    lines.append("```\n" + (100 * ex).round(2).to_string() + "\n```\n")
    lines.append("\nG2 sanity, iid failure-detection AUGRC (lower better):\n\n")
    iid = rows[rows["eval_dataset"] == "iid_test"].sort_values("augrc")
    lines.append("```\n"
                 + iid[["csf", "augrc"]].to_string(index=False) + "\n```\n")

    report = out / f"g1_gate_{args.tag}.md"
    report.write_text("".join(lines))
    print("".join(lines))
    if failures:
        logger.error(f"G1 GATE FAILED: {failures}")
        raise SystemExit(1)
    logger.info(f"G1/G2 gates PASSED; report at {report}")


if __name__ == "__main__":
    main()
