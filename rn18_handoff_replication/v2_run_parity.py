"""Run-to-run parity of two version-2 extractions (forward parity check the
status review asked for). Compares, per record present in both directories,
the checkpoint identity (file sha256, state-dict digests, epoch), the sample
identities (ID offsets, label digests), the balancing indices, and the
per-example arrays (scores, logits) by max absolute difference and by the
count of cells whose frozen rounded outcomes differ. Reports differences
only; no outcome value is printed.

Usage (from code/):
    python rn18_handoff_replication/v2_run_parity.py DIR_A DIR_B [--out report.json]
    python rn18_handoff_replication/v2_run_parity.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
SCORES = ("Energy", "CTM", "MSR", "MLS", "Maha", "fDBD")


def compare_record(a: Path, b: Path) -> dict:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    za, zb = np.load(a.with_suffix(".npz")), np.load(b.with_suffix(".npz"))
    out = {"slug": ra["slug"]}
    ca, cb = ra["v2"]["checkpoint"], rb["v2"]["checkpoint"]
    out["checkpoint_same"] = {k: ca.get(k) == cb.get(k) for k in ("file_sha256", "state_dict_digest_loaded_module", "epoch", "global_step")}
    out["id_test_same"] = {k: ra["v2"]["id_test"].get(k) == rb["v2"]["id_test"].get(k) for k in ("label_sequence_sha256", "n", "tenPercent_pre_slice", "devries_val_slice")}
    out["balancing_same"] = all(np.array_equal(za[f"set__{s}__balance_id_idx"], zb[f"set__{s}__balance_id_idx"]) and np.array_equal(za[f"set__{s}__balance_ood_idx"], zb[f"set__{s}__balance_ood_idx"]) for s in SHIFTS if f"set__{s}__balance_id_idx" in za and f"set__{s}__balance_id_idx" in zb)
    diffs = {}
    for key in za.files:
        if key in zb.files and za[key].shape == zb[key].shape and za[key].dtype.kind == "f":
            diffs[key] = float(np.max(np.abs(za[key].astype(np.float64) - zb[key].astype(np.float64)))) if za[key].size else 0.0
    out["max_abs_diff"] = {"id_scores": max((v for k, v in diffs.items() if k.startswith("id__score__")), default=None),
                           "id_logits": diffs.get("id__logits"),
                           "ood_scores": max((v for k, v in diffs.items() if "__score__" in k and k.startswith("set__")), default=None),
                           "ood_logits": max((v for k, v in diffs.items() if k.endswith("__logits") and k.startswith("set__")), default=None),
                           "sigma_w": diffs.get("sigma_w"), "cov_res": max((v for k, v in diffs.items() if k.endswith("__cov_res")), default=None)}
    out["shape_mismatch"] = sorted(k for k in za.files if k in zb.files and za[k].shape != zb[k].shape)
    out["keys_only_in_a"] = sorted(set(za.files) - set(zb.files)); out["keys_only_in_b"] = sorted(set(zb.files) - set(za.files))
    n_round_diff = 0
    for s in SHIFTS:
        oa, ob = ra["ood"].get(s, {}), rb["ood"].get(s, {})
        if "error" in oa or "error" in ob:
            continue
        for sc in SCORES:
            n_round_diff += int(oa.get(f"auroc_id_vs_ood_{sc}") != ob.get(f"auroc_id_vs_ood_{sc}")) + int(oa.get(f"augrc_balanced_{sc}") != ob.get(f"augrc_balanced_{sc}"))
    out["n_frozen_rounded_fields_differing"] = n_round_diff
    return out


def run(dir_a: Path, dir_b: Path, out: Path | None) -> dict:
    ja = {p.stem: p for p in dir_a.glob("*.json") if not p.name.startswith("FAILED_")}
    jb = {p.stem: p for p in dir_b.glob("*.json") if not p.name.startswith("FAILED_")}
    common = sorted(set(ja) & set(jb))
    recs = [compare_record(ja[s], jb[s]) for s in common]
    summ = {"dir_a": str(dir_a), "dir_b": str(dir_b), "n_a": len(ja), "n_b": len(jb), "n_common": len(common),
            "only_in_a": sorted(set(ja) - set(jb)), "only_in_b": sorted(set(jb) - set(ja)),
            "checkpoint_identity_same_all": all(all(r["checkpoint_same"].values()) for r in recs) if recs else None,
            "sample_identity_same_all": all(all(r["id_test_same"].values()) for r in recs) if recs else None,
            "balancing_same_all": all(r["balancing_same"] for r in recs) if recs else None,
            "max_abs_diff_over_records": {k: max((r["max_abs_diff"][k] for r in recs if r["max_abs_diff"][k] is not None), default=None)
                                          for k in ("id_scores", "id_logits", "ood_scores", "ood_logits", "sigma_w", "cov_res")},
            "records_with_frozen_rounded_differences": sum(r["n_frozen_rounded_fields_differing"] > 0 for r in recs),
            "records": recs}
    if out:
        out.write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "records"}, indent=1, default=str))
    return summ


def self_test() -> None:
    d = Path("rn18_handoff_replication/outputs/fourshift_v2_rn18")
    if not d.exists() or not list(d.glob("*.json")):
        print("[parity] self-test SKIPPED (no version-2 outputs)"); return
    s = run(d, d, None)
    assert s["n_common"] == s["n_a"] and s["checkpoint_identity_same_all"] and s["balancing_same_all"]
    assert all(v == 0.0 for v in s["max_abs_diff_over_records"].values() if v is not None) and s["records_with_frozen_rounded_differences"] == 0
    print("[parity] self-test PASS: a directory is identical to itself")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    assert len(a.dirs) == 2, "two directories"
    run(Path(a.dirs[0]), Path(a.dirs[1]), a.out)


if __name__ == "__main__":
    main()
