#!/usr/bin/env python
"""Registry-level tests for the GradPCA family wiring in src/csf_pipeline.py.

Imports csf_pipeline with its heavy dependencies (fd_shifts, faiss-backed
CSFs, scores_funcs, ...) stubbed out, then checks that the new families
resolve, gate, and filter exactly like any other plain-mode family:

  1. Aliases normalize to the canonical names (incl. 'actpca_classmeans').
  2. build_active on the three families adds Temperature but NOT
     ProjectionFiltering (they are not PF dependents).
  3. _detect_mode parses all three confids keys as 'plain' (the
     'ActPCA_cmeans' spelling exists precisely to dodge the '_class'
     substring rule).
  4. filter_confids keeps exactly the requested keys.
  5. skip-csfs removes a single variant without touching the others.

Run:  .venv/bin/python tests/test_gradpca_registry.py
"""
import logging
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ---------------------------------------------------------------------------
# Stub every heavy import csf_pipeline pulls in at module level.
# ---------------------------------------------------------------------------
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_fd = _stub("fd_shifts", logger=logging.getLogger("test"))
_stub("fd_shifts.analysis", metrics=types.SimpleNamespace())
_fd.analysis = sys.modules["fd_shifts.analysis"]

_csf_names = [
    "ClassTypicalMatching", "EntropyScores", "fDBD", "GradNorm", "GradPCA",
    "KernelPCA", "MahalanobisDistance", "MahalanobisPP", "NCI", "NeCo",
    "NNGuide", "pNML", "ProjectionFiltering", "ResidualScore",
    "TemperatureScaling", "ViMScore",
]
_stub("src.csfs", **{n: type(n, (), {}) for n in _csf_names})
_stub("src.neural_collapse", NeuralCollapseMetrics=type("NeuralCollapseMetrics", (), {}))
_stub("src.scores_funcs")
_stub("src.rc_stats", RiskCoverageStats=type("RiskCoverageStats", (), {}))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "csf_pipeline_standalone", os.path.join(REPO, "src", "csf_pipeline.py"))
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def main():
    n_pass = 0

    # 1. alias resolution
    assert cp.normalize_family("gradpca_head_sum") == "GradPCA_head_sum"
    assert cp.normalize_family("GradPCA_head_max") == "GradPCA_head_max"
    assert cp.normalize_family("actpca_cmeans") == "ActPCA_cmeans"
    assert cp.normalize_family("actpca_classmeans") == "ActPCA_cmeans"
    print("1 [PASS] aliases normalize to canonical family names")
    n_pass += 1

    # 2. build_active: Temperature added, ProjectionFiltering NOT triggered
    active = cp.build_active(csfs=["gradpca_head_sum", "gradpca_head_max", "actpca_classmeans"])
    assert active == {"GradPCA_head_sum", "GradPCA_head_max", "ActPCA_cmeans", "Temperature"}, active
    print("2 [PASS] build_active adds Temperature only (no PF setup)")
    n_pass += 1

    # 3. mode detection
    for key in ("GradPCA_head_sum", "GradPCA_head_max", "ActPCA_cmeans"):
        assert cp._detect_mode(key) == "plain", key
    print("3 [PASS] all three confids keys parse as plain mode")
    n_pass += 1

    # 4. filter_confids keeps exactly the requested keys
    confids = {
        "GradPCA_head_sum": torch.zeros(2),
        "GradPCA_head_max": torch.zeros(2),
        "ActPCA_cmeans": torch.zeros(2),
        "Maha": torch.zeros(2),
        "Maha_global": torch.zeros(2),
        "GradNorm": torch.zeros(2),
    }
    kept = cp.filter_confids(confids, {"GradPCA_head_sum", "ActPCA_cmeans"}, {"plain"})
    assert set(kept) == {"GradPCA_head_sum", "ActPCA_cmeans"}, set(kept)
    print("4 [PASS] filter_confids keeps exactly the requested keys")
    n_pass += 1

    # 5. skip-csfs removes one variant only
    active = cp.build_active(skip_csfs=["gradpca_head_max"])
    assert "GradPCA_head_max" not in active
    assert {"GradPCA_head_sum", "ActPCA_cmeans"} <= active
    print("5 [PASS] --skip-csfs removes a single variant")
    n_pass += 1

    print(f"\n{n_pass}/5 tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
