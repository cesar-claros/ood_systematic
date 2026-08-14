#!/usr/bin/env python
"""Standalone tests for src/csfs/gradpca.py (the production CSF class).

Verifies the production implementation against independent autograd
references, mirroring the theorem-level checks in
documentation/x6_spectral_scripts/gradpca_equivalence_checks.py:

  1. E1 deployment cross-check: variant 'head_sum' scores == variant
     'act_cmeans' scores exactly, same retained k.
  2. 'head_sum' matches a reference pipeline built from per-sample autograd
     gradients of the summed logits of a real linear head.
  3. 'head_max' matches the same reference built from autograd gradients of
     the max logit (routing by the head's own argmax).
  4. save/load round-trip reproduces scores bit-for-bit.
  5. Scoring is chunk-size invariant.
  6. A missing train class raises a ValueError.

Runs WITHOUT the fd-shifts stack: `fd_shifts` and `src.utils` are stubbed
in sys.modules before importing gradpca.py directly from its file path.

Run:  .venv/bin/python tests/test_gradpca.py
"""
import importlib.util
import logging
import os
import sys
import tempfile
import types

import torch

torch.manual_seed(3)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Stub fd_shifts + src.utils, then import gradpca.py from its file path.
# ---------------------------------------------------------------------------
_fd = types.ModuleType("fd_shifts")
_fd.logger = logging.getLogger("test_gradpca")
sys.modules.setdefault("fd_shifts", _fd)

D_FEAT, C, N_FIT, N_EVAL = 17, 6, 480, 90
HEAD = torch.nn.Linear(D_FEAT, C).double()

_src = types.ModuleType("src")
_utils = types.ModuleType("src.utils")
_utils.get_model_and_last_layer = lambda module, study_name, return_model=True: (
    HEAD.weight.detach(), HEAD.bias.detach())
_src.utils = _utils
sys.modules.setdefault("src", _src)
sys.modules["src.utils"] = _utils

spec = importlib.util.spec_from_file_location(
    "gradpca_standalone", os.path.join(REPO, "src", "csfs", "gradpca.py"))
gradpca_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradpca_mod)
GradPCA = gradpca_mod.GradPCA


def make_cf(tmpdir):
    cf = types.SimpleNamespace()
    cf.data = types.SimpleNamespace(num_classes=C)
    cf.exp = types.SimpleNamespace(dir=tmpdir)
    return cf


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-30)).item()


# ---------------------------------------------------------------------------
# Reference implementation: per-sample autograd gradients + dual PCA
# (independent of the production accumulation identities).
# ---------------------------------------------------------------------------
def autograd_features(h_batch, agg):
    rows = []
    for i in range(h_batch.shape[0]):
        HEAD.zero_grad()
        logits = HEAD(h_batch[i:i + 1])
        s = logits.sum() if agg == "sum" else logits.max()
        gw, gb = torch.autograd.grad(s, [HEAD.weight, HEAD.bias])
        rows.append(torch.cat([gw.reshape(-1), gb]))
    return torch.stack(rows)


def reference_fit_score(feats_fit, labels_fit, feats_eval, eps=0.99):
    cmeans = torch.stack([feats_fit[labels_fit == c].mean(0) for c in range(C)])
    mean = cmeans.mean(0)
    M = cmeans - mean
    evals, evecs = torch.linalg.eigh(M @ M.T)
    evals, evecs = evals.flip(0).clamp(min=0.0), evecs.flip(1)
    k = int(((torch.cumsum(evals, 0) / evals.sum()) >= eps).nonzero()[0].item()) + 1
    U = M.T @ evecs[:, :k]
    U = U / U.norm(dim=0, keepdim=True)
    Z = feats_eval - mean
    return (Z @ U).pow(2).sum(1) / Z.pow(2).sum(1), k


def main():
    n_pass = 0
    h_fit = torch.randn(N_FIT, D_FEAT, dtype=torch.float64)
    labels_fit = torch.randint(0, C, (N_FIT,))
    for c in range(C):  # ensure every class occupied
        labels_fit[c] = c
    h_eval = torch.cat([torch.randn(N_EVAL // 2, D_FEAT, dtype=torch.float64),
                        torch.randn(N_EVAL // 2, D_FEAT, dtype=torch.float64) + 2.0])

    with tempfile.TemporaryDirectory() as tmp:
        cf = make_cf(tmp)

        # 1. E1 deployment cross-check: head_sum == act_cmeans
        g_sum = GradPCA(None, "vit", cf, variant="head_sum")
        g_sum.compute_GradPCA_params(h_fit, labels_fit)
        s_sum = g_sum.get_scores(h_eval)
        g_act = GradPCA(None, "vit", cf, variant="act_cmeans")
        g_act.compute_GradPCA_params(h_fit, labels_fit)
        s_act = g_act.get_scores(h_eval)
        err = rel_err(s_sum, s_act)
        assert err < 1e-12 and g_sum.n_components == g_act.n_components, err
        print(f"1 [PASS] E1: head_sum == act_cmeans (rel {err:.1e}, k={g_sum.n_components})")
        n_pass += 1

        # 2. head_sum vs autograd reference
        ref_scores, ref_k = reference_fit_score(
            autograd_features(h_fit, "sum"), labels_fit, autograd_features(h_eval, "sum"))
        err = rel_err(s_sum, ref_scores)
        assert err < 1e-12 and g_sum.n_components == ref_k, (err, g_sum.n_components, ref_k)
        print(f"2 [PASS] head_sum matches autograd reference (rel {err:.1e})")
        n_pass += 1

        # 3. head_max vs autograd reference
        g_max = GradPCA(None, "vit", cf, variant="head_max")
        g_max.compute_GradPCA_params(h_fit, labels_fit)
        s_max = g_max.get_scores(h_eval)
        ref_scores, ref_k = reference_fit_score(
            autograd_features(h_fit, "max"), labels_fit, autograd_features(h_eval, "max"))
        err = rel_err(s_max, ref_scores)
        assert err < 1e-12 and g_max.n_components == ref_k, (err, g_max.n_components, ref_k)
        print(f"3 [PASS] head_max matches autograd reference (rel {err:.1e}, k={ref_k})")
        n_pass += 1

        # 4. save/load round-trip
        g_max.save_params(filename="GradPCA_head_max_params_test")
        g_max2 = GradPCA(None, "vit", cf, variant="head_max")
        g_max2.load_params(filename="GradPCA_head_max_params_test")
        assert torch.equal(g_max2.get_scores(h_eval), s_max)
        print("4 [PASS] save/load round-trip bit-identical")
        n_pass += 1

        # 5. chunk invariance (different chunk shapes change BLAS summation
        # order, so tolerance-level equality is the correct expectation)
        g_sum.score_chunk = 7
        err = rel_err(g_sum.get_scores(h_eval), s_sum)
        assert err < 1e-13, err
        print(f"5 [PASS] scoring chunk-size invariant (rel {err:.1e})")
        n_pass += 1

        # 6. missing class raises
        try:
            bad = GradPCA(None, "vit", cf, variant="act_cmeans")
            bad.compute_GradPCA_params(h_fit[labels_fit != 0], labels_fit[labels_fit != 0])
            raise AssertionError("missing class did not raise")
        except ValueError:
            print("6 [PASS] missing train class raises ValueError")
            n_pass += 1

    print(f"\n{n_pass}/6 tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
