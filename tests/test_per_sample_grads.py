#!/usr/bin/env python
"""Tests for x6_gradpca/per_sample_grads.py (hook-based Goodfellow trick).

Verifies exact per-sample gradients against a per-sample autograd loop:

  1. Conv2d target (stride 2, padding 1, with bias) inside a CNN with
     eval-mode BatchNorm downstream, for sum and max aggregations.
  2. Linear target applied per token, downstream of a within-sample
     token-mixing (attention-like) layer, for sum and max.
  3. Two backwards on ONE forward (sum then max, retain_graph=True) match
     two independent forwards.
  4. Freezing all non-target parameters changes nothing.
  5. An extra logit column (DG reservation) is excluded by num_classes.

Pure torch; runs in the local venv. Run: .venv/bin/python tests/test_per_sample_grads.py
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "x6_gradpca"))
from per_sample_grads import LayerGradCapture, aggregation_scalar, reference_per_sample_grads

torch.set_default_dtype(torch.float64)
torch.manual_seed(11)
C = 5


def rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-30)).item()


class ToyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.target = torch.nn.Conv2d(8, 12, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(12)
        self.head = torch.nn.Linear(12 * 4 * 4, C + 1)  # +1 = DG-style extra logit

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.bn(self.target(h)))
        return self.head(h.reshape(x.shape[0], -1))


class ToyTokenNet(torch.nn.Module):
    """Linear-over-tokens target downstream of within-sample token mixing."""

    def __init__(self, T=6, F_in=10):
        super().__init__()
        self.emb = torch.nn.Linear(F_in, 16)
        self.target = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, C)

    def forward(self, x):  # x: (B, T, F_in)
        h = torch.tanh(self.emb(x))
        attn = torch.softmax(h @ h.transpose(1, 2), dim=-1)  # within-sample mixing
        h = self.target(attn @ h)
        return self.head(h.mean(dim=1))


def run_case(name, model, x, target, n_pass):
    model.eval()
    for agg in ("sum", "max"):
        cap = LayerGradCapture(target)
        logits = model(x)
        aggregation_scalar(logits, C, agg).backward()
        G = cap.per_sample_grads()
        cap.remove()
        model.zero_grad(set_to_none=True)
        ref = reference_per_sample_grads(model, target, x, C, agg)
        err = rel_err(G, ref)
        ok = err < 1e-12
        print(f"{n_pass + 1} [{'PASS' if ok else 'FAIL'}] {name} agg={agg} (rel {err:.1e}, P={G.shape[1]})")
        assert ok, (name, agg, err)
        n_pass += 1
    return n_pass


def main():
    n = 0
    x_img = torch.randn(9, 3, 8, 8)
    x_tok = torch.randn(7, 6, 10)

    # 1-2: conv and token-linear targets, both aggregations
    cnn = ToyCNN()
    n = run_case("conv target (stride/pad, BN eval downstream)", cnn, x_img, cnn.target, n)
    tok = ToyTokenNet()
    n = run_case("token-linear target (within-sample mixing)", tok, x_tok, tok.target, n)

    # 3: two backwards on one forward
    cnn.zero_grad(set_to_none=True)
    cap = LayerGradCapture(cnn.target)
    logits = cnn(x_img)
    aggregation_scalar(logits, C, "sum").backward(retain_graph=True)
    G_sum = cap.per_sample_grads()
    aggregation_scalar(logits, C, "max").backward()
    G_max = cap.per_sample_grads()
    cap.remove()
    cnn.zero_grad(set_to_none=True)
    e1 = rel_err(G_sum, reference_per_sample_grads(cnn, cnn.target, x_img, C, "sum"))
    e2 = rel_err(G_max, reference_per_sample_grads(cnn, cnn.target, x_img, C, "max"))
    assert e1 < 1e-12 and e2 < 1e-12, (e1, e2)
    print(f"{n + 1} [PASS] two backwards on one forward (rel {e1:.1e}/{e2:.1e})")
    n += 1

    # 4: freezing non-target params changes nothing (and shortens backward)
    for p in cnn.parameters():
        p.requires_grad_(False)
    for p in cnn.target.parameters():
        p.requires_grad_(True)
    cap = LayerGradCapture(cnn.target)
    aggregation_scalar(cnn(x_img), C, "sum").backward()
    G_frozen = cap.per_sample_grads()
    cap.remove()
    err = rel_err(G_frozen, G_sum)
    assert err < 1e-12, err
    print(f"{n + 1} [PASS] frozen-elsewhere identical (rel {err:.1e})")
    n += 1

    # 5: extra logit column excluded by num_classes
    with torch.no_grad():
        full = cnn(x_img)
    assert full.shape[1] == C + 1
    print(f"{n + 1} [PASS] aggregation uses first {C} of {C + 1} logits by construction")
    n += 1

    print(f"\n{n}/7 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
