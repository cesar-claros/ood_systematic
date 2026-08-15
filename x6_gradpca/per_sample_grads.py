"""Exact per-sample gradients for a single Conv2d or Linear layer via hooks.

The Goodfellow trick: for a layer y = W a (+ b), the per-sample gradient of a
scalar S = sum_i A_i (one summand per batch sample, samples independent
through the network, which holds for eval-mode BN, per-sample LayerNorm, and
within-sample attention) is

    dA_i/dW = g_i (x) a_i        (outer product of grad_output and input)
    dA_i/db = sum_positions g_i

so ONE batched forward + ONE batched backward yields all per-sample
gradients, with no functorch/vmap dependency (torch >= 1.10 suffices; the
paper container's torch 1.13 is fine).

Usage:
    cap = LayerGradCapture(layer)          # registers hooks
    ... forward the model on a batch x (graph reaches `layer`) ...
    scalar = per_sample_scalars.sum()      # e.g. sum-of-logits or max-logit
    scalar.backward()                      # (retain_graph=True for a 2nd agg)
    G = cap.per_sample_grads()             # (B, P) float32, [W.flatten(), b]
    cap.remove()                           # detach hooks when done

Only the target layer's parameters need requires_grad=True; freezing all
other parameters keeps the backward pass short (it stops at the layer) and
skips graph construction below it entirely.

Flattening convention: [weight.reshape(-1), bias] per layer, matching the
row-major layout used by src/csfs/gradpca.py and the E-series theorems.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class LayerGradCapture:
    """Captures (input, grad_output) for one Conv2d/Linear layer and forms
    exact per-sample gradients of a per-sample-summed scalar."""

    def __init__(self, layer: torch.nn.Module):
        if isinstance(layer, torch.nn.Conv2d):
            if layer.groups != 1:
                raise NotImplementedError("grouped conv not supported")
        elif not isinstance(layer, torch.nn.Linear):
            raise NotImplementedError(f"unsupported layer type {type(layer).__name__}")
        self.layer = layer
        self.has_bias = layer.bias is not None
        self._a = None
        self._g = None
        self._h_fwd = layer.register_forward_hook(self._save_input)
        self._h_bwd = layer.register_full_backward_hook(self._save_grad_output)

    def _save_input(self, module, inputs, output):
        self._a = inputs[0].detach()

    def _save_grad_output(self, module, grad_inputs, grad_outputs):
        self._g = grad_outputs[0].detach()

    @property
    def flat_dim(self) -> int:
        return self.layer.weight.numel() + (self.layer.bias.numel() if self.has_bias else 0)

    def param_names_shapes(self, prefix: str = "") -> list:
        out = [(f"{prefix}weight", tuple(self.layer.weight.shape))]
        if self.has_bias:
            out.append((f"{prefix}bias", tuple(self.layer.bias.shape)))
        return out

    def per_sample_grads(self) -> torch.Tensor:
        """(B, P) per-sample gradients from the captured pair; call after a
        backward. Consumes the captured grad_output (a second backward on the
        same forward refills it; the input capture persists)."""
        assert self._a is not None, "no forward captured"
        assert self._g is not None, "no backward captured (did the graph reach this layer?)"
        a, g, layer = self._a, self._g, self.layer
        B = a.shape[0]
        if isinstance(layer, torch.nn.Conv2d):
            ua = F.unfold(a, layer.kernel_size, dilation=layer.dilation,
                          padding=layer.padding, stride=layer.stride)  # (B, Cin*kh*kw, L)
            go = g.reshape(B, layer.out_channels, -1)                  # (B, Cout, L)
            gw = torch.bmm(go, ua.transpose(1, 2))                     # (B, Cout, Cin*kh*kw)
            parts = [gw.reshape(B, -1)]
            if self.has_bias:
                parts.append(go.sum(dim=-1))
        else:  # Linear, possibly applied per token: (B, F) or (B, T, F)
            ai = a.reshape(B, -1, layer.in_features)
            go = g.reshape(B, -1, layer.out_features)
            gw = torch.einsum("bto,bti->boi", go, ai)                  # (B, out, in)
            parts = [gw.reshape(B, -1)]
            if self.has_bias:
                parts.append(go.sum(dim=1))
        self._g = None
        return torch.cat(parts, dim=1)

    def remove(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
        self._a = self._g = None


def aggregation_scalar(logits: torch.Tensor, num_classes: int, agg: str) -> torch.Tensor:
    """Per-sample-summed aggregation scalar. 'sum' = sum of the first
    num_classes logits (v = 1; a DG reservation logit is excluded);
    'max' = maximum over the first num_classes logits."""
    l = logits[:, :num_classes]
    if agg == "sum":
        return l.sum()
    if agg == "max":
        return l.max(dim=1).values.sum()
    raise ValueError(f"unknown aggregation {agg!r}")


def reference_per_sample_grads(forward_fn, layer: torch.nn.Module, x: torch.Tensor,
                               num_classes: int, agg: str) -> torch.Tensor:
    """Slow reference: per-sample autograd loop over the same layer. Used by
    the unit tests and the stage's runtime self-check."""
    params = [layer.weight] + ([layer.bias] if layer.bias is not None else [])
    rows = []
    for i in range(x.shape[0]):
        logits = forward_fn(x[i: i + 1])
        s = aggregation_scalar(logits, num_classes, agg)
        gs = torch.autograd.grad(s, params, retain_graph=False)
        rows.append(torch.cat([g.reshape(-1) for g in gs]))
    return torch.stack(rows)
