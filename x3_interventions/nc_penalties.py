"""Differentiable Neural-Collapse penalties for the X3 intervention campaign.

Drop-in additions for the fd-shifts fork: each penalty attaches to the standard
cross-entropy objective as ``loss = ce + lam * penalty``. EMA class means are
tracked in feature space; gradients are stopped through the EMA buffers so the
penalties steer the live batch (and the head), not the running statistics.

Paradigm naming convention (mirrors ``rew`` suffix handling):
    etfreg_bb{arch}_do{d}_run{r}_lam{lam}   -> self_duality_penalty (I1)
    varreg_bb{arch}_do{d}_run{r}_lam{lam}   -> variability_penalty  (I2)
    eqnreg_bb{arch}_do{d}_run{r}_lam{lam}   -> equinorm_penalty     (I3)
Hard-mode I1 endpoint: replace the head with a fixed simplex-ETF matrix and a
learnable scalar scale (see fdshifts_integration.md).
"""
import torch


class EMAClassMeans(torch.nn.Module):
    """Running class means of penultimate features, momentum-updated per batch."""

    def __init__(self, n_classes: int, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("means", torch.zeros(n_classes, dim))
        self.register_buffer("initialized", torch.zeros(n_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        for c in labels.unique():
            batch_mean = feats[labels == c].mean(0)
            if self.initialized[c]:
                self.means[c].mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
            else:
                self.means[c].copy_(batch_mean)
                self.initialized[c] = True


def self_duality_penalty(weight: torch.Tensor, ema_means: torch.Tensor) -> torch.Tensor:
    """I1: the paper's self-duality metric as a loss, EMA means detached."""
    m = ema_means - ema_means.mean(0, keepdim=True)
    a = weight / weight.norm(p="fro").clamp_min(1e-8)
    b = (m / m.norm(p="fro").clamp_min(1e-8)).detach()
    return (a - b).pow(2).sum()


def variability_penalty(feats: torch.Tensor, labels: torch.Tensor,
                        ema_means: torch.Tensor) -> torch.Tensor:
    """I2: batch within-class scatter over EMA between-class scatter (NC1 proxy)."""
    mu = ema_means.detach()
    within = (feats - mu[labels]).pow(2).sum(1).mean()
    centered = mu - mu.mean(0, keepdim=True)
    between = centered.pow(2).sum(1).mean().clamp_min(1e-8)
    return within / between


def equinorm_penalty(feats: torch.Tensor, labels: torch.Tensor,
                     ema_means: torch.Tensor, momentum: float = 0.9) -> torch.Tensor:
    """I3: squared coefficient of variation of class-mean norms.

    Uses a convex combination of the detached EMA mean and the live batch mean per
    class so the batch receives gradient while the statistic stays stable.
    """
    norms = []
    for c in labels.unique():
        live = feats[labels == c].mean(0)
        blended = momentum * ema_means[c].detach() + (1 - momentum) * live
        norms.append(blended.norm())
    norms = torch.stack(norms)
    return norms.var(unbiased=False) / norms.mean().pow(2).clamp_min(1e-8)


PENALTIES = {"etfreg": "self_duality", "varreg": "variability", "eqnreg": "equinorm"}
