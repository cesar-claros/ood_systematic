"""Head-side logit diagnostics (raw and temperature-scaled).

Mirrors the structure of ``src.scores_methods.NeuralCollapseMetrics``:
a small class that takes the run's ``cf``, computes a flat dict of raw and
T*-scaled head-side metrics, stores them on ``self``, and supports
``save_params`` / ``load_params`` to ``{cf.exp.dir}/params/{filename}.pt``.

Raw diagnostics operate on the untouched logits; scaled diagnostics operate
on logits / T*. Keep both: raw describes the trained head directly (what
CSFs read when ``temp_scaled=False``); scaled describes the post-calibration
state CSFs read when ``temp_scaled=True``. T* itself is stored as a scalar
summary of miscalibration.
"""
from __future__ import annotations

import math
import os

import torch
from loguru import logger
from torch.nn import functional as F


class HeadLogitDiagnostics:

    def __init__(self, cf):
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.temperature = None
        self.diagnostics = None

    @staticmethod
    def _scalar_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
        x = x.detach().cpu().to(dtype=torch.float64).flatten()
        return {
            f"{prefix}_mean": float(x.mean()),
            f"{prefix}_std":  float(x.std(unbiased=False)),
            f"{prefix}_p50":  float(x.median()),
        }

    def _softmax_diagnostics(self, logits: torch.Tensor, prefix: str) -> dict[str, float]:
        logits = logits.detach().cpu().to(dtype=torch.float64)
        K = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=1)
        max_prob = probs.max(dim=1).values
        k_top = min(4, K)
        topk = torch.topk(logits, k=k_top, dim=1).values
        margin = topk[:, 0] - topk[:, 1]
        logit_norm = logits.norm(dim=1)
        kl_uniform = math.log(K) - entropy

        out: dict[str, float] = {}
        out.update(self._scalar_stats(entropy,    f"{prefix}_entropy"))
        out.update(self._scalar_stats(max_prob,   f"{prefix}_maxprob"))
        out.update(self._scalar_stats(margin,     f"{prefix}_margin"))
        out.update(self._scalar_stats(logit_norm, f"{prefix}_logitnorm"))
        out.update(self._scalar_stats(kl_uniform, f"{prefix}_kl_uniform"))
        if k_top >= 3:
            out.update(self._scalar_stats(topk[:, 1] - topk[:, 2], f"{prefix}_top23margin"))
        if k_top >= 4:
            out.update(self._scalar_stats(topk[:, 2] - topk[:, 3], f"{prefix}_top34margin"))
        return out

    def _logit_classmean_geometry(self, logits: torch.Tensor,
                                  labels: torch.Tensor) -> dict[str, float]:
        """NC-on-logits: class-mean logit vector geometry.

        All quantities here are invariant under uniform rescaling of the logits
        (cosines, CV of norms, trace ratio), so no separate ``scaled`` variant is
        needed - one report covers raw and T*-scaled logits jointly.
        """
        logits = logits.detach().cpu().to(dtype=torch.float64)
        labels = labels.detach().cpu().to(dtype=torch.long)
        unique = torch.unique(labels)
        nC = int(unique.numel())
        if nC < 2:
            return {}

        K = logits.shape[1]
        mu = torch.zeros(nC, K, dtype=torch.float64)
        within_ss = 0.0
        for i, c in enumerate(unique):
            mask = labels == c
            zc = logits[mask]
            mu[i] = zc.mean(dim=0)
            within_ss += float(((zc - mu[i]) ** 2).sum())
        mu_G = mu.mean(dim=0)
        m = mu - mu_G

        norms = m.norm(dim=1)
        mean_norm = float(norms.mean())
        cv_norm = float(norms.std(unbiased=False) / max(mean_norm, 1e-12))

        norms_safe = norms.clamp_min(1e-12)
        m_unit = m / norms_safe.unsqueeze(1)
        cos_mat = m_unit @ m_unit.T
        iu = torch.triu_indices(nC, nC, offset=1)
        cosines = cos_mat[iu[0], iu[1]]
        target = -1.0 / (nC - 1)
        cos_mse = float(((cosines - target) ** 2).mean())
        cos_max_dev = float((cosines - target).abs().max())
        cos_mean = float(cosines.mean())

        between_ss = float((m ** 2).sum())
        wb_trace_ratio = (within_ss / logits.shape[0]) / max(between_ss / nC, 1e-12)

        return {
            "logit_classmean_cv_norm":     cv_norm,
            "logit_classmean_cos_mse":     cos_mse,
            "logit_classmean_cos_max_dev": cos_max_dev,
            "logit_classmean_cos_mean":    cos_mean,
            "logit_within_between_ratio":  wb_trace_ratio,
            "logit_classmean_mean_norm":   mean_norm,
            "n_classes_seen":              float(nC),
        }

    def _logit_covariance_geometry(self, logits: torch.Tensor) -> dict[str, float]:
        """Effective dimensionality of the sample-level logit covariance.

        Eigenvalue ratios are scale-invariant, so no separate ``scaled`` report.
        """
        logits = logits.detach().cpu().to(dtype=torch.float64)
        N = logits.shape[0]
        centered = logits - logits.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / max(N - 1, 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        total = float(eig.sum())
        if total <= 0:
            return {"logit_cov_participation": float("nan"),
                    "logit_cov_effrank":       float("nan")}
        p = eig / total
        participation = float(1.0 / (p * p).sum().clamp_min(1e-12))
        p_pos = p[p > 0]
        spectral_entropy = float(-(p_pos * p_pos.log()).sum())
        effrank = float(math.exp(spectral_entropy))
        return {
            "logit_cov_participation": participation,
            "logit_cov_effrank":       effrank,
        }

    def compute_head_diagnostics(self,
                                 logits: torch.Tensor,
                                 labels: torch.Tensor,
                                 temperature: float,
                                 logits_dist: torch.Tensor | None = None) -> None:
        """Compute raw + T*-scaled head-side diagnostics and store on ``self``.

        Parameters
        ----------
        logits : (N, K) tensor
            Raw classifier logits, no T* applied.
        labels : (N,) tensor
            Ground-truth labels.
        temperature : float
            Previously fit T* from the validation set.
        logits_dist : (N, K, T) tensor, optional
            MCD-sampled logits. When provided, ``raw_mcd_*`` / ``scaled_mcd_*``
            diagnostics are computed on the mean-over-samples logits.
        """
        logger.info("Head Logit Diagnostics: Computing raw and T*-scaled head-side metrics...")
        T = float(temperature)
        self.temperature = T
        out: dict[str, float] = {"temperature": T}
        out.update(self._softmax_diagnostics(logits,     prefix="raw"))
        out.update(self._softmax_diagnostics(logits / T, prefix="scaled"))
        out.update(self._logit_classmean_geometry(logits, labels))
        out.update(self._logit_covariance_geometry(logits))
        if logits_dist is not None:
            mean_logits = logits_dist.mean(dim=2)
            out.update(self._softmax_diagnostics(mean_logits,     prefix="raw_mcd"))
            out.update(self._softmax_diagnostics(mean_logits / T, prefix="scaled_mcd"))

        logger.info(f"Temperature T*: {T}")
        logger.info(f"Raw maxprob (mean): {out.get('raw_maxprob_mean')}")
        logger.info(f"Scaled maxprob (mean): {out.get('scaled_maxprob_mean')}")
        logger.info(f"Raw entropy (mean): {out.get('raw_entropy_mean')}")
        logger.info(f"Scaled entropy (mean): {out.get('scaled_entropy_mean')}")
        logger.info(f"Logit class-mean cos MSE: {out.get('logit_classmean_cos_mse')}")
        logger.info(f"Logit class-mean CV norm: {out.get('logit_classmean_cv_norm')}")
        logger.info(f"Logit within/between ratio: {out.get('logit_within_between_ratio')}")
        logger.info(f"Logit cov participation: {out.get('logit_cov_participation')}")
        logger.info(f"Logit cov effrank: {out.get('logit_cov_effrank')}")

        self.diagnostics = out

    def save_params(self, path: str | None = None,
                    filename: str = 'HeadLogit_params'):
        assert self.diagnostics is not None, 'Head Logit diagnostics have not been computed...'
        assert self.temperature is not None, 'Temperature has not been set...'
        params_dict = {
            'temperature': self.temperature,
            'diagnostics': self.diagnostics,
        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Head Logit Diagnostics: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path: str | None = None,
                    filename: str = 'HeadLogit_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Head Logit Diagnostics: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.temperature = params_dict['temperature']
        self.diagnostics = params_dict['diagnostics']
