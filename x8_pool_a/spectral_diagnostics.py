"""Spectral diagnostics for representation-surgery selection (X6 Section 4).

Verbatim copy of documentation/x6_spectral_scripts/spectral_diagnostics.py
so the code repo is self-contained on the cluster (the documentation tree is
not part of this git repo). Keep the two copies in sync.

Prototype for the real-spectra campaign: operates on cached penultimate
activations H (N x D), labels y, and classifier weights W (C x D).
Self-test at bottom verifies each diagnostic on a planted configuration.
"""
import numpy as np


def mp_edge(sigma2: float, D: int, N: int) -> float:
    """Upper Marchenko-Pastur bulk edge for noise variance sigma2."""
    return sigma2 * (1 + np.sqrt(D / N)) ** 2


def estimate_bulk_sigma2(eigs: np.ndarray) -> float:
    """Robust bulk-noise estimate: median eigenvalue scaled by the MP median.

    For y <~ 0.5 the MP median is close to sigma2; the median is robust to a
    handful of spikes. Refinements (MP quantile matching) are a later pass.
    """
    return float(np.median(eigs))


def spike_census(H: np.ndarray, safety: float = 1.10) -> dict:
    """Count and size covariance spikes above the MP edge.

    Returns eigenvalues, bulk estimate, edge, spike count and strengths.
    """
    N, D = H.shape
    Hc = H - H.mean(0)
    eigs = np.linalg.eigvalsh(Hc.T @ Hc / N)
    s2 = estimate_bulk_sigma2(eigs)
    edge = safety * mp_edge(s2, D, N)
    spikes = eigs[eigs > edge]
    return {"eigs": eigs, "sigma2_bulk": s2, "edge": edge,
            "n_spikes": int(len(spikes)), "spike_strengths": spikes[::-1]}


def viability(H: np.ndarray, y: np.ndarray, C: int, safety: float = 1.10) -> dict:
    """P6.1 check for global and class-conditional subspace estimation."""
    N, D = H.shape
    census = spike_census(H, safety)
    s2 = census["sigma2_bulk"]
    omega_top = (census["spike_strengths"][0] - s2) / s2 if census["n_spikes"] else 0.0
    return {"omega_top": omega_top,
            "thr_global": np.sqrt(D / N),
            "thr_class": np.sqrt(C * D / N),
            "global_viable": omega_top > np.sqrt(D / N),
            "class_viable": omega_top > np.sqrt(C * D / N)}


def common_mode_fraction(H: np.ndarray) -> dict:
    """m-channel diagnostics: mean-direction energy fraction and amplitude CV."""
    mu = H.mean(0)
    g = mu / (np.linalg.norm(mu) + 1e-12)
    amp = H @ g
    return {"energy_fraction": float((amp ** 2).mean() / ((H ** 2).sum(1).mean() + 1e-12)),
            "amplitude_cv": float(amp.std() / (abs(amp.mean()) + 1e-12))}


def weight_top_gap(W: np.ndarray) -> float:
    """RankWeight dial: top singular value over the mean of the remainder.

    ~1 under exact NC (degenerate ETF spectrum, RankWeight ~inert);
    >> 1 signals a removable weight common mode.
    """
    s = np.linalg.svd(W, compute_uv=False)
    return float(s[0] / (s[1:].mean() + 1e-12))


def axis_alignment(MU: np.ndarray) -> float:
    """ASH dial: inverse participation ratio of class-mean mass over coordinates.

    Mean per-class inverse participation ratio: ~1/D for dense (rotated)
    means; ~1/k0 for k0-sparse means.
    """
    P = MU ** 2
    P = P / (P.sum(1, keepdims=True) + 1e-12)
    return float((P ** 2).sum(1).mean())


def token_top_ratio(X: np.ndarray) -> np.ndarray:
    """RankFeat dial: per-sample s1^2 / ||X||_F^2 for token/spatial maps X (n,T,D)."""
    s = np.linalg.svd(X, compute_uv=False)
    return (s[:, 0] ** 2) / (s ** 2).sum(1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    C, D, K, N = 10, 256, 8, 4000
    sig, tau2, R = 1.0, 3.0, 3.0
    Q = np.linalg.qr(rng.standard_normal((D, C + K)))[0]
    MU = ((np.eye(C) - np.ones((C, C)) / C) * np.sqrt(C / (C - 1)) * R) @ Q[:, :C].T
    B = Q[:, C:]
    y = rng.integers(0, C, N)
    H = MU[y] + (rng.standard_normal((N, K)) * np.sqrt(tau2)) @ B.T \
        + rng.standard_normal((N, D)) * sig
    cen = spike_census(H)
    via = viability(H, y, C)
    print(f"planted spikes = {C-1+K}, census = {cen['n_spikes']} "
          f"(bulk sigma2 = {cen['sigma2_bulk']:.2f}, true 1.0)")
    print(f"viability: omega_top={via['omega_top']:.2f} "
          f"thr_global={via['thr_global']:.2f} thr_class={via['thr_class']:.2f} "
          f"-> global={via['global_viable']} class={via['class_viable']}")
    g = np.abs(rng.standard_normal(D)); g /= np.linalg.norm(g)
    m = 2.0 + 0.5 * rng.standard_normal(N)
    cm0, cm1 = common_mode_fraction(H), common_mode_fraction(H + m[:, None] * g)
    print(f"common-mode energy fraction: without={cm0['energy_fraction']:.4f} "
          f"with planted m-channel={cm1['energy_fraction']:.4f} (cv={cm1['amplitude_cv']:.2f})")
    W_etf, W_cm = MU, MU + 1.5 * g[None, :]
    beta, s_etf = 1.5, R * np.sqrt(C / (C - 1))
    print(f"weight_top_gap: ETF={weight_top_gap(W_etf):.2f} (expect ~1), "
          f"common-mode={weight_top_gap(W_cm):.2f} "
          f"(expect ~beta*sqrt(C)/s_etf={beta*np.sqrt(C)/s_etf:.2f})")
    MU_sparse = np.zeros((C, D))
    for c in range(C):
        MU_sparse[c, rng.choice(D, 12, replace=False)] = 1.0
    Qr = np.linalg.qr(rng.standard_normal((D, D)))[0]
    print(f"axis_alignment (per-class IPR): sparse={axis_alignment(MU_sparse):.4f} "
          f"(~{1/12:.4f}), rotated={axis_alignment(MU_sparse @ Qr):.4f} (~{1/D:.4f})")
    Xn = rng.standard_normal((200, 32, D)) * 0.6
    r0, r1 = token_top_ratio(Xn), token_top_ratio(Xn + (2.5 * g)[None, None, :])
    print(f"token_top_ratio: noise-only={r0.mean():.3f} vs common-mode={r1.mean():.3f} "
          f"(excess over baseline => RankFeat viable)")
