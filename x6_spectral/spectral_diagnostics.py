"""Spectral diagnostics for representation-surgery selection (X6, pass 3).

Operates on cached penultimate activations H (N x D), labels y, and classifier
weights W (C x D). Corrected after the X6 review:

- bulk noise: median of the nonzero eigenvalues divided by the numerical
  median of the Marchenko-Pastur law at aspect ratio y (valid for y >= 1 as
  well), with a deflation pass that removes above-edge spikes;
- spike strength: inversion of the outlier map
  lam = sigma2 (1 + omega)(1 + y / omega), not bulk subtraction, with the
  predicted eigenvector overlap attached per spike;
- class-level viability: per-class within-class spectra at y_c = D / N_c
  (after per-class centering there is no class-mean spike; the class-level
  condition concerns within-class structure);
- selection inputs: class-mean-span and classifier-row-space alignment per
  spike, and split-half projector stability, so a score-specific rule can
  choose components by utility instead of one MP-edge cut.

The bulk model assumes near-isotropic noise after optional per-coordinate
standardization (standardize=True mirrors the deployed pipeline's
TorchStandardScaler); anisotropic-bulk corrections are future work.
Self-test at the bottom verifies each estimator on planted configurations.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

_EIG_TOL = 1e-10


@lru_cache(maxsize=256)
def mp_median(y: float) -> float:
    """Median of the nonzero part of the Marchenko-Pastur law (sigma2 = 1).

    Args:
        y: Aspect ratio D / N; for y > 1 the point mass at zero is excluded
            and the median refers to the continuous part.

    Returns:
        The conditional median of the continuous MP density.
    """
    if y <= 1e-4:
        return 1.0
    lo, hi = (1 - np.sqrt(y)) ** 2, (1 + np.sqrt(y)) ** 2

    def density(x: float) -> float:
        return np.sqrt(max((hi - x) * (x - lo), 0.0)) / (2 * np.pi * x * y)

    total = min(1.0, 1.0 / y)

    def cdf_minus_half(m: float) -> float:
        return quad(density, lo, m, limit=200)[0] - total / 2

    return float(brentq(cdf_minus_half, lo + 1e-12, hi - 1e-12))


def mp_edge(sigma2: float, dim: int, n: int) -> float:
    """Upper Marchenko-Pastur bulk edge for noise variance sigma2."""
    return sigma2 * (1 + np.sqrt(dim / n)) ** 2


def estimate_bulk_sigma2(eigs: np.ndarray, y: float,
                         safety: float = 1.05) -> float:
    """MP-median-corrected bulk estimate with one deflation pass.

    Args:
        eigs: Eigenvalues of the sample covariance (any order).
        y: Aspect ratio D / N.
        safety: Multiplier on the MP edge used to peel off spikes before
            re-estimating.

    Returns:
        Estimated bulk noise variance sigma2.
    """
    nonzero = np.sort(eigs[eigs > _EIG_TOL * max(eigs.max(), 1.0)])
    med_mp = mp_median(round(float(y), 6))
    sigma2 = float(np.median(nonzero)) / med_mp
    for _ in range(2):
        core = nonzero[nonzero <= safety * sigma2 * (1 + np.sqrt(y)) ** 2]
        if len(core) < max(10, len(nonzero) // 2):
            break
        sigma2 = float(np.median(core)) / med_mp
    return sigma2


def invert_spike(lam_emp: float, sigma2: float, y: float
                 ) -> tuple[float, float] | None:
    """Population spike strength and overlap from an empirical outlier.

    Inverts lam = sigma2 (1 + omega)(1 + y / omega) for omega > sqrt(y).

    Args:
        lam_emp: Empirical outlier eigenvalue.
        sigma2: Bulk noise variance.
        y: Aspect ratio D / N.

    Returns:
        Tuple (omega, predicted overlap), or None when lam_emp is not above
        the bulk edge (the inversion has no real solution beyond sqrt(y)).
    """
    ratio = lam_emp / sigma2
    disc = (ratio - 1 - y) ** 2 - 4 * y
    if disc <= 0:
        return None
    omega = ((ratio - 1 - y) + np.sqrt(disc)) / 2
    if omega <= np.sqrt(y):
        return None
    overlap = (1 - y / omega**2) / (1 + y / omega)
    return float(omega), float(overlap)


def _center_scale(h: np.ndarray, standardize: bool) -> np.ndarray:
    centered = h - h.mean(0)
    if standardize:
        centered = centered / (centered.std(0) + 1e-12)
    return centered


def spike_census(h: np.ndarray, safety: float = 1.05,
                 standardize: bool = False) -> dict:
    """Count and size covariance spikes above the corrected MP edge.

    Args:
        h: Activations, shape (N, D).
        safety: Multiplier on the MP edge before declaring a spike.
        standardize: Per-coordinate z-scoring before the census, matching
            the deployed pipeline's scaler (correlation-matrix census).

    Returns:
        Dict with eigs, sigma2_bulk, edge, n_spikes, spike_strengths
        (descending empirical eigenvalues), spike_omegas, spike_overlaps, y.
    """
    n, dim = h.shape
    y = dim / n
    centered = _center_scale(h, standardize)
    eigs = np.linalg.eigvalsh(centered.T @ centered / n)
    sigma2 = estimate_bulk_sigma2(eigs, y, safety)
    edge = safety * mp_edge(sigma2, dim, n)
    spikes = np.sort(eigs[eigs > edge])[::-1]
    inverted = [invert_spike(s, sigma2, y) for s in spikes]
    return {"eigs": eigs, "sigma2_bulk": sigma2, "edge": edge, "y": y,
            "n_spikes": int(len(spikes)), "spike_strengths": spikes,
            "spike_omegas": np.array([o[0] if o else np.nan
                                      for o in inverted]),
            "spike_overlaps": np.array([o[1] if o else np.nan
                                        for o in inverted])}


def class_censuses(h: np.ndarray, labels: np.ndarray, n_classes: int,
                   safety: float = 1.05, standardize: bool = False
                   ) -> list[dict]:
    """Within-class spike census per class (per-class centering inside)."""
    out = []
    for c in range(n_classes):
        block = h[labels == c]
        if len(block) < 8:
            out.append({"n_spikes": 0, "spike_omegas": np.array([]),
                        "y": np.inf, "n_c": len(block)})
            continue
        census = spike_census(block, safety, standardize)
        census["n_c"] = len(block)
        out.append(census)
    return out


def _mean_span(h: np.ndarray, labels: np.ndarray, n_classes: int
               ) -> np.ndarray:
    mu_c = np.stack([h[labels == c].mean(0) for c in range(n_classes)])
    centered = (mu_c - mu_c.mean(0)).T
    return np.linalg.qr(centered, mode="reduced")[0][:, :n_classes - 1]


def viability(h: np.ndarray, labels: np.ndarray, n_classes: int,
              safety: float = 1.05, standardize: bool = False) -> dict:
    """Recovery-condition check, global and class-conditional (P6.1').

    Global viability asks for an above-edge spike whose eigenvector aligns
    with the class-mean span; class viability asks each class's within-class
    spectrum for spikes above its own threshold sqrt(D / N_c). Both are
    recovery statements, not benefit guarantees (the benefit rule is the
    capture crossing of P6.2').
    """
    n, dim = h.shape
    centered = _center_scale(h, standardize)
    eigs, vecs = np.linalg.eigh(centered.T @ centered / n)
    y = dim / n
    sigma2 = estimate_bulk_sigma2(eigs, y, safety)
    edge = safety * mp_edge(sigma2, dim, n)
    span = _mean_span(centered, labels, n_classes)
    spike_idx = np.where(eigs > edge)[0]
    omega_top, omega_aligned = 0.0, 0.0
    n_residue = 0
    for idx in spike_idx:
        inv = invert_spike(float(eigs[idx]), sigma2, y)
        if inv is None:
            continue
        omega_top = max(omega_top, inv[0])
        alignment = float(np.linalg.norm(span.T @ vecs[:, idx]) ** 2)
        if alignment > 0.5:
            omega_aligned = max(omega_aligned, inv[0])
        else:
            n_residue += 1
    per_class = class_censuses(h, labels, n_classes, safety, standardize)
    omega_c, thr_c = [], []
    for census in per_class:
        omegas = census["spike_omegas"]
        omega_c.append(float(np.nanmedian(omegas)) if len(omegas) else 0.0)
        thr_c.append(np.sqrt(census["y"]))
    viable_frac = float(np.mean([o > t for o, t in zip(omega_c, thr_c)]))
    return {"omega_top": omega_top, "omega_mean_aligned": omega_aligned,
            "thr_global": np.sqrt(dim / n),
            "thr_class": np.sqrt(n_classes * dim / n),
            "global_viable": omega_aligned > np.sqrt(dim / n),
            "class_viable": viable_frac >= 0.5,
            "frac_classes_viable": viable_frac,
            "omega_per_class": omega_c, "thr_per_class": thr_c,
            "n_residue_spikes": n_residue}


def split_half_stability(h: np.ndarray, k: int, n_splits: int = 5,
                         seed: int = 0) -> tuple[float, float]:
    """Mean and sd of top-k projector agreement between disjoint halves.

    This is STABILITY (agreement of two independent estimates), not the
    unobservable population capture tr(P P_hat)/k; under independent noise it
    behaves roughly like squared signal overlap plus a random-subspace floor
    of k/D. Compare against that null floor, and do not substitute it for
    the capture a in the P6.2 crossing rule. Near-threshold components show
    low stability regardless of what the asymptotic formula says.
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    captures = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        halves = []
        for part in (perm[:n // 2], perm[n // 2:]):
            block = h[part] - h[part].mean(0)
            cov = block.T @ block / len(part)
            halves.append(np.linalg.eigh(cov)[1][:, -k:])
        captures.append(np.linalg.norm(halves[0].T @ halves[1]) ** 2 / k)
    return float(np.mean(captures)), float(np.std(captures))


def class_projector_heterogeneity(h: np.ndarray, labels: np.ndarray,
                                  n_classes: int, k: int, n_splits: int = 3,
                                  seed: int = 0) -> dict:
    """Split-sample evidence for class-dependent within-class subspaces.

    Replaces the v2 harness statistic that measured centered class means
    outside their own span and was therefore algebraically zero. Per split,
    per-class top-k projectors are estimated on disjoint halves; the
    cross-class distance uses independent halves (P_c^(1) vs P_d^(2), c != d)
    and the within-class distance across halves (P_c^(1) vs P_c^(2)) is the
    estimation-noise floor. Distances are |P - Q|_F^2 / (2k), in [0, 1].

    Returns:
        Dict with cross, within, within_sd, heterogeneity = cross - within
        (<= 0 means no evidence of class-dependent subspaces), and
        n_classes_used (classes with at least 2(k+2) samples).
    """
    local = np.random.default_rng(seed)
    cross_vals: list[float] = []
    within_vals: list[float] = []
    used = 0
    for _ in range(n_splits):
        p1: dict[int, np.ndarray] = {}
        p2: dict[int, np.ndarray] = {}
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            if len(idx) < 2 * (k + 2):
                continue
            perm = local.permutation(idx)
            for store, part in ((p1, perm[:len(idx) // 2]),
                                (p2, perm[len(idx) // 2:])):
                block = h[part] - h[part].mean(0)
                cov = block.T @ block / len(part)
                store[c] = np.linalg.eigh(cov)[1][:, -k:]
        shared = [c for c in p1 if c in p2]
        used = max(used, len(shared))
        for c in shared:
            within_vals.append(
                1 - np.linalg.norm(p1[c].T @ p2[c]) ** 2 / k)
            for d in shared:
                if d != c:
                    cross_vals.append(
                        1 - np.linalg.norm(p1[c].T @ p2[d]) ** 2 / k)
    cross = float(np.mean(cross_vals)) if cross_vals else 0.0
    within = float(np.mean(within_vals)) if within_vals else 0.0
    return {"cross": cross, "within": within,
            "within_sd": float(np.std(within_vals)) if within_vals else 0.0,
            "heterogeneity": cross - within, "n_classes_used": used}


def spike_alignments(h: np.ndarray, labels: np.ndarray, w: np.ndarray,
                     n_classes: int, safety: float = 1.05) -> dict:
    """Per-spike alignment with the class-mean span and W's row space.

    The inputs of a score-specific selection rule: distance scores want
    mean-aligned components, projected-logit scores want row-space-aligned
    components, reconstruction scores also need the stable ID residue.
    """
    n, dim = h.shape
    centered = h - h.mean(0)
    eigs, vecs = np.linalg.eigh(centered.T @ centered / n)
    y = dim / n
    sigma2 = estimate_bulk_sigma2(eigs, y, safety)
    edge = safety * mp_edge(sigma2, dim, n)
    idx = np.where(eigs > edge)[0][::-1]
    span = _mean_span(h, labels, n_classes)
    w_row = np.linalg.qr(w.T, mode="reduced")[0]
    return {"eig": eigs[idx],
            "align_mean_span": np.array(
                [np.linalg.norm(span.T @ vecs[:, i]) ** 2 for i in idx]),
            "align_w_rowspace": np.array(
                [np.linalg.norm(w_row.T @ vecs[:, i]) ** 2 for i in idx])}


def common_mode_fraction(h: np.ndarray) -> dict:
    """m-channel diagnostics: mean-direction energy fraction, amplitude CV."""
    mu = h.mean(0)
    g = mu / (np.linalg.norm(mu) + 1e-12)
    amp = h @ g
    return {"energy_fraction": float((amp ** 2).mean()
                                     / ((h ** 2).sum(1).mean() + 1e-12)),
            "amplitude_cv": float(amp.std() / (abs(amp.mean()) + 1e-12))}


def weight_top_gap(w: np.ndarray) -> float:
    """Top singular value of W over the mean of the remainder."""
    s = np.linalg.svd(w, compute_uv=False)
    return float(s[0] / (s[1:].mean() + 1e-12))


def weight_top_class_alignment(w: np.ndarray) -> float:
    """Energy of W's top right singular vector inside the centered row span.

    RankWeight safety dial: under a degenerate (near-ETF) spectrum the top
    direction is arbitrary and can legitimately lie inside the span of the
    centered class rows; removing it then destroys class-discriminative
    margin (worst case: one class's logits collapse). A genuine common mode
    is orthogonal to the centered rows, so this reads near 0 exactly when
    removal is safe. Apply RankWeight only when weight_top_gap is material
    AND this alignment is small.
    """
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    centered_rows = (w - w.mean(0)).T
    span = np.linalg.qr(centered_rows, mode="reduced")[0]
    rank = np.linalg.matrix_rank(w - w.mean(0))
    return float(np.linalg.norm(span[:, :rank].T @ vt[0]) ** 2)


def axis_alignment(mu: np.ndarray) -> float:
    """ASH dial: inverse participation ratio of class-mean mass (mean IPR)."""
    p = mu ** 2
    p = p / (p.sum(1, keepdims=True) + 1e-12)
    return float((p ** 2).sum(1).mean())


def token_top_ratio(x: np.ndarray) -> np.ndarray:
    """RankFeat dial: per-sample s1^2 / |X|_F^2 for token maps (n, T, D)."""
    s = np.linalg.svd(x, compute_uv=False)
    return (s[:, 0] ** 2) / (s ** 2).sum(1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_cls, dim, k_res = 10, 256, 8
    radius, tau2 = 3.0, 3.0
    basis = np.linalg.qr(rng.standard_normal((dim, n_cls + k_res)))[0]
    simplex = (np.eye(n_cls) - np.ones((n_cls, n_cls)) / n_cls)
    simplex *= np.sqrt(n_cls / (n_cls - 1)) * radius
    mu = simplex @ basis[:, :n_cls].T
    b_res = basis[:, n_cls:]

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_cls, n)
        h = mu[y] + (rng.standard_normal((n, k_res)) * np.sqrt(tau2)) \
            @ b_res.T + rng.standard_normal((n, dim))
        return h, y

    print("[a] base planted case: N=4000, y=0.064, planted omegas: "
          "8 x 3.0 (residue) + 9 x 1.0 (class means)")
    h, y_lab = draw(4000)
    cen = spike_census(h)
    print(f"    n_spikes={cen['n_spikes']} (planted 17), bulk sigma2="
          f"{cen['sigma2_bulk']:.3f} (true 1.0)")
    print(f"    top-8 inverted omegas: {np.round(cen['spike_omegas'][:8], 2)}")
    print(f"    next-9 inverted omegas: "
          f"{np.round(cen['spike_omegas'][8:17], 2)}")
    via = viability(h, y_lab, n_cls)
    print(f"    viability: omega_aligned={via['omega_mean_aligned']:.2f} "
          f"thr_global={via['thr_global']:.2f} "
          f"global={via['global_viable']}  frac_class_viable="
          f"{via['frac_classes_viable']:.2f} class={via['class_viable']}")
    stab = split_half_stability(h, 17)
    print(f"    split-half capture(k=17) = {stab[0]:.3f}+-{stab[1]:.3f}")
    ali = spike_alignments(h, y_lab, mu, n_cls)
    print(f"    residue spikes mean-span align: "
          f"{ali['align_mean_span'][:8].mean():.3f} (expect ~0); "
          f"mean spikes: {ali['align_mean_span'][8:17].mean():.3f} "
          "(expect ~1)")

    print("[b] hard aspect ratio: N=320, y=0.8 (raw median would be biased)")
    h2, _ = draw(320)
    cen2 = spike_census(h2)
    raw_median = float(np.median(np.linalg.eigvalsh(
        (h2 - h2.mean(0)).T @ (h2 - h2.mean(0)) / len(h2))))
    print(f"    bulk sigma2={cen2['sigma2_bulk']:.3f} (true 1.0; "
          f"uncorrected median={raw_median:.3f})")

    print("[c] per-class regime: N=2000 (N_c~200, y_c~1.28, thr_c~1.13)")
    h3, y3 = draw(2000)
    via3 = viability(h3, y3, n_cls)
    print(f"    omega_per_class median="
          f"{np.median(via3['omega_per_class']):.2f} vs thr "
          f"{np.median(via3['thr_per_class']):.2f} -> class_viable="
          f"{via3['class_viable']} (residue omega=3 recoverable)")

    print("[d] weight diagnostics: ETF vs planted common mode")
    g_dir = np.linalg.qr(rng.standard_normal((dim, dim)))[0][:, -1]
    w_cm = mu + 1.5 * g_dir[None, :]
    print(f"    ETF: gap={weight_top_gap(mu):.2f} "
          f"align={weight_top_class_alignment(mu):.2f} (degenerate: do not "
          "trust removal)")
    print(f"    common mode: gap={weight_top_gap(w_cm):.2f} "
          f"align={weight_top_class_alignment(w_cm):.2f} (identifiable)")

    print("[e] mp_median sanity: y=0.1 -> "
          f"{mp_median(0.1):.3f}, y=0.5 -> {mp_median(0.5):.3f}, "
          f"y=0.8 -> {mp_median(0.8):.3f}, y=1.5 -> {mp_median(1.5):.3f}")
