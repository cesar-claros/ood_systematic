"""Verified projector-based descriptors (2026-09-15), separately named from the legacy X8 fields in probes_and_descriptors.py.

residue_energy_projector(H, y): 1 - Tr(P Sigma_W P) / Tr(Sigma_W), with P the orthogonal projector onto the span of the centered
class means (the written X8 section-5 definition). The legacy `rho_res` is an excess-spike-mass statistic and is NOT this quantity;
the legacy `class_dep_residue` is zero by construction (centered means lie in their own span). Self-test at the bottom reproduces
the counterexample of the 2026-09-15 pilot-readiness review: identical class means and identical within-class eigenvalues, variance
concentrated inside versus outside the mean span.
"""
import sys, pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))


def mean_span_projector(H, y, C):
    mu_c = np.stack([H[y == c].mean(0) for c in range(C)])
    cen = mu_c - mu_c.mean(0)
    Q = np.linalg.qr(cen.T, mode="reduced")[0][:, :C - 1]
    return Q @ Q.T, mu_c


def residue_energy_projector(H, y, C):
    """Fraction of within-class variance outside the span of the centered class means."""
    P, mu_c = mean_span_projector(H, y, C)
    Hc = H - mu_c[y]
    SW = Hc.T @ Hc / len(Hc)
    return float(1.0 - np.trace(P @ SW @ P) / (np.trace(SW) + 1e-12))


if __name__ == "__main__":
    from probes_and_descriptors import descriptors
    rng = np.random.default_rng(0)
    C, D, N = 10, 128, 4000
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    span_basis, comp_basis = Q[:, :C - 1], Q[:, C - 1:]
    MU = ((np.eye(C) - np.ones((C, C)) / C) * 3.0) @ np.vstack([span_basis.T, np.zeros((1, D))])[:C]  # class means inside the span
    y = rng.integers(0, C, N)
    eig_big, eig_small = 4.0, 0.05
    for where in ["inside span", "outside span"]:
        noise = rng.standard_normal((N, D)) * np.sqrt(eig_small)
        big = rng.standard_normal((N, C - 1)) * np.sqrt(eig_big - eig_small)
        H = MU[y] + noise + (big @ span_basis.T if where == "inside span" else big @ comp_basis[:, :C - 1].T)
        leg = descriptors(H, y, C)
        print(f"variance concentrated {where}: legacy rho_res {leg['rho_res']:.6f} | projector residue energy {residue_energy_projector(H, y, C):.6f} | legacy class_dep_residue {leg['class_dep_residue']:.2e}")
