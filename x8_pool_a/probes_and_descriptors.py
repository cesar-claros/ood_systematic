"""Pool A probes and weak-collapse descriptors (CPU, numpy only).

Trains multinomial logistic probes on cached frozen features and measures the
X8 descriptor vector per probe model. Self-test on synthetic features at bottom.
"""
import sys, pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1].parent
                    / "documentation" / "x6_spectral_scripts"))
from spectral_diagnostics import spike_census, viability, common_mode_fraction  # noqa: E402


def train_probe(H, y, C, seed=0, steps=300, lr=0.5, weight_decay=1e-4):
    """Multinomial logistic regression with momentum GD on standardized features."""
    rng = np.random.default_rng(seed)
    mu, sd = H.mean(0), H.std(0) + 1e-8
    Z = (H - mu) / sd
    W = rng.standard_normal((C, Z.shape[1])) * 0.01
    b = np.zeros(C)
    vel_w, vel_b = 0.0, 0.0
    onehot = np.eye(C)[y]
    for _ in range(steps):
        logits = Z @ W.T + b
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        g = (p - onehot) / len(y)
        gw = g.T @ Z + weight_decay * W
        vel_w = 0.9 * vel_w - lr * gw
        vel_b = 0.9 * vel_b - lr * g.sum(0)
        W += vel_w
        b += vel_b
    acc = (np.argmax(Z @ W.T + b, 1) == y).mean()
    return {"W": W, "b": b, "mu": mu, "sd": sd, "acc": acc}


def descriptors(H, y, C):
    """X8 descriptor vector: residue energy, spike census, cone occupancy,
    class-dependent-residue proxy, plus P6.1 viability flags."""
    mu_c = np.stack([H[y == c].mean(0) for c in range(C)])
    cen = mu_c - mu_c.mean(0)
    span = np.linalg.qr(cen.T, mode="reduced")[0][:, :C - 1]
    Hc = H - mu_c[y]
    within_census = spike_census(Hc)
    eigs, s2 = within_census["eigs"], within_census["sigma2_bulk"]
    spikes = eigs[eigs > within_census["edge"]]
    rho_res = float((spikes - s2).sum() / (eigs.sum() + 1e-12))
    cos_own = np.einsum("nd,nd->n", H - mu_c.mean(0), cen[y]) / (
        np.linalg.norm(H - mu_c.mean(0), axis=1) * np.linalg.norm(cen[y], axis=1) + 1e-12)
    off = cen - cen @ span @ span.T
    class_dep = float((off ** 2).sum() / ((cen ** 2).sum() + 1e-12))
    cen_sus = spike_census(H)
    via = viability(H, y, C)
    return {"rho_res": rho_res, "n_spikes": cen_sus["n_spikes"],
            "n_residue_spikes": int(len(spikes)),
            "median_cos_own": float(np.median(cos_own)),
            "class_dep_residue": class_dep,
            "global_viable": bool(via["global_viable"]),
            "class_viable": bool(via["class_viable"]),
            "common_mode": common_mode_fraction(H)}


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    C, D, K, N = 10, 384, 25, 6000
    Q = np.linalg.qr(rng.standard_normal((D, C + K)))[0]
    MU = ((np.eye(C) - np.ones((C, C)) / C) * 2.0) @ Q[:, :C].T
    B = Q[:, C:]
    y = rng.integers(0, C, N)
    for tau, tag in [(0.0, "strong-collapse"), (1.2, "foundation-like")]:
        H = MU[y] + (rng.standard_normal((N, K)) * tau) @ B.T \
            + rng.standard_normal((N, D)) * 0.4
        probe = train_probe(H, y, C)
        d = descriptors(H, y, C)
        print(f"[{tag}] probe acc={probe['acc']:.3f}  rho_res={d['rho_res']:.3f}  "
              f"residue_spikes={d['n_residue_spikes']}  med_cos={d['median_cos_own']:.3f}  "
              f"viable(g/c)={d['global_viable']}/{d['class_viable']}")
