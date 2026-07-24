"""CPU verification that each X3 penalty moves its target NC coordinate.

Unconstrained-features toy: trainable class means MU (C x D) and head W (C x D)
with frozen within-class noise; loss = CE + lam * penalty. CE gradients are
analytic; penalty gradients use central finite differences (no autograd needed).
Passes iff each dial reduces its target metric substantially and selectively.
"""
import numpy as np

rng = np.random.default_rng(0)
C, D, N, SIG, STEPS, LR = 6, 24, 600, 0.35, 400, 0.15


def metrics(MU, W, H, y):
    m = MU - MU.mean(0)
    sd = np.sum((W.T / np.linalg.norm(W) - m.T / np.linalg.norm(m)) ** 2)
    mu_smp = np.stack([H[y == c].mean(0) for c in range(C)])
    cen = mu_smp - mu_smp.mean(0)
    sw = np.mean([(H[y == c] - mu_smp[c]).T @ (H[y == c] - mu_smp[c]) / (y == c).sum()
                  for c in range(C)], axis=0)
    sb = cen.T @ cen / C
    nc1 = np.trace(sw @ np.linalg.pinv(sb)) / C
    nrm = np.linalg.norm(cen, axis=1)
    eqn = nrm.std() / nrm.mean()
    return {"self_duality": sd, "nc1": nc1, "equinorm": eqn}


def penalty(name, MU, W):
    m = MU - MU.mean(0)
    if name == "etfreg":
        return np.sum((W.T / np.linalg.norm(W) - m.T / np.linalg.norm(m)) ** 2)
    if name == "varreg":
        return (SIG ** 2 * D) / max(np.mean(np.sum(m ** 2, 1)), 1e-8)
    if name == "eqnreg":
        nrm = np.linalg.norm(m, axis=1)
        return nrm.var() / max(nrm.mean() ** 2, 1e-8)
    return 0.0


def fd_grad(name, MU, W, eps=1e-5):
    gM, gW = np.zeros_like(MU), np.zeros_like(W)
    for arr, g in ((MU, gM), (W, gW)):
        flat, gf = arr.ravel(), g.ravel()
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps; hi = penalty(name, MU, W)
            flat[i] = old - eps; lo = penalty(name, MU, W)
            flat[i] = old
            gf[i] = (hi - lo) / (2 * eps)
    return gM, gW


def train(name, lam):
    r = np.random.default_rng(7)
    MU = r.standard_normal((C, D)) * (0.5 if name != "eqnreg" else 1.0)
    if name == "eqnreg":
        MU *= (1 + 0.6 * r.standard_normal(C))[:, None]
    W = r.standard_normal((C, D)) * 0.3
    y = r.integers(0, C, N)
    log_sig = np.log(np.full(C, SIG))
    XI = r.standard_normal((N, D))
    for _ in range(STEPS):
        sig = np.exp(log_sig)
        H = MU[y] + sig[y][:, None] * XI
        logits = H @ W.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        dlog = (p - np.eye(C)[y]) / N
        gW_ce = dlog.T @ H
        gH = dlog @ W
        gMU_ce = np.zeros_like(MU)
        np.add.at(gMU_ce, y, gH)
        g_ls = np.zeros(C)
        np.add.at(g_ls, y, np.sum(gH * XI, 1) * sig[y])
        if lam > 0:
            if name == "varreg":
                m = MU - MU.mean(0)
                denom = max(np.mean(np.sum(m ** 2, 1)), 1e-8)
                g_ls += lam * (2 * D * sig ** 2 / C) / denom
                gM_p = -lam * D * np.mean(sig ** 2) * (2 * (m - m.mean(0)) / C) / denom ** 2
                gW_p = 0.0
            else:
                gM_p, gW_p = fd_grad(name, MU, W)
        else:
            gM_p = gW_p = 0.0
        MU -= LR * (gMU_ce + (lam * gM_p if (lam > 0 and name != "varreg") else gM_p if lam > 0 else 0.0))
        W -= LR * (gW_ce + lam * gW_p)
        log_sig -= LR * g_ls
    H = MU[y] + np.exp(log_sig)[y][:, None] * XI
    acc = (np.argmax(H @ W.T, 1) == y).mean()
    return metrics(MU, W, H, y), acc


TARGET = {"etfreg": "self_duality", "varreg": "nc1", "eqnreg": "equinorm"}
if __name__ == "__main__":
    for name in ("etfreg", "varreg", "eqnreg"):
        base, acc0 = train(name, 0.0)
        dial, acc1 = train(name, {"etfreg": 3.0, "varreg": 3.0, "eqnreg": 3.0}[name])
        tgt = TARGET[name]
        rel = {k: (dial[k] - base[k]) / max(abs(base[k]), 1e-9) for k in base}
        ok = rel[tgt] < -0.5 and all(abs(rel[k]) < 0.5 * abs(rel[tgt]) or k == tgt for k in rel)
        print(f"{name}: target {tgt} {base[tgt]:.4f} -> {dial[tgt]:.4f} "
              f"({100*rel[tgt]:+.0f}%)  off-target "
              + " ".join(f"{k}:{100*rel[k]:+.0f}%" for k in rel if k != tgt)
              + f"  acc {acc0:.2f}->{acc1:.2f}  [{'PASS' if ok else 'CHECK'}]")
