"""Outcome-free spot check of plan v2 section 9.2: delete-family jackknife SE
+ Student t(N-1) interval on the rank-slope average A (section 7.1), ranks
recomputed inside each deletion (section 9.3). Null: gaps independent of
geometry. Reports the two-sided 98% false-claim rate (target <= 0.02)."""
import numpy as np
from scipy import stats

S, O = 4, 4


def midrank(v):
    return stats.rankdata(v, method="average")


def A_stat(g, D):
    # g: (S, n) geometry; D: (S, n, O) gaps. Returns equal-weight A.
    n = g.shape[1]
    T = np.empty((S, O))
    for s in range(S):
        x = (midrank(g[s]) - 0.5) / n - 0.5
        xc = x - x.mean()
        den = (xc ** 2).sum()
        for o in range(O):
            d = D[s, :, o]
            b = (xc * (d - d.mean())).sum() / den
            T[s, o] = -0.8 * b
    return T.mean()


def jackknife_interval(g, D, alpha=0.02):
    n = g.shape[1]
    A = A_stat(g, D)
    Aj = np.array([A_stat(np.delete(g, j, 1), np.delete(D, j, 1)) for j in range(n)])
    se = np.sqrt((n - 1) / n * ((Aj - Aj.mean()) ** 2).sum())
    if se == 0:
        return A, np.nan, np.nan
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    return A, A - tcrit * se, A + tcrit * se


def gen(rng, n, kind, effect=0.0):
    g = rng.standard_normal((S, n))
    if kind == "gauss":
        u = 0.01 * rng.standard_normal(n)              # shared family effect
        v = 0.01 * rng.standard_normal((S, n))         # source-family effect
        e = 0.01 * rng.standard_normal((S, n, O))      # cell noise
    elif kind == "t3":
        sc = 1 / np.sqrt(3)                            # standardize t3
        u = 0.01 * sc * rng.standard_t(3, n)
        v = 0.01 * sc * rng.standard_t(3, (S, n))
        e = 0.01 * sc * rng.standard_t(3, (S, n, O))
    elif kind == "hetero":                             # geometry-dependent variance
        u = 0.01 * rng.standard_normal(n)
        v = 0.01 * rng.standard_normal((S, n))
        sd = 0.01 * (0.5 + 1.5 * (g > 0))              # 3x sd on one side
        e = sd[:, :, None] * rng.standard_normal((S, n, O))
    D = u[None, :, None] + v[:, :, None] + e
    if effect:
        # linear-in-rank effect with A = effect (uses population rank x)
        x = (np.argsort(np.argsort(g, axis=1), axis=1) + 0.5) / n - 0.5
        D = D - (effect / 0.8) * x[:, :, None]
    return g, D


def run(n, kind, reps, seed, effect=0.0):
    rng = np.random.default_rng(seed)
    hits = 0; degenerate = 0; widths = []
    for _ in range(reps):
        g, D = gen(rng, n, kind, effect)
        A, lo, hi = jackknife_interval(g, D)
        if np.isnan(lo):
            degenerate += 1; continue
        widths.append(hi - lo)
        if effect == 0 and (lo > 0 or hi < 0):
            hits += 1
        if effect and lo > 0:
            hits += 1
    k = reps - degenerate
    rate = hits / k
    ci = stats.binomtest(hits, k).proportion_ci(0.95)
    return rate, ci.low, ci.high, np.median(widths) / 2, degenerate


print(f"{'N':>3} {'scenario':>8} {'effect':>6} {'rate':>7} {'95% MC CI':>17} {'med half-width':>15} {'degen':>5}")
for n in (5, 20, 40):
    reps = 20000 if n == 5 else 6000
    for kind in ("gauss", "t3", "hetero"):
        r, lo, hi, hw, dg = run(n, kind, reps, seed=20260908 + n)
        print(f"{n:>3} {kind:>8} {0.0:>6} {r:>7.4f} [{lo:.4f}, {hi:.4f}] {hw:>15.4f} {dg:>5}")
    r, lo, hi, hw, dg = run(n, "gauss", 3000, seed=99 + n, effect=0.01)
    print(f"{n:>3} {'gauss':>8} {0.01:>6} {r:>7.4f} [{lo:.4f}, {hi:.4f}] {hw:>15.4f} {dg:>5}   <- one-sided power at A=0.01")
