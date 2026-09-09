"""CE plan v3, section 7.1: operating characteristics of the delete-family
jackknife-t candidate at the MEASURED VGG nuisance scales (review F3).

Model (fast metric-level simulator): N seed families x 4 sources x 4
shifts. gap_{s,j,o} = -(A/0.8) * x_{s,j} + u_{s,j} + e_{s,j,o}, with
x the section-7.1 midrank transform of an independent reliable geometry
draw per source, u ~ N(0, 0.0015^2) the within-source family effect
(cross-source shared-run covariance measured as zero), e ~ N(0, 0.0032^2)
the cell residual. Statistic A^G = equal-weight mean of T_{s,o} = -0.8 b;
delete-one-family jackknife with ranks recomputed; t_{N-1} reference,
multiplier 1; two-sided 98% interval. Reports false-claim rate at the
global null, one-sided power at A in {0.002, 0.005, 0.01}, and median
half-width. 3000 replications per cell, seed 20260908.
Usage (from code/): .venv/bin/python tests/ce_plan_v3_measured_scale_power_20260908.py
"""
import numpy as np
from scipy.stats import t as student

SD_FAM, SD_CELL, REPS, ALPHA = 0.0015, 0.0032, 3000, 0.02


def stat(gap, g):
    # gap: (N, 4, 4) families x sources x shifts; g: (N, 4) geometry
    N = gap.shape[0]
    r = np.argsort(np.argsort(g, axis=0), axis=0) + 1.0
    x = (r - 0.5) / N - 0.5
    xc = x - x.mean(0)
    num = (xc[:, :, None] * (gap - gap.mean(0))).sum(0)
    b = num / (xc ** 2).sum(0)[:, None]
    return float((-0.8 * b).mean())


def one(rng, N, A):
    g = rng.standard_normal((N, 4))
    r = np.argsort(np.argsort(g, axis=0), axis=0) + 1.0
    x = (r - 0.5) / N - 0.5
    gap = (-(A / 0.8) * x[:, :, None] + rng.normal(0, SD_FAM, (N, 4))[:, :, None]
           + rng.normal(0, SD_CELL, (N, 4, 4)))
    full = stat(gap, g)
    loo = np.array([stat(np.delete(gap, j, 0), np.delete(g, j, 0)) for j in range(N)])
    se = np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum())
    hw = student.ppf(1 - ALPHA / 2, N - 1) * se
    return full, hw


def main():
    rng = np.random.default_rng(20260908)
    print(f"{'N':>3} {'false-claim(98%)':>17} {'pow@0.002':>10} {'pow@0.005':>10} {'pow@0.01':>9} {'med half-width':>15}")
    for N in (5, 20, 30, 40):
        fc = 0; hws = []
        for _ in range(REPS):
            f, hw = one(rng, N, 0.0); hws.append(hw)
            fc += (abs(f) > hw)
        pw = []
        for A in (0.002, 0.005, 0.01):
            k = 0
            for _ in range(REPS):
                f, hw = one(rng, N, A)
                k += (f - hw > 0)
            pw.append(k / REPS)
        print(f"{N:>3} {fc / REPS:>17.3f} {pw[0]:>10.2f} {pw[1]:>10.2f} {pw[2]:>9.2f} {np.median(hws):>15.4f}")


if __name__ == "__main__":
    main()
