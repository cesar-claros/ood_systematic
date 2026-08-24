# MC phase-map audit (P0; frozen grid in mc_phase_audit.py)

Configs audited: 348; adaptive MC SE target 0.0025; tolerance 0.01; overall predicted-winner accuracy 0.718.

| score | median err | 95th err | max err | within tol | boundary sign acc | known failure regime |
|---|---|---|---|---|---|---|
| MSR | 0.0123 | 0.1686 | 0.3161 | 0.47 | 0.930 | scattered (53% of configs) |
| MLS | 0.0005 | 0.0053 | 0.2200 | 0.97 | 0.975 | tied/diffuse alignment, theta_w >= 21 deg, spiked covariance, large gamma*a (3% of configs) |
| Energy | 0.0006 | 0.0056 | 0.3033 | 0.97 | 0.978 | tied/diffuse alignment, theta_w >= 21 deg, spiked covariance (3% of configs) |
| CTM_head | 0.0001 | 0.0034 | 0.0218 | 0.99 | 0.947 | tied/diffuse alignment, theta_w >= 21 deg, spiked covariance, large gamma*a (1% of configs) |
| CTM_mean | 0.0001 | 0.0029 | 0.0283 | 0.99 | 0.983 | tied/diffuse alignment, theta_w >= 21 deg, spiked covariance, large gamma*a (1% of configs) |
| Maha | 0.0000 | 0.0010 | 0.0553 | 0.99 | 0.995 | scattered (1% of configs) |
| fDBD | 0.0001 | 0.0389 | 0.2029 | 0.90 | 0.933 | theta_w >= 21 deg, clustered classes (10% of configs) |

## Crossing displacement along gamma*a sweeps (analytic vs MC, gamma*a units)

| sweep/pair | analytic | MC | displacement |
|---|---|---|---|
| C100_s10_CTM_head|CTM_mean | 0.88 | 0.876 | -0.004 |
| C100_s10_CTM_head|Maha | 0.912 | 0.912 | 0.0 |
| C100_s10_CTM_mean|Maha | 0.912 | 0.911 | -0.0 |
| C100_s10_CTM_mean|fDBD | 0.88 | 0.872 | -0.008 |
| C100_s10_Energy|CTM_head | 0.259 | 0.504 | 0.245 |
| C100_s10_Energy|CTM_mean | 0.259 | 0.502 | 0.242 |
| C100_s10_Energy|Maha | 0.9 | 0.899 | -0.001 |
| C100_s10_Energy|fDBD | 0.259 | 0.505 | 0.246 |
| C100_s10_MLS|CTM_head | 0.506 | 0.504 | -0.002 |
| C100_s10_MLS|CTM_mean | 0.503 | 0.502 | -0.001 |
| C100_s10_MLS|Maha | 0.9 | 0.899 | -0.001 |
| C100_s10_MLS|fDBD | 0.506 | 0.505 | -0.001 |
| C100_s10_MSR|CTM_head | 1.106 | 0.503 | -0.603 |
| C100_s10_MSR|CTM_mean | 1.111 | 0.501 | -0.61 |
| C100_s10_MSR|Energy | 0.979 | 0.509 | -0.469 |
| C100_s10_MSR|MLS | 0.979 | 0.512 | -0.467 |
| C100_s10_MSR|Maha | 0.874 | 0.9 | 0.026 |
| C100_s10_MSR|fDBD | 1.106 | 0.504 | -0.602 |
| C100_s10_Maha|fDBD | 0.912 | 0.912 | 0.0 |
| C100_s24_CTM_head|CTM_mean | 0.5 | 0.828 | 0.328 |
| C100_s24_CTM_head|Maha | 0.504 | 0.502 | -0.002 |
| C100_s24_CTM_mean|Maha | 0.502 | 0.501 | -0.001 |
| C100_s24_CTM_mean|fDBD | 0.5 | 0.824 | 0.324 |
| C100_s24_Energy|Maha | 0.5 | 0.5 | -0.0 |
| C100_s24_MLS|Maha | 0.5 | 0.5 | -0.0 |
| C100_s24_MSR|Maha | 0.25 | 0.5 | 0.25 |
| C100_s24_Maha|fDBD | 0.504 | 0.502 | -0.002 |
| C100_s65_CTM_head|CTM_mean | 1.1 | 1.1 | 0.0 |
| C100_s65_CTM_mean|fDBD | 1.1 | 1.1 | 0.0 |
| C10_s10_CTM_head|CTM_mean | 0.88 | 0.375 | -0.505 |
| C10_s10_CTM_head|Maha | 0.918 | 0.918 | -0.001 |
| C10_s10_CTM_mean|Maha | 0.916 | 0.915 | -0.001 |
| C10_s10_CTM_mean|fDBD | 0.88 | 0.865 | -0.015 |
| C10_s10_Energy|CTM_head | 0.504 | 0.502 | -0.003 |
| C10_s10_Energy|CTM_mean | 0.504 | 0.502 | -0.002 |
| C10_s10_Energy|Maha | 0.858 | 0.857 | -0.001 |
| C10_s10_Energy|fDBD | 0.504 | 0.502 | -0.002 |
| C10_s10_MLS|CTM_head | 0.504 | 0.502 | -0.002 |
| C10_s10_MLS|CTM_mean | 0.504 | 0.502 | -0.002 |
| C10_s10_MLS|Energy | 0.928 | 0.507 | -0.421 |
| C10_s10_MLS|Maha | 0.858 | 0.857 | -0.001 |
| C10_s10_MLS|fDBD | 0.504 | 0.502 | -0.002 |
| C10_s10_MSR|CTM_head | 1.387 | 0.501 | -0.886 |
| C10_s10_MSR|CTM_mean | 1.379 | 0.501 | -0.878 |
| C10_s10_MSR|Energy | 0.993 | 0.841 | -0.152 |
| C10_s10_MSR|MLS | 0.993 | 0.842 | -0.152 |
| C10_s10_MSR|fDBD | 1.387 | 0.501 | -0.886 |
| C10_s10_Maha|fDBD | 0.918 | 0.919 | 0.001 |
| C10_s24_CTM_head|CTM_mean | 0.912 | 0.912 | -0.0 |
| C10_s24_CTM_head|Maha | 0.25 | 0.82 | 0.57 |
| C10_s24_CTM_mean|Maha | 0.25 | 0.829 | 0.579 |
| C10_s24_CTM_mean|fDBD | 0.912 | 0.905 | -0.008 |
| C10_s24_Energy|Maha | 0.5 | 0.5 | -0.0 |
| C10_s24_MLS|Maha | 0.5 | 0.5 | -0.0 |
| C10_s24_Maha|fDBD | 0.25 | 0.826 | 0.576 |
| C10_s65_MLS|Energy | 0.8 | 0.927 | 0.127 |
| C10_s65_MSR|Energy | 0.8 | 0.919 | 0.119 |
| C10_s65_MSR|MLS | 0.8 | 0.919 | 0.119 |

## Decision per score (section 3.6)

- **MSR**: boundary from simulation required (sign accuracy below 0.95 in audited regimes); EXCLUDE tied-logit regimes from the +-0.01 claim (state in abstract and formula table)
- **MLS**: calibrated formula (within tolerance)
- **Energy**: calibrated formula (within tolerance)
- **CTM_head**: calibrated formula (within tolerance)
- **CTM_mean**: calibrated formula (within tolerance)
- **Maha**: calibrated formula (within tolerance)
- **fDBD**: boundary from simulation required (sign accuracy below 0.95 in audited regimes); proxy = degeneracy with head-CTM, valid near collapse only (two-axis divergence is the paper's own claim)
