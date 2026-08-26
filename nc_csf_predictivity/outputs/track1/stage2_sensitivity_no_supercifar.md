# Exploratory sensitivity probe: no_supercifar

EXPLORATORY, post-hoc roster surgery (2026-08-25); NOT a frozen analysis. Pool without the supercifar100 source (3 sources). Question: is any conclusion supercifar-driven?
Reproduce: `python stage2_sensitivity_probes.py` (from code/; seeds in the module docstring; reuses the frozen theory cache and registered fold/CI/gate machinery).

```
pool: 190 checkpoints, 1520 cells
var_collapse spread: [0.001, 0.065]
[ckpt5] material 404, frac+ 0.62; theory 0.035, severity 0.686, geometry 0.916
  G-S sign: {'point': 0.23, 'ci95': [0.147, 0.315]}
  G-S balanced: {'point': 0.299, 'ci95': [0.222, 0.367]}
[loso] material 404, frac+ 0.62; theory 0.035, severity 0.550, geometry 0.552
  G-S sign: {'point': 0.002, 'ci95': [-0.043, 0.05]}
  G-S balanced: {'point': -0.038, 'ci95': [-0.066, -0.011]}
  held-out cifar10: n 170, frac+ 0.11, geo 0.112, sev 0.259
  held-out cifar100: n 90, frac+ 0.98, geo 0.967, sev 0.989
  held-out tinyimagenet: n 144, frac+ 1.00, geo 0.812, sev 0.618
pooled first up-crossing -1.113, tie region [-1.225, 1.557]
in-sample tertile crossings: {'strong': -1.196, 'middle': -1.062, 'weak': inf}, ordering retained: True
Gate-3-style held-out ordering: ['RETAINED', 'RETAINED', 'RETAINED'] -> PASS
```
