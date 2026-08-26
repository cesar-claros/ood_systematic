# Exploratory sensitivity probe: confidnet_devries

EXPLORATORY, post-hoc roster surgery (2026-08-25); NOT a frozen analysis. Pool restricted to confidnet+devries (no DG paradigm, single reward). Question: how much rides on DG's reward-induced geometry variation?
Reproduce: `python stage2_sensitivity_probes.py` (from code/; seeds in the module docstring; reuses the frozen theory cache and registered fold/CI/gate machinery).

```
pool: 80 checkpoints, 640 cells
var_collapse spread: [0.002, 0.026]
[ckpt5] material 240, frac+ 0.64; theory 0.058, severity 0.725, geometry 0.979
  G-S sign: {'point': 0.254, 'ci95': [0.131, 0.376]}
  G-S balanced: {'point': 0.367, 'ci95': [0.317, 0.401]}
[loso] material 240, frac+ 0.64; theory 0.058, severity 0.525, geometry 0.583
  G-S sign: {'point': 0.058, 'ci95': [-0.054, 0.169]}
  G-S balanced: {'point': 0.053, 'ci95': [-0.045, 0.16]}
  held-out cifar10: n 65, frac+ 0.00, geo 0.031, sev 0.154
  held-out cifar100: n 49, frac+ 1.00, geo 0.980, sev 1.000
  held-out supercifar100: n 61, frac+ 0.66, geo 0.525, sev 0.820
  held-out tinyimagenet: n 65, frac+ 1.00, geo 0.892, sev 0.262
pooled first up-crossing -1.081, tie region [-1.255, 1.557]
in-sample tertile crossings: {'strong': -1.184, 'middle': -1.2, 'weak': 0.18}, ordering retained: True
Gate-3-style held-out ordering: ['REVERSED', 'RETAINED', 'REVERSED', 'RETAINED'] -> FAIL
```
