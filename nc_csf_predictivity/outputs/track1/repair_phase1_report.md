# Repair-campaign Phase-1 extraction diagnostics

Descriptive; no detector outcome read. Frozen rules in pilot0/repair_stats.py; manifest sha256 e15752da...

```
{
 "n_checkpoints": 33,
 "n_pool": 28,
 "n_breeds": 5,
 "n_ood_set_records": {
  "pool": 224,
  "breeds": 5
 },
 "runtime_total_min": 54.9,
 "id_prototype_switch_rate": {
  "median": 0.0405,
  "min": 0.0075,
  "max": 0.4046
 },
 "id_dir_over_avg_max": {
  "median": 45.541,
  "min": 6.4161,
  "max": 148.8362
 },
 "pool_ood": {
  "n_raw": {
   "median": 39.5,
   "min": 8.0,
   "max": 200.0
  },
  "n_kept": {
   "median": 19.0,
   "min": 4.0,
   "max": 115.0
  },
  "other_weight": {
   "median": 0.0074,
   "min": 0.0,
   "max": 0.2025
  },
  "top_share": {
   "median": 0.2869,
   "min": 0.0468,
   "max": 0.8559
  },
  "class_switch": {
   "median": 0.1341,
   "min": 0.0093,
   "max": 0.6045
  },
  "align_global": {
   "median": 0.5687,
   "min": 0.2121,
   "max": 0.8824
  },
  "align_comp_wmean": {
   "median": 0.7925,
   "min": 0.4306,
   "max": 0.9775
  },
  "mixing_bias": {
   "median": 0.2062,
   "min": 0.0165,
   "max": 0.5481
  },
  "ood_dir_over_avg_max": {
   "median": 102.29,
   "min": 19.88,
   "max": 287.25
  },
  "rho_shared": {
   "median": 0.9056,
   "min": 0.4607,
   "max": 1.551
  }
 },
 "breeds_ood": {
  "n_raw": {
   "median": 13.0,
   "min": 13.0,
   "max": 13.0
  },
  "n_kept": {
   "median": 13.0,
   "min": 13.0,
   "max": 13.0
  },
  "other_weight": {
   "median": 0.0,
   "min": 0.0,
   "max": 0.0
  },
  "top_share": {
   "median": 0.0889,
   "min": 0.0873,
   "max": 0.0915
  },
  "class_switch": {
   "median": 0.0556,
   "min": 0.0331,
   "max": 0.0804
  },
  "align_global": {
   "median": 0.4574,
   "min": 0.2543,
   "max": 0.5147
  },
  "align_comp_wmean": {
   "median": 0.9729,
   "min": 0.9419,
   "max": 0.9795
  },
  "mixing_bias": {
   "median": 0.5151,
   "min": 0.4594,
   "max": 0.6875
  },
  "ood_dir_over_avg_max": {
   "median": 216.43,
   "min": 189.59,
   "max": 224.56
  },
  "rho_shared": {
   "median": 0.9932,
   "min": 0.9305,
   "max": 1.0123
  }
 }
}
```
