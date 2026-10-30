# NC1 sensitivity of the headline predictor and the Mantel tests

**Source:** `nc_csf_predictivity/evaluation/nc1_sensitivity.py` (decision rule in the docstring and the camera-ready checklist, fixed before the run). Gates GS, GM, GC, GV passed.

**Verdict from C-raw and C-log: PARTLY ROBUST.**

**K-log, judged on its own: P1 True, P2 True, M1 True.**

## Criteria

```
variant  pred_near  pred_mid  pred_far  ctm_near  ctm_mid  ctm_far  reject_p  reject_p_worse  reject_p_mean  reject_p_mean_worse  mantel_r_confidnet  mantel_r_devries  mantel_r_dg  mantel_r_vit  mantel_p_max   P1    P2   M1
  L-raw      0.931     0.881     0.505     2.084    3.559    2.572        65               0             65                    0               0.644             0.679        0.309         0.590         0.003 True  True True
  L-log      0.875     0.881     0.716     2.084    3.559    2.572        65               0             65                    0               0.648             0.652        0.280         0.583         0.005 True  True True
  C-raw      1.235     1.025     0.823     2.084    3.559    2.572        63               1             52                    0               0.527             0.756        0.298         0.624         0.004 True False True
  C-log      0.918     0.933     0.752     2.084    3.559    2.572        65               0             65                    0               0.530             0.759        0.284         0.649         0.005 True  True True
  K-log      0.984     1.000     0.757     2.084    3.559    2.572        65               0             58                    0               0.645             0.675        0.280         0.582         0.005 True  True True
```

## Joint side: predictor vs size-matched random set

```
variant regime  pred_mean  random_mean  set_size  p_mean_holm
  L-raw    far      0.505        1.429     8.380        0.000
  L-raw    mid      0.881        1.379     8.380        0.000
  L-raw   near      0.931        1.460     8.095        0.008
  L-log    far      0.716        1.418     8.330        0.000
  L-log    mid      0.881        1.335     8.330        0.001
  L-log   near      0.875        1.609     7.818        0.028
  C-raw    far      0.823        1.370     8.800        0.020
  C-raw    mid      1.025        1.323     8.800        0.028
  C-raw   near      1.235        1.745     7.784        0.017
  C-log    far      0.752        1.628     7.690        0.000
  C-log    mid      0.933        1.598     7.690        0.000
  C-log   near      0.918        1.870     7.007        0.002
  K-log    far      0.757        1.590     7.760        0.000
  K-log    mid      1.000        1.525     7.760        0.000
  K-log   near      0.984        1.870     7.297        0.002
```

## Mean regret vs Always-CTM per side

```
variant         L-raw  L-log   C-raw  C-log  K-log  always_ctm
side    regime                                                
all     far     0.505  0.716   0.823  0.752  0.757       2.572
        mid     0.881  0.881   1.025  0.933  1.000       3.559
        near    0.931  0.875   1.235  0.918  0.984       2.084
feature far     0.866  1.098   3.877  1.213  2.171       2.449
        mid     1.100  0.976   4.057  1.157  2.413       2.983
        near    1.760  1.740  14.684  1.695  6.313       1.582
```

## Mantel r: no control, ID performance, source identity

```
variant                                 L-raw  L-log  C-raw  C-log  K-log
backbone study     control                                               
Conv     confidnet (none, replication)  0.644  0.648  0.527  0.530  0.645
                   dataset              0.326  0.334  0.105  0.111  0.329
                   iid_msr              0.581  0.582  0.396  0.411  0.580
         devries   (none, replication)  0.679  0.652  0.756  0.759  0.675
                   dataset              0.474  0.427  0.606  0.612  0.466
                   iid_msr              0.639  0.636  0.718  0.712  0.656
         dg        (none, replication)  0.309  0.280  0.298  0.284  0.280
                   dataset             -0.175 -0.225 -0.195 -0.219 -0.226
                   iid_msr              0.216  0.188  0.160  0.130  0.186
ViT      vit       (none, replication)  0.590  0.583  0.624  0.649  0.582
                   dataset              0.189  0.179  0.254  0.304  0.178
                   iid_msr              0.429  0.417  0.464  0.490  0.414
```

## Mantel p: same rows

```
variant                                  L-raw   L-log   C-raw   C-log   K-log
backbone study     control                                                    
Conv     confidnet (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
                   dataset              0.0004  0.0004  0.0898  0.0774  0.0004
                   iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
         devries   (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
                   dataset              0.0001  0.0001  0.0001  0.0001  0.0001
                   iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
         dg        (none, replication)  0.0034  0.0053  0.0045  0.0053  0.0053
                   dataset              0.9806  0.9952  0.9779  0.9862  0.9947
                   iid_msr              0.0102  0.0172  0.0314  0.0551  0.0177
ViT      vit       (none, replication)  0.0001  0.0001  0.0001  0.0001  0.0001
                   dataset              0.0109  0.0140  0.0018  0.0004  0.0149
                   iid_msr              0.0001  0.0001  0.0001  0.0001  0.0001
```
