# Regret table — best NC predictor vs best baseline

Per (track, split, regime, side): the NC predictor with lowest mean regret (using imputation for binary heads or raw top-1 for regression) is compared to the best baseline. `nc_minus_bl < 0` ⇒ NC wins.

```
 track         split regime    side                            best_nc  nc_regret  nc_ci_lo  nc_ci_hi        best_baseline  bl_regret  bl_ci_lo  bl_ci_hi  nc_minus_bl
     1    lodo_vgg13    far     all                  multilabel/clique      6.452     5.787     7.193           Always-CTM      5.535     4.871     6.181        0.917
     1    lodo_vgg13    far feature                         regression      8.517     7.801     9.250           Always-CTM      5.104     4.455     5.748        3.413
     1    lodo_vgg13    far    head                         regression      4.052     3.633     4.510           Always-MLS      3.765     3.370     4.199        0.287
     1    lodo_vgg13    mid     all     multilabel/within_eps_majority      4.957     4.471     5.511           Always-CTM      5.538     5.055     6.024       -0.582
     1    lodo_vgg13    mid feature                         regression      6.882     6.339     7.481           Always-CTM      4.670     4.211     5.131        2.212
     1    lodo_vgg13    mid    head                         regression      7.901     7.381     8.448        Always-Energy      4.784     4.353     5.252        3.117
     1    lodo_vgg13   near     all     multilabel/within_eps_majority      3.250     2.768     3.776       Always-NNGuide      4.088     3.747     4.512       -0.838
     1    lodo_vgg13   near feature                         regression      5.124     4.560     5.724       Always-NNGuide      3.423     3.089     3.846        1.701
     1    lodo_vgg13   near    head                         regression      4.936     4.500     5.383           Always-MLS      3.458     3.152     3.789        1.478
     1          lopo    far     all per_csf_binary/within_eps_majority      1.693     1.424     1.956           Always-CTM      5.682     5.041     6.353       -3.989
     1          lopo    far feature per_csf_binary/within_eps_majority      2.928     2.464     3.426           Always-CTM      5.300     4.692     5.961       -2.372
     1          lopo    far    head                         regression      5.707     5.057     6.383 Oracle-on-train (PE)      4.407     3.932     4.946        1.300
     1          lopo    mid     all     multilabel/within_eps_majority      1.422     1.238     1.624           Always-CTM      6.525     5.967     7.107       -5.102
     1          lopo    mid feature     multilabel/within_eps_majority      1.697     1.408     2.026           Always-CTM      5.712     5.199     6.272       -4.016
     1          lopo    mid    head                         regression      7.302     6.772     7.880           Always-MLS      5.339     4.942     5.778        1.963
     1          lopo   near     all     multilabel/within_eps_majority      0.960     0.767     1.168       Always-NNGuide      4.483     4.140     4.821       -3.523
     1          lopo   near feature     multilabel/within_eps_majority      1.253     0.919     1.631       Always-NNGuide      3.744     3.398     4.087       -2.491
     1          lopo   near    head                         regression      4.897     4.460     5.333 Oracle-on-train (PE)      3.571     3.281     3.873        1.327
     1 lopo_cnn_only    far     all     multilabel/within_eps_majority      3.047     2.392     3.722           Always-CTM      5.013     4.458     5.561       -1.966
     1 lopo_cnn_only    far feature     multilabel/within_eps_majority      3.689     3.002     4.373           Always-CTM      4.618     4.079     5.168       -0.929
     1 lopo_cnn_only    far    head                         regression      5.213     4.599     5.856           Always-MLS      3.766     3.410     4.158        1.446
     1 lopo_cnn_only    mid     all     multilabel/within_eps_majority      1.581     1.380     1.797           Always-CTM      5.202     4.792     5.645       -3.621
     1 lopo_cnn_only    mid feature     multilabel/within_eps_majority      1.890     1.564     2.266           Always-CTM      4.378     4.000     4.799       -2.488
     1 lopo_cnn_only    mid    head                         regression      6.926     6.398     7.475        Always-Energy      4.687     4.292     5.077        2.239
     1 lopo_cnn_only   near     all     multilabel/within_eps_majority      1.068     0.854     1.315       Always-NNGuide      4.081     3.770     4.442       -3.013
     1 lopo_cnn_only   near feature     multilabel/within_eps_majority      1.392     1.034     1.828       Always-NNGuide      3.441     3.137     3.809       -2.049
     1 lopo_cnn_only   near    head                         regression      4.897     4.481     5.357           Always-MLS      3.464     3.188     3.761        1.432
     1     pxs_vgg13    far     all     multilabel/within_eps_majority      2.981     2.287     3.760           Always-CTM      5.535     4.871     6.181       -2.554
     1     pxs_vgg13    far feature     multilabel/within_eps_majority      3.373     2.684     4.101           Always-CTM      5.104     4.455     5.748       -1.731
     1     pxs_vgg13    far    head                         regression      3.825     3.446     4.244           Always-MLS      3.765     3.370     4.199        0.060
     1     pxs_vgg13    mid     all     multilabel/within_eps_majority      1.880     1.638     2.145           Always-CTM      5.538     5.055     6.024       -3.658
     1     pxs_vgg13    mid feature     multilabel/within_eps_majority      2.059     1.748     2.398           Always-CTM      4.670     4.211     5.131       -2.611
     1     pxs_vgg13    mid    head                         regression      5.842     5.395     6.302        Always-Energy      4.784     4.353     5.252        1.058
     1     pxs_vgg13   near     all     multilabel/within_eps_majority      1.002     0.800     1.225       Always-NNGuide      4.088     3.747     4.512       -3.086
     1     pxs_vgg13   near feature     multilabel/within_eps_majority      1.020     0.758     1.310       Always-NNGuide      3.423     3.089     3.846       -2.403
     1     pxs_vgg13   near    head                         regression      3.789     3.457     4.134           Always-MLS      3.458     3.152     3.789        0.330
     1  single_vgg13    far     all                         regression      6.964     6.399     7.564           Always-CTM      5.535     4.871     6.181        1.429
     1  single_vgg13    far feature     multilabel/within_eps_majority      5.949     4.872     7.072           Always-CTM      5.104     4.455     5.748        0.845
     1  single_vgg13    far    head                         regression      3.765     3.370     4.199           Always-MLS      3.765     3.370     4.199        0.000
     1  single_vgg13    mid     all     multilabel/within_eps_majority      1.211     0.990     1.453           Always-CTM      5.538     5.055     6.024       -4.327
     1  single_vgg13    mid feature     multilabel/within_eps_majority      1.348     1.112     1.604           Always-CTM      4.670     4.211     5.131       -3.322
     1  single_vgg13    mid    head                         regression      4.854     4.464     5.272        Always-Energy      4.784     4.353     5.252        0.070
     1  single_vgg13   near     all     multilabel/within_eps_majority      0.692     0.502     0.923       Always-NNGuide      4.088     3.747     4.512       -3.396
     1  single_vgg13   near feature     multilabel/within_eps_majority      1.290     0.908     1.695       Always-NNGuide      3.423     3.089     3.846       -2.133
     1  single_vgg13   near    head                         regression      3.458     3.152     3.789           Always-MLS      3.458     3.152     3.789        0.000
     1         xarch    far     all     per_csf_binary/within_eps_rank      1.488     1.023     2.075           Always-CTM      2.403     1.777     3.157       -0.915
     1         xarch    far feature     per_csf_binary/within_eps_rank      1.454     0.994     2.047           Always-CTM      2.188     1.589     2.923       -0.734
     1         xarch    far    head                         regression      3.773     2.944     4.753           Always-MLS      3.773     2.944     4.753        0.000
     1         xarch    mid     all     multilabel/within_eps_majority      1.839     1.448     2.308           Always-CTM      3.519     2.939     4.161       -1.680
     1         xarch    mid feature     per_csf_binary/within_eps_rank      1.821     1.474     2.201           Always-CTM      2.919     2.434     3.457       -1.097
     1         xarch    mid    head                         regression      4.438     3.719     5.250        Always-Energy      4.201     3.395     5.101        0.237
     1         xarch   near     all per_csf_binary/within_eps_majority      1.298     0.876     1.790           Always-CTM      2.066     1.474     2.707       -0.768
     1         xarch   near feature      per_csf_binary/within_eps_raw      1.088     0.695     1.537           Always-CTM      1.548     1.038     2.120       -0.460
     1         xarch   near    head                         regression      3.492     2.878     4.183           Always-MLS      3.492     2.878     4.183        0.000
     2    track2_loo    far     all                  multilabel/clique      3.131     1.064     5.620          Always-fDBD      3.361     1.855     5.148       -0.230
     2    track2_loo    far feature                  multilabel/clique      4.030     1.289     7.335          Always-fDBD      3.305     1.811     5.083        0.725
     2    track2_loo    far    head                         regression      5.230     3.351     7.643           Always-MSR      3.369     1.755     5.970        1.861
     2    track2_loo    mid     all                  multilabel/clique      3.823     2.061     5.962           Always-CTM      4.843     2.975     7.072       -1.020
     2    track2_loo    mid feature                         regression      5.060     3.489     6.947           Always-CTM      3.672     2.102     5.649        1.388
     2    track2_loo    mid    head                         regression      6.678     4.430     9.119        Always-Energy      4.264     2.205     6.524        2.414
     2    track2_loo   near     all                  multilabel/clique      0.600     0.202     1.155           Always-CTM      2.630     1.450     3.988       -2.029
     2    track2_loo   near feature                  multilabel/clique      3.087     1.142     5.551           Always-CTM      2.134     1.064     3.440        0.953
     2    track2_loo   near    head                         regression      5.418     4.056     6.947           Always-MLS      3.083     2.260     3.921        2.334
```
