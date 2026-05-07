# Wilcoxon — NC predictors that significantly beat baselines (Holm-corrected, α=0.05)

Sign convention: `median_diff = regret(predictor) − regret(baseline)`. Negative ⇒ NC wins.

```
        split regime    side                          predictor              baseline  median_diff  p_holm
   lodo_vgg13    far     all                  multilabel/clique         Always-Energy       0.0000  0.0000
   lodo_vgg13    far     all                  multilabel/clique            Always-MLS      -0.6028  0.0000
   lodo_vgg13    far     all                  multilabel/clique            Always-MSR      -1.9166  0.0000
   lodo_vgg13    far     all                  multilabel/clique           Always-fDBD      -3.0013  0.0000
   lodo_vgg13    far     all                  multilabel/clique            Random-CSF      -7.7739  0.0000
   lodo_vgg13    far     all     multilabel/within_eps_majority            Random-CSF      -8.0837  0.0000
   lodo_vgg13    far     all                         regression            Random-CSF      -5.7256  0.0000
   lodo_vgg13    far     all     multilabel/within_eps_majority           Always-fDBD      -2.3894  0.0004
   lodo_vgg13    far     all                  multilabel/clique        Always-NNGuide      -0.9759  0.0012
   lodo_vgg13    far     all     multilabel/within_eps_majority            Always-MSR      -1.6068  0.0188
   lodo_vgg13    far feature                         regression            Random-CSF      -6.9829  0.0000
   lodo_vgg13    far feature     multilabel/within_eps_majority            Random-CSF      -6.4492  0.0184
   lodo_vgg13    far    head                  multilabel/clique            Random-CSF      -2.5611  0.0000
   lodo_vgg13    far    head                         regression            Random-CSF      -2.7005  0.0000
   lodo_vgg13    far    head                         regression         Always-Energy      -0.4233  0.0144
   lodo_vgg13    mid     all                  multilabel/clique            Always-MSR      -2.5543  0.0000
   lodo_vgg13    mid     all                  multilabel/clique           Always-fDBD      -2.2848  0.0000
   lodo_vgg13    mid     all                  multilabel/clique            Random-CSF      -8.3702  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority         Always-Energy      -0.8005  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority            Always-MLS      -2.0203  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority            Always-MSR      -4.2308  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority        Always-NNGuide      -0.9510  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority           Always-fDBD      -3.2121  0.0000
   lodo_vgg13    mid     all     multilabel/within_eps_majority            Random-CSF     -11.0448  0.0000
   lodo_vgg13    mid     all         multilabel/within_eps_rank            Always-MSR      -2.4320  0.0000
   lodo_vgg13    mid     all         multilabel/within_eps_rank           Always-fDBD      -2.0883  0.0000
   lodo_vgg13    mid     all         multilabel/within_eps_rank            Random-CSF      -8.9865  0.0000
   lodo_vgg13    mid     all          multilabel/within_eps_raw            Always-MSR      -2.3583  0.0000
   lodo_vgg13    mid     all          multilabel/within_eps_raw           Always-fDBD      -2.0883  0.0000
   lodo_vgg13    mid     all          multilabel/within_eps_raw            Random-CSF      -9.1537  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority         Always-Energy      -0.4305  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority            Always-MLS      -1.5787  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority            Always-MSR      -3.7145  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority        Always-NNGuide      -0.5048  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority           Always-fDBD      -2.8507  0.0000
   lodo_vgg13    mid     all per_csf_binary/within_eps_majority            Random-CSF      -9.9034  0.0000
   lodo_vgg13    mid     all                         regression            Always-MSR      -2.5668  0.0000
   lodo_vgg13    mid     all                         regression           Always-fDBD      -1.8711  0.0000
   lodo_vgg13    mid     all                         regression            Random-CSF      -8.7738  0.0000
   lodo_vgg13    mid feature     multilabel/within_eps_majority           Always-fDBD      -2.1448  0.0000
   lodo_vgg13    mid feature     multilabel/within_eps_majority            Random-CSF     -12.2170  0.0000
   lodo_vgg13    mid feature per_csf_binary/within_eps_majority            Random-CSF     -10.1976  0.0000
   lodo_vgg13    mid feature                         regression           Always-fDBD      -1.8711  0.0000
   lodo_vgg13    mid feature                         regression            Random-CSF     -11.7410  0.0000
   lodo_vgg13    mid feature per_csf_binary/within_eps_majority           Always-fDBD      -1.6333  0.0001
   lodo_vgg13    mid    head     multilabel/within_eps_majority            Random-CSF      -4.3340  0.0000
   lodo_vgg13    mid    head                         regression            Random-CSF      -2.6208  0.0000
   lodo_vgg13   near     all                  multilabel/clique         Always-Energy      -1.3123  0.0000
   lodo_vgg13   near     all                  multilabel/clique            Always-MLS      -0.3096  0.0000
   lodo_vgg13   near     all                  multilabel/clique            Always-MSR      -0.8252  0.0000
   lodo_vgg13   near     all                  multilabel/clique           Always-fDBD      -2.1223  0.0000
   lodo_vgg13   near     all                  multilabel/clique            Random-CSF     -13.1009  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority         Always-Energy      -2.7392  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority            Always-MLS      -1.7976  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority            Always-MSR      -2.7059  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority        Always-NNGuide      -0.5123  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority           Always-fDBD      -2.9331  0.0000
   lodo_vgg13   near     all     multilabel/within_eps_majority            Random-CSF     -14.6399  0.0000
   lodo_vgg13   near     all         multilabel/within_eps_rank         Always-Energy      -1.9816  0.0000
   lodo_vgg13   near     all         multilabel/within_eps_rank            Always-MLS      -1.1944  0.0000
   lodo_vgg13   near     all         multilabel/within_eps_rank            Always-MSR      -2.4697  0.0000
   lodo_vgg13   near     all         multilabel/within_eps_rank           Always-fDBD      -1.3086  0.0000
   lodo_vgg13   near     all         multilabel/within_eps_rank            Random-CSF     -13.8181  0.0000
   lodo_vgg13   near     all          multilabel/within_eps_raw         Always-Energy      -2.3614  0.0000
   lodo_vgg13   near     all          multilabel/within_eps_raw            Always-MLS      -1.3559  0.0000
   lodo_vgg13   near     all          multilabel/within_eps_raw            Always-MSR      -2.4697  0.0000
   lodo_vgg13   near     all          multilabel/within_eps_raw           Always-fDBD      -1.2896  0.0000
   lodo_vgg13   near     all          multilabel/within_eps_raw            Random-CSF     -13.9147  0.0000
   lodo_vgg13   near     all                         regression         Always-Energy      -1.4983  0.0000
   lodo_vgg13   near     all                         regression            Always-MSR      -1.6558  0.0000
   lodo_vgg13   near     all                         regression           Always-fDBD      -1.2722  0.0000
   lodo_vgg13   near     all                         regression            Random-CSF     -12.6935  0.0000
   lodo_vgg13   near     all                         regression            Always-MLS      -0.3276  0.0002
   lodo_vgg13   near     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0036
   lodo_vgg13   near     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0036
   lodo_vgg13   near feature     multilabel/within_eps_majority           Always-fDBD      -1.5981  0.0000
   lodo_vgg13   near feature     multilabel/within_eps_majority            Random-CSF     -21.1291  0.0000
   lodo_vgg13   near feature         multilabel/within_eps_rank            Random-CSF     -19.6190  0.0000
   lodo_vgg13   near feature          multilabel/within_eps_raw            Random-CSF     -19.9843  0.0000
   lodo_vgg13   near feature                         regression           Always-fDBD      -1.2722  0.0000
   lodo_vgg13   near feature                         regression            Random-CSF     -20.6837  0.0000
   lodo_vgg13   near    head                  multilabel/clique         Always-Energy      -0.9647  0.0000
   lodo_vgg13   near    head                  multilabel/clique            Random-CSF      -6.5093  0.0000
   lodo_vgg13   near    head     multilabel/within_eps_majority            Random-CSF      -6.5545  0.0000
   lodo_vgg13   near    head         multilabel/within_eps_rank            Random-CSF      -5.6697  0.0000
   lodo_vgg13   near    head          multilabel/within_eps_raw            Random-CSF      -6.0372  0.0000
   lodo_vgg13   near    head                         regression            Random-CSF      -4.8724  0.0000
   lodo_vgg13   near    head                  multilabel/clique            Always-MSR      -0.5704  0.0001
   lodo_vgg13   near    head     multilabel/within_eps_majority         Always-Energy      -1.2477  0.0002
   lodo_vgg13   near    head     multilabel/within_eps_majority            Always-MSR      -1.0208  0.0036
         lopo    far     all                  multilabel/clique            Always-CTM      -0.2217  0.0000
         lopo    far     all                  multilabel/clique         Always-Energy      -2.8136  0.0000
         lopo    far     all                  multilabel/clique            Always-MLS      -2.7585  0.0000
         lopo    far     all                  multilabel/clique            Always-MSR      -3.9815  0.0000
         lopo    far     all                  multilabel/clique        Always-NNGuide      -3.1235  0.0000
         lopo    far     all                  multilabel/clique           Always-fDBD      -3.4531  0.0000
         lopo    far     all                  multilabel/clique Oracle-on-train (CTM)      -0.2217  0.0000
         lopo    far     all                  multilabel/clique            Random-CSF     -10.2215  0.0000
         lopo    far     all     multilabel/within_eps_majority            Always-CTM      -0.8334  0.0000
         lopo    far     all     multilabel/within_eps_majority         Always-Energy      -3.9173  0.0000
         lopo    far     all     multilabel/within_eps_majority            Always-MLS      -3.6733  0.0000
         lopo    far     all     multilabel/within_eps_majority            Always-MSR      -4.1846  0.0000
         lopo    far     all     multilabel/within_eps_majority        Always-NNGuide      -3.7291  0.0000
         lopo    far     all     multilabel/within_eps_majority           Always-fDBD      -4.3341  0.0000
         lopo    far     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.8334  0.0000
         lopo    far     all     multilabel/within_eps_majority            Random-CSF     -11.1359  0.0000
         lopo    far     all         multilabel/within_eps_rank         Always-Energy      -2.7830  0.0000
         lopo    far     all         multilabel/within_eps_rank            Always-MLS      -2.9288  0.0000
         lopo    far     all         multilabel/within_eps_rank            Always-MSR      -3.2308  0.0000
         lopo    far     all         multilabel/within_eps_rank        Always-NNGuide      -3.1292  0.0000
         lopo    far     all         multilabel/within_eps_rank           Always-fDBD      -1.9274  0.0000
         lopo    far     all         multilabel/within_eps_rank            Random-CSF      -9.5888  0.0000
         lopo    far     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    far     all          multilabel/within_eps_raw         Always-Energy      -3.7428  0.0000
         lopo    far     all          multilabel/within_eps_raw            Always-MLS      -3.5156  0.0000
         lopo    far     all          multilabel/within_eps_raw            Always-MSR      -3.8427  0.0000
         lopo    far     all          multilabel/within_eps_raw        Always-NNGuide      -3.5296  0.0000
         lopo    far     all          multilabel/within_eps_raw           Always-fDBD      -2.9187  0.0000
         lopo    far     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far     all          multilabel/within_eps_raw            Random-CSF     -10.7083  0.0000
         lopo    far     all              per_csf_binary/clique            Always-CTM       0.0000  0.0000
         lopo    far     all              per_csf_binary/clique         Always-Energy      -2.1432  0.0000
         lopo    far     all              per_csf_binary/clique            Always-MLS      -2.2102  0.0000
         lopo    far     all              per_csf_binary/clique            Always-MSR      -3.2257  0.0000
         lopo    far     all              per_csf_binary/clique        Always-NNGuide      -2.7297  0.0000
         lopo    far     all              per_csf_binary/clique           Always-fDBD      -2.7693  0.0000
         lopo    far     all              per_csf_binary/clique Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far     all              per_csf_binary/clique            Random-CSF      -9.9804  0.0000
         lopo    far     all per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo    far     all per_csf_binary/within_eps_majority         Always-Energy      -3.9668  0.0000
         lopo    far     all per_csf_binary/within_eps_majority            Always-MLS      -3.6733  0.0000
         lopo    far     all per_csf_binary/within_eps_majority            Always-MSR      -4.2831  0.0000
         lopo    far     all per_csf_binary/within_eps_majority        Always-NNGuide      -3.7751  0.0000
         lopo    far     all per_csf_binary/within_eps_majority           Always-fDBD      -4.4421  0.0000
         lopo    far     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far     all per_csf_binary/within_eps_majority            Random-CSF     -11.0031  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank         Always-Energy      -3.5022  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank            Always-MLS      -3.4340  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank            Always-MSR      -3.3148  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank        Always-NNGuide      -3.4865  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank           Always-fDBD      -2.6800  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank            Random-CSF      -9.6555  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw         Always-Energy      -4.1219  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw            Always-MLS      -3.6649  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw            Always-MSR      -4.0447  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw        Always-NNGuide      -3.6529  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw           Always-fDBD      -3.4104  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far     all      per_csf_binary/within_eps_raw            Random-CSF     -10.8764  0.0000
         lopo    far     all                         regression         Always-Energy      -1.2733  0.0000
         lopo    far     all                         regression            Always-MLS      -0.7468  0.0000
         lopo    far     all                         regression            Always-MSR      -1.5651  0.0000
         lopo    far     all                         regression           Always-fDBD      -0.7847  0.0000
         lopo    far     all                         regression            Random-CSF      -6.1569  0.0000
         lopo    far     all     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0011
         lopo    far     all     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0011
         lopo    far feature                  multilabel/clique        Always-NNGuide      -1.6926  0.0000
         lopo    far feature                  multilabel/clique           Always-fDBD      -1.3780  0.0000
         lopo    far feature                  multilabel/clique            Random-CSF      -8.4782  0.0000
         lopo    far feature     multilabel/within_eps_majority            Always-CTM      -0.0521  0.0000
         lopo    far feature     multilabel/within_eps_majority        Always-NNGuide      -3.1292  0.0000
         lopo    far feature     multilabel/within_eps_majority           Always-fDBD      -3.6469  0.0000
         lopo    far feature     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.0521  0.0000
         lopo    far feature     multilabel/within_eps_majority            Random-CSF     -10.9005  0.0000
         lopo    far feature         multilabel/within_eps_rank           Always-fDBD      -1.3641  0.0000
         lopo    far feature         multilabel/within_eps_rank            Random-CSF      -8.6626  0.0000
         lopo    far feature          multilabel/within_eps_raw        Always-NNGuide      -3.0072  0.0000
         lopo    far feature          multilabel/within_eps_raw           Always-fDBD      -2.6493  0.0000
         lopo    far feature          multilabel/within_eps_raw            Random-CSF     -10.3868  0.0000
         lopo    far feature              per_csf_binary/clique            Random-CSF      -6.9149  0.0000
         lopo    far feature per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo    far feature per_csf_binary/within_eps_majority        Always-NNGuide      -3.1474  0.0000
         lopo    far feature per_csf_binary/within_eps_majority           Always-fDBD      -3.4987  0.0000
         lopo    far feature per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far feature per_csf_binary/within_eps_majority            Random-CSF     -11.4391  0.0000
         lopo    far feature     per_csf_binary/within_eps_rank           Always-fDBD      -2.3662  0.0000
         lopo    far feature     per_csf_binary/within_eps_rank            Random-CSF      -8.4782  0.0000
         lopo    far feature      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    far feature      per_csf_binary/within_eps_raw        Always-NNGuide      -3.1488  0.0000
         lopo    far feature      per_csf_binary/within_eps_raw           Always-fDBD      -2.8692  0.0000
         lopo    far feature      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    far feature      per_csf_binary/within_eps_raw            Random-CSF     -10.6632  0.0000
         lopo    far feature                         regression           Always-fDBD      -0.7847  0.0000
         lopo    far feature                         regression            Random-CSF      -7.6206  0.0000
         lopo    far feature     per_csf_binary/within_eps_rank        Always-NNGuide      -2.8308  0.0003
         lopo    far feature              per_csf_binary/clique           Always-fDBD       0.0000  0.0004
         lopo    far feature          multilabel/within_eps_raw            Always-CTM       0.0000  0.0011
         lopo    far feature          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0011
         lopo    far feature         multilabel/within_eps_rank        Always-NNGuide      -2.6964  0.0407
         lopo    far    head                         regression            Always-MSR      -0.5175  0.0000
         lopo    far    head                         regression            Random-CSF      -2.7083  0.0000
         lopo    mid     all                  multilabel/clique            Always-CTM       0.0000  0.0000
         lopo    mid     all                  multilabel/clique         Always-Energy      -1.6413  0.0000
         lopo    mid     all                  multilabel/clique            Always-MLS      -2.5326  0.0000
         lopo    mid     all                  multilabel/clique            Always-MSR      -5.3075  0.0000
         lopo    mid     all                  multilabel/clique        Always-NNGuide      -1.7674  0.0000
         lopo    mid     all                  multilabel/clique           Always-fDBD      -3.3314  0.0000
         lopo    mid     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all                  multilabel/clique            Random-CSF     -11.6124  0.0000
         lopo    mid     all     multilabel/within_eps_majority            Always-CTM      -1.3560  0.0000
         lopo    mid     all     multilabel/within_eps_majority         Always-Energy      -3.0895  0.0000
         lopo    mid     all     multilabel/within_eps_majority            Always-MLS      -3.4301  0.0000
         lopo    mid     all     multilabel/within_eps_majority            Always-MSR      -6.3524  0.0000
         lopo    mid     all     multilabel/within_eps_majority        Always-NNGuide      -2.4890  0.0000
         lopo    mid     all     multilabel/within_eps_majority           Always-fDBD      -5.2810  0.0000
         lopo    mid     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -1.3560  0.0000
         lopo    mid     all     multilabel/within_eps_majority            Random-CSF     -13.0249  0.0000
         lopo    mid     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
         lopo    mid     all         multilabel/within_eps_rank         Always-Energy      -2.7094  0.0000
         lopo    mid     all         multilabel/within_eps_rank            Always-MLS      -3.2659  0.0000
         lopo    mid     all         multilabel/within_eps_rank            Always-MSR      -5.8379  0.0000
         lopo    mid     all         multilabel/within_eps_rank        Always-NNGuide      -2.2096  0.0000
         lopo    mid     all         multilabel/within_eps_rank           Always-fDBD      -3.8063  0.0000
         lopo    mid     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all         multilabel/within_eps_rank            Random-CSF     -11.7145  0.0000
         lopo    mid     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    mid     all          multilabel/within_eps_raw         Always-Energy      -2.6376  0.0000
         lopo    mid     all          multilabel/within_eps_raw            Always-MLS      -3.1927  0.0000
         lopo    mid     all          multilabel/within_eps_raw            Always-MSR      -5.9126  0.0000
         lopo    mid     all          multilabel/within_eps_raw        Always-NNGuide      -2.1873  0.0000
         lopo    mid     all          multilabel/within_eps_raw           Always-fDBD      -3.8063  0.0000
         lopo    mid     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all          multilabel/within_eps_raw            Random-CSF     -12.1073  0.0000
         lopo    mid     all              per_csf_binary/clique            Always-CTM       0.0000  0.0000
         lopo    mid     all              per_csf_binary/clique         Always-Energy      -1.3360  0.0000
         lopo    mid     all              per_csf_binary/clique            Always-MLS      -2.2991  0.0000
         lopo    mid     all              per_csf_binary/clique            Always-MSR      -4.9450  0.0000
         lopo    mid     all              per_csf_binary/clique        Always-NNGuide      -1.6896  0.0000
         lopo    mid     all              per_csf_binary/clique           Always-fDBD      -3.1651  0.0000
         lopo    mid     all              per_csf_binary/clique Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all              per_csf_binary/clique            Random-CSF     -11.3328  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority            Always-CTM      -0.3305  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority         Always-Energy      -3.1614  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority            Always-MLS      -3.4074  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority            Always-MSR      -6.1452  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority        Always-NNGuide      -2.4822  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority           Always-fDBD      -4.9159  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)      -0.3305  0.0000
         lopo    mid     all per_csf_binary/within_eps_majority            Random-CSF     -12.6261  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank         Always-Energy      -2.5672  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank            Always-MLS      -3.1619  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank            Always-MSR      -5.5993  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank        Always-NNGuide      -2.1038  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank           Always-fDBD      -3.7484  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all     per_csf_binary/within_eps_rank            Random-CSF     -11.3837  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw         Always-Energy      -2.5929  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw            Always-MLS      -3.2113  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw            Always-MSR      -5.7140  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw        Always-NNGuide      -2.1870  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw           Always-fDBD      -4.2997  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid     all      per_csf_binary/within_eps_raw            Random-CSF     -12.2237  0.0000
         lopo    mid     all                         regression         Always-Energy      -0.5950  0.0000
         lopo    mid     all                         regression            Always-MLS      -0.8245  0.0000
         lopo    mid     all                         regression            Always-MSR      -2.9126  0.0000
         lopo    mid     all                         regression           Always-fDBD      -2.1142  0.0000
         lopo    mid     all                         regression            Random-CSF      -8.5418  0.0000
         lopo    mid feature                  multilabel/clique           Always-fDBD      -0.5657  0.0000
         lopo    mid feature                  multilabel/clique            Random-CSF      -9.7802  0.0000
         lopo    mid feature     multilabel/within_eps_majority            Always-CTM      -0.3967  0.0000
         lopo    mid feature     multilabel/within_eps_majority        Always-NNGuide      -1.8172  0.0000
         lopo    mid feature     multilabel/within_eps_majority           Always-fDBD      -5.0132  0.0000
         lopo    mid feature     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.3967  0.0000
         lopo    mid feature     multilabel/within_eps_majority            Random-CSF     -14.0626  0.0000
         lopo    mid feature         multilabel/within_eps_rank        Always-NNGuide      -1.5955  0.0000
         lopo    mid feature         multilabel/within_eps_rank           Always-fDBD      -3.2080  0.0000
         lopo    mid feature         multilabel/within_eps_rank            Random-CSF     -12.0987  0.0000
         lopo    mid feature          multilabel/within_eps_raw        Always-NNGuide      -1.4519  0.0000
         lopo    mid feature          multilabel/within_eps_raw           Always-fDBD      -3.1060  0.0000
         lopo    mid feature          multilabel/within_eps_raw            Random-CSF     -12.5617  0.0000
         lopo    mid feature              per_csf_binary/clique            Random-CSF      -8.0622  0.0000
         lopo    mid feature per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo    mid feature per_csf_binary/within_eps_majority        Always-NNGuide      -1.8691  0.0000
         lopo    mid feature per_csf_binary/within_eps_majority           Always-fDBD      -4.4679  0.0000
         lopo    mid feature per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid feature per_csf_binary/within_eps_majority            Random-CSF     -14.0272  0.0000
         lopo    mid feature     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0000
         lopo    mid feature     per_csf_binary/within_eps_rank        Always-NNGuide      -1.4649  0.0000
         lopo    mid feature     per_csf_binary/within_eps_rank           Always-fDBD      -3.3424  0.0000
         lopo    mid feature     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid feature     per_csf_binary/within_eps_rank            Random-CSF     -12.0842  0.0000
         lopo    mid feature      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo    mid feature      per_csf_binary/within_eps_raw        Always-NNGuide      -1.5322  0.0000
         lopo    mid feature      per_csf_binary/within_eps_raw           Always-fDBD      -3.5208  0.0000
         lopo    mid feature      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo    mid feature      per_csf_binary/within_eps_raw            Random-CSF     -13.6066  0.0000
         lopo    mid feature                         regression           Always-fDBD      -2.1142  0.0000
         lopo    mid feature                         regression            Random-CSF     -10.8299  0.0000
         lopo    mid feature          multilabel/within_eps_raw            Always-CTM       0.0000  0.0001
         lopo    mid feature          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0001
         lopo    mid feature         multilabel/within_eps_rank            Always-CTM       0.0000  0.0004
         lopo    mid feature         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0004
         lopo    mid feature              per_csf_binary/clique           Always-fDBD       0.0000  0.0068
         lopo    mid    head                         regression            Always-MSR      -1.0557  0.0000
         lopo    mid    head                         regression            Random-CSF      -3.2424  0.0000
         lopo   near     all                  multilabel/clique            Always-CTM       0.0000  0.0000
         lopo   near     all                  multilabel/clique         Always-Energy      -2.7498  0.0000
         lopo   near     all                  multilabel/clique            Always-MLS      -1.8532  0.0000
         lopo   near     all                  multilabel/clique            Always-MSR      -2.9057  0.0000
         lopo   near     all                  multilabel/clique        Always-NNGuide      -1.7747  0.0000
         lopo   near     all                  multilabel/clique           Always-fDBD      -3.8087  0.0000
         lopo   near     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all                  multilabel/clique            Random-CSF     -14.9477  0.0000
         lopo   near     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo   near     all     multilabel/within_eps_majority         Always-Energy      -3.8459  0.0000
         lopo   near     all     multilabel/within_eps_majority            Always-MLS      -2.9891  0.0000
         lopo   near     all     multilabel/within_eps_majority            Always-MSR      -4.6940  0.0000
         lopo   near     all     multilabel/within_eps_majority        Always-NNGuide      -2.4057  0.0000
         lopo   near     all     multilabel/within_eps_majority           Always-fDBD      -4.6529  0.0000
         lopo   near     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all     multilabel/within_eps_majority            Random-CSF     -17.1764  0.0000
         lopo   near     all         multilabel/within_eps_rank         Always-Energy      -2.1849  0.0000
         lopo   near     all         multilabel/within_eps_rank            Always-MLS      -1.4012  0.0000
         lopo   near     all         multilabel/within_eps_rank            Always-MSR      -2.5088  0.0000
         lopo   near     all         multilabel/within_eps_rank        Always-NNGuide      -1.4671  0.0000
         lopo   near     all         multilabel/within_eps_rank           Always-fDBD      -2.9408  0.0000
         lopo   near     all         multilabel/within_eps_rank            Random-CSF     -11.4381  0.0000
         lopo   near     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo   near     all          multilabel/within_eps_raw         Always-Energy      -2.6916  0.0000
         lopo   near     all          multilabel/within_eps_raw            Always-MLS      -1.9129  0.0000
         lopo   near     all          multilabel/within_eps_raw            Always-MSR      -3.0968  0.0000
         lopo   near     all          multilabel/within_eps_raw        Always-NNGuide      -1.6020  0.0000
         lopo   near     all          multilabel/within_eps_raw           Always-fDBD      -3.0467  0.0000
         lopo   near     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all          multilabel/within_eps_raw            Random-CSF     -14.4108  0.0000
         lopo   near     all              per_csf_binary/clique            Always-CTM       0.0000  0.0000
         lopo   near     all              per_csf_binary/clique         Always-Energy      -2.5824  0.0000
         lopo   near     all              per_csf_binary/clique            Always-MLS      -1.7060  0.0000
         lopo   near     all              per_csf_binary/clique            Always-MSR      -2.8175  0.0000
         lopo   near     all              per_csf_binary/clique        Always-NNGuide      -1.7639  0.0000
         lopo   near     all              per_csf_binary/clique           Always-fDBD      -3.5328  0.0000
         lopo   near     all              per_csf_binary/clique Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all              per_csf_binary/clique            Random-CSF     -14.5381  0.0000
         lopo   near     all per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo   near     all per_csf_binary/within_eps_majority         Always-Energy      -3.6167  0.0000
         lopo   near     all per_csf_binary/within_eps_majority            Always-MLS      -2.9044  0.0000
         lopo   near     all per_csf_binary/within_eps_majority            Always-MSR      -4.4727  0.0000
         lopo   near     all per_csf_binary/within_eps_majority        Always-NNGuide      -2.3782  0.0000
         lopo   near     all per_csf_binary/within_eps_majority           Always-fDBD      -4.5705  0.0000
         lopo   near     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all per_csf_binary/within_eps_majority            Random-CSF     -16.8401  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank         Always-Energy      -2.0263  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank            Always-MLS      -1.1859  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank            Always-MSR      -2.3599  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank        Always-NNGuide      -1.4881  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank           Always-fDBD      -3.0111  0.0000
         lopo   near     all     per_csf_binary/within_eps_rank            Random-CSF     -11.3151  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw         Always-Energy      -2.8192  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw            Always-MLS      -2.0774  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw            Always-MSR      -3.2789  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw        Always-NNGuide      -1.7021  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw           Always-fDBD      -3.5629  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near     all      per_csf_binary/within_eps_raw            Random-CSF     -14.5490  0.0000
         lopo   near     all                         regression         Always-Energy      -1.2177  0.0000
         lopo   near     all                         regression            Always-MLS      -0.4462  0.0000
         lopo   near     all                         regression            Always-MSR      -1.9593  0.0000
         lopo   near     all                         regression           Always-fDBD      -2.2261  0.0000
         lopo   near     all                         regression            Random-CSF     -12.6166  0.0000
         lopo   near feature                  multilabel/clique           Always-fDBD      -1.7123  0.0000
         lopo   near feature                  multilabel/clique            Random-CSF     -21.0119  0.0000
         lopo   near feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo   near feature     multilabel/within_eps_majority        Always-NNGuide      -1.7192  0.0000
         lopo   near feature     multilabel/within_eps_majority           Always-fDBD      -4.0705  0.0000
         lopo   near feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near feature     multilabel/within_eps_majority            Random-CSF     -22.1426  0.0000
         lopo   near feature         multilabel/within_eps_rank            Random-CSF     -11.8664  0.0000
         lopo   near feature          multilabel/within_eps_raw        Always-NNGuide      -0.4989  0.0000
         lopo   near feature          multilabel/within_eps_raw           Always-fDBD      -2.3584  0.0000
         lopo   near feature          multilabel/within_eps_raw            Random-CSF     -21.4378  0.0000
         lopo   near feature              per_csf_binary/clique            Random-CSF     -19.5454  0.0000
         lopo   near feature per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
         lopo   near feature per_csf_binary/within_eps_majority        Always-NNGuide      -1.8779  0.0000
         lopo   near feature per_csf_binary/within_eps_majority           Always-fDBD      -4.0184  0.0000
         lopo   near feature per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
         lopo   near feature per_csf_binary/within_eps_majority            Random-CSF     -22.1426  0.0000
         lopo   near feature     per_csf_binary/within_eps_rank            Random-CSF     -15.4967  0.0000
         lopo   near feature      per_csf_binary/within_eps_raw        Always-NNGuide      -0.7049  0.0000
         lopo   near feature      per_csf_binary/within_eps_raw           Always-fDBD      -2.3833  0.0000
         lopo   near feature      per_csf_binary/within_eps_raw            Random-CSF     -21.0970  0.0000
         lopo   near feature                         regression           Always-fDBD      -2.2261  0.0000
         lopo   near feature                         regression            Random-CSF     -19.8178  0.0000
         lopo   near feature              per_csf_binary/clique           Always-fDBD       0.0000  0.0026
         lopo   near feature                  multilabel/clique        Always-NNGuide      -0.6352  0.0029
         lopo   near    head                         regression         Always-Energy      -0.7548  0.0000
         lopo   near    head                         regression            Always-MSR      -0.4001  0.0000
         lopo   near    head                         regression            Random-CSF      -6.2603  0.0000
lopo_cnn_only    far     all                  multilabel/clique            Always-CTM       0.0000  0.0000
lopo_cnn_only    far     all                  multilabel/clique         Always-Energy      -3.2757  0.0000
lopo_cnn_only    far     all                  multilabel/clique            Always-MLS      -3.0887  0.0000
lopo_cnn_only    far     all                  multilabel/clique            Always-MSR      -4.0447  0.0000
lopo_cnn_only    far     all                  multilabel/clique        Always-NNGuide      -3.4667  0.0000
lopo_cnn_only    far     all                  multilabel/clique           Always-fDBD      -3.9193  0.0000
lopo_cnn_only    far     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    far     all                  multilabel/clique            Random-CSF     -10.4947  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority         Always-Energy      -3.9798  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority            Always-MLS      -3.7461  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority            Always-MSR      -4.1380  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority        Always-NNGuide      -3.7291  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority           Always-fDBD      -4.3341  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    far     all     multilabel/within_eps_majority            Random-CSF     -11.1069  0.0000
lopo_cnn_only    far     all         multilabel/within_eps_rank           Always-fDBD      -1.9274  0.0000
lopo_cnn_only    far     all         multilabel/within_eps_rank            Random-CSF      -9.5611  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw         Always-Energy      -3.9173  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw            Always-MLS      -3.6233  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw            Always-MSR      -3.5440  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw        Always-NNGuide      -3.4962  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw           Always-fDBD      -2.8734  0.0000
lopo_cnn_only    far     all          multilabel/within_eps_raw            Random-CSF     -10.6847  0.0000
lopo_cnn_only    far     all                         regression         Always-Energy      -1.6829  0.0000
lopo_cnn_only    far     all                         regression            Always-MLS      -1.1459  0.0000
lopo_cnn_only    far     all                         regression            Always-MSR      -1.6894  0.0000
lopo_cnn_only    far     all                         regression           Always-fDBD      -1.3897  0.0000
lopo_cnn_only    far     all                         regression            Random-CSF      -6.6817  0.0000
lopo_cnn_only    far     all         multilabel/within_eps_rank            Always-MSR      -3.2257  0.0001
lopo_cnn_only    far     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0033
lopo_cnn_only    far     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0033
lopo_cnn_only    far     all         multilabel/within_eps_rank        Always-NNGuide      -3.1635  0.0034
lopo_cnn_only    far     all         multilabel/within_eps_rank         Always-Energy      -2.8664  0.0074
lopo_cnn_only    far     all         multilabel/within_eps_rank            Always-MLS      -3.0183  0.0168
lopo_cnn_only    far     all                         regression        Always-NNGuide       0.0000  0.0468
lopo_cnn_only    far feature                  multilabel/clique           Always-fDBD      -1.6469  0.0000
lopo_cnn_only    far feature                  multilabel/clique            Random-CSF      -9.1975  0.0000
lopo_cnn_only    far feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
lopo_cnn_only    far feature     multilabel/within_eps_majority        Always-NNGuide      -3.1292  0.0000
lopo_cnn_only    far feature     multilabel/within_eps_majority           Always-fDBD      -3.6239  0.0000
lopo_cnn_only    far feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    far feature     multilabel/within_eps_majority            Random-CSF     -11.5576  0.0000
lopo_cnn_only    far feature         multilabel/within_eps_rank            Random-CSF      -9.1238  0.0000
lopo_cnn_only    far feature          multilabel/within_eps_raw           Always-fDBD      -2.4201  0.0000
lopo_cnn_only    far feature          multilabel/within_eps_raw            Random-CSF     -10.8410  0.0000
lopo_cnn_only    far feature                         regression           Always-fDBD      -1.3897  0.0000
lopo_cnn_only    far feature                         regression            Random-CSF      -8.3885  0.0000
lopo_cnn_only    far feature          multilabel/within_eps_raw        Always-NNGuide      -3.0836  0.0001
lopo_cnn_only    far feature                  multilabel/clique        Always-NNGuide      -2.2907  0.0004
lopo_cnn_only    far feature         multilabel/within_eps_rank           Always-fDBD      -1.2410  0.0007
lopo_cnn_only    far    head                         regression            Random-CSF      -2.7810  0.0000
lopo_cnn_only    mid     all                  multilabel/clique            Always-CTM       0.0000  0.0000
lopo_cnn_only    mid     all                  multilabel/clique         Always-Energy      -1.4620  0.0000
lopo_cnn_only    mid     all                  multilabel/clique            Always-MLS      -2.4370  0.0000
lopo_cnn_only    mid     all                  multilabel/clique            Always-MSR      -5.5700  0.0000
lopo_cnn_only    mid     all                  multilabel/clique        Always-NNGuide      -1.7727  0.0000
lopo_cnn_only    mid     all                  multilabel/clique           Always-fDBD      -3.7538  0.0000
lopo_cnn_only    mid     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    mid     all                  multilabel/clique            Random-CSF     -11.7749  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority            Always-CTM      -0.6464  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority         Always-Energy      -2.6819  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority            Always-MLS      -3.2309  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority            Always-MSR      -6.2056  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority        Always-NNGuide      -2.2079  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority           Always-fDBD      -5.4191  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.6464  0.0000
lopo_cnn_only    mid     all     multilabel/within_eps_majority            Random-CSF     -13.2581  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank         Always-Energy      -2.1641  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank            Always-MLS      -3.0621  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank            Always-MSR      -5.7239  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank        Always-NNGuide      -1.9651  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank           Always-fDBD      -3.7180  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    mid     all         multilabel/within_eps_rank            Random-CSF     -11.8650  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw         Always-Energy      -2.1166  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw            Always-MLS      -3.0408  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw            Always-MSR      -5.8321  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw        Always-NNGuide      -1.9741  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw           Always-fDBD      -3.9323  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    mid     all          multilabel/within_eps_raw            Random-CSF     -12.5020  0.0000
lopo_cnn_only    mid     all                         regression         Always-Energy      -0.8634  0.0000
lopo_cnn_only    mid     all                         regression            Always-MLS      -1.0026  0.0000
lopo_cnn_only    mid     all                         regression            Always-MSR      -3.2955  0.0000
lopo_cnn_only    mid     all                         regression           Always-fDBD      -3.1685  0.0000
lopo_cnn_only    mid     all                         regression            Random-CSF      -9.5530  0.0000
lopo_cnn_only    mid     all                         regression        Always-NNGuide       0.0000  0.0369
lopo_cnn_only    mid feature                  multilabel/clique            Random-CSF     -10.2408  0.0000
lopo_cnn_only    mid feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
lopo_cnn_only    mid feature     multilabel/within_eps_majority        Always-NNGuide      -1.4061  0.0000
lopo_cnn_only    mid feature     multilabel/within_eps_majority           Always-fDBD      -5.0465  0.0000
lopo_cnn_only    mid feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only    mid feature     multilabel/within_eps_majority            Random-CSF     -14.5600  0.0000
lopo_cnn_only    mid feature         multilabel/within_eps_rank        Always-NNGuide      -1.1745  0.0000
lopo_cnn_only    mid feature         multilabel/within_eps_rank           Always-fDBD      -3.0344  0.0000
lopo_cnn_only    mid feature         multilabel/within_eps_rank            Random-CSF     -12.6529  0.0000
lopo_cnn_only    mid feature          multilabel/within_eps_raw        Always-NNGuide      -1.0980  0.0000
lopo_cnn_only    mid feature          multilabel/within_eps_raw           Always-fDBD      -3.1060  0.0000
lopo_cnn_only    mid feature          multilabel/within_eps_raw            Random-CSF     -13.2612  0.0000
lopo_cnn_only    mid feature                         regression           Always-fDBD      -3.1685  0.0000
lopo_cnn_only    mid feature                         regression            Random-CSF     -12.2866  0.0000
lopo_cnn_only    mid feature                  multilabel/clique           Always-fDBD      -0.4512  0.0001
lopo_cnn_only    mid    head                         regression            Always-MSR      -1.1930  0.0000
lopo_cnn_only    mid    head                         regression            Random-CSF      -3.8389  0.0000
lopo_cnn_only   near     all                  multilabel/clique            Always-CTM       0.0000  0.0000
lopo_cnn_only   near     all                  multilabel/clique         Always-Energy      -3.2262  0.0000
lopo_cnn_only   near     all                  multilabel/clique            Always-MLS      -2.0673  0.0000
lopo_cnn_only   near     all                  multilabel/clique            Always-MSR      -3.3281  0.0000
lopo_cnn_only   near     all                  multilabel/clique        Always-NNGuide      -1.7080  0.0000
lopo_cnn_only   near     all                  multilabel/clique           Always-fDBD      -3.7772  0.0000
lopo_cnn_only   near     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only   near     all                  multilabel/clique            Random-CSF     -15.7458  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority         Always-Energy      -4.0111  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority            Always-MLS      -3.0633  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority            Always-MSR      -4.8028  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority        Always-NNGuide      -2.3059  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority           Always-fDBD      -4.4930  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only   near     all     multilabel/within_eps_majority            Random-CSF     -17.4520  0.0000
lopo_cnn_only   near     all         multilabel/within_eps_rank            Random-CSF     -11.1550  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw         Always-Energy      -2.6838  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw            Always-MLS      -1.7607  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw            Always-MSR      -2.9846  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw        Always-NNGuide      -1.2278  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw           Always-fDBD      -2.7501  0.0000
lopo_cnn_only   near     all          multilabel/within_eps_raw            Random-CSF     -14.7608  0.0000
lopo_cnn_only   near     all                         regression         Always-Energy      -1.7193  0.0000
lopo_cnn_only   near     all                         regression            Always-MLS      -0.9727  0.0000
lopo_cnn_only   near     all                         regression            Always-MSR      -2.4255  0.0000
lopo_cnn_only   near     all                         regression           Always-fDBD      -2.3825  0.0000
lopo_cnn_only   near     all                         regression            Random-CSF     -13.2584  0.0000
lopo_cnn_only   near     all         multilabel/within_eps_rank            Always-MSR      -2.3296  0.0001
lopo_cnn_only   near     all         multilabel/within_eps_rank           Always-fDBD      -2.6158  0.0020
lopo_cnn_only   near     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0035
lopo_cnn_only   near     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0035
lopo_cnn_only   near     all         multilabel/within_eps_rank            Always-MLS      -0.9923  0.0077
lopo_cnn_only   near     all         multilabel/within_eps_rank         Always-Energy      -1.8996  0.0220
lopo_cnn_only   near feature                  multilabel/clique           Always-fDBD      -1.6111  0.0000
lopo_cnn_only   near feature                  multilabel/clique            Random-CSF     -21.7319  0.0000
lopo_cnn_only   near feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
lopo_cnn_only   near feature     multilabel/within_eps_majority        Always-NNGuide      -1.6358  0.0000
lopo_cnn_only   near feature     multilabel/within_eps_majority           Always-fDBD      -4.0624  0.0000
lopo_cnn_only   near feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
lopo_cnn_only   near feature     multilabel/within_eps_majority            Random-CSF     -22.7144  0.0000
lopo_cnn_only   near feature          multilabel/within_eps_raw           Always-fDBD      -2.1578  0.0000
lopo_cnn_only   near feature          multilabel/within_eps_raw            Random-CSF     -21.8519  0.0000
lopo_cnn_only   near feature                         regression           Always-fDBD      -2.3825  0.0000
lopo_cnn_only   near feature                         regression            Random-CSF     -20.7256  0.0000
lopo_cnn_only   near feature          multilabel/within_eps_raw        Always-NNGuide      -0.1189  0.0151
lopo_cnn_only   near feature         multilabel/within_eps_rank            Random-CSF     -11.3570  0.0430
lopo_cnn_only   near    head                         regression         Always-Energy      -0.8967  0.0000
lopo_cnn_only   near    head                         regression            Random-CSF      -6.2080  0.0000
lopo_cnn_only   near    head                         regression            Always-MSR      -0.3324  0.0003
    pxs_vgg13    far     all                  multilabel/clique            Always-MLS      -1.6308  0.0000
    pxs_vgg13    far     all                  multilabel/clique            Always-MSR      -2.6623  0.0000
    pxs_vgg13    far     all                  multilabel/clique        Always-NNGuide      -2.9011  0.0000
    pxs_vgg13    far     all                  multilabel/clique           Always-fDBD      -2.6036  0.0000
    pxs_vgg13    far     all                  multilabel/clique            Random-CSF      -8.6967  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority            Always-CTM      -1.5900  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority         Always-Energy      -3.2220  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority            Always-MLS      -3.2922  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority            Always-MSR      -3.4616  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority        Always-NNGuide      -3.4558  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority           Always-fDBD      -5.0143  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -1.5900  0.0000
    pxs_vgg13    far     all     multilabel/within_eps_majority            Random-CSF     -11.3540  0.0000
    pxs_vgg13    far     all         multilabel/within_eps_rank            Random-CSF      -9.3840  0.0000
    pxs_vgg13    far     all                         regression            Random-CSF      -5.7256  0.0000
    pxs_vgg13    far     all          multilabel/within_eps_raw            Random-CSF     -10.4036  0.0001
    pxs_vgg13    far     all                  multilabel/clique         Always-Energy      -0.7647  0.0059
    pxs_vgg13    far     all         multilabel/within_eps_rank            Always-MSR      -2.8806  0.0118
    pxs_vgg13    far feature                  multilabel/clique            Random-CSF      -8.0033  0.0000
    pxs_vgg13    far feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
    pxs_vgg13    far feature     multilabel/within_eps_majority        Always-NNGuide      -2.8605  0.0000
    pxs_vgg13    far feature     multilabel/within_eps_majority           Always-fDBD      -4.5369  0.0000
    pxs_vgg13    far feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13    far feature     multilabel/within_eps_majority            Random-CSF     -12.7687  0.0000
    pxs_vgg13    far feature         multilabel/within_eps_rank            Random-CSF      -9.6152  0.0000
    pxs_vgg13    far feature          multilabel/within_eps_raw            Random-CSF     -10.3387  0.0000
    pxs_vgg13    far feature                         regression            Random-CSF      -6.9829  0.0000
    pxs_vgg13    far    head                         regression         Always-Energy      -0.6295  0.0000
    pxs_vgg13    far    head                         regression            Random-CSF      -3.2904  0.0000
    pxs_vgg13    mid     all                  multilabel/clique            Always-MLS      -1.7768  0.0000
    pxs_vgg13    mid     all                  multilabel/clique            Always-MSR      -4.7830  0.0000
    pxs_vgg13    mid     all                  multilabel/clique        Always-NNGuide      -0.9957  0.0000
    pxs_vgg13    mid     all                  multilabel/clique           Always-fDBD      -2.1093  0.0000
    pxs_vgg13    mid     all                  multilabel/clique            Random-CSF     -10.1955  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority            Always-CTM      -0.5120  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority         Always-Energy      -2.4659  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority            Always-MLS      -3.1591  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority            Always-MSR      -5.8354  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority        Always-NNGuide      -2.2661  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority           Always-fDBD      -5.2067  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.5120  0.0000
    pxs_vgg13    mid     all     multilabel/within_eps_majority            Random-CSF     -12.9547  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank         Always-Energy      -2.2906  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank            Always-MLS      -2.9198  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank            Always-MSR      -5.4355  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank        Always-NNGuide      -1.9070  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank           Always-fDBD      -4.5227  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13    mid     all         multilabel/within_eps_rank            Random-CSF     -12.4562  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw         Always-Energy      -1.9132  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw            Always-MLS      -2.7979  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw            Always-MSR      -5.5241  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw        Always-NNGuide      -2.0137  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw           Always-fDBD      -4.6398  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13    mid     all          multilabel/within_eps_raw            Random-CSF     -12.5007  0.0000
    pxs_vgg13    mid     all                         regression            Always-MSR      -2.5668  0.0000
    pxs_vgg13    mid     all                         regression           Always-fDBD      -1.8711  0.0000
    pxs_vgg13    mid     all                         regression            Random-CSF      -8.7738  0.0000
    pxs_vgg13    mid     all                  multilabel/clique         Always-Energy      -0.1306  0.0002
    pxs_vgg13    mid feature                  multilabel/clique            Random-CSF      -8.2604  0.0000
    pxs_vgg13    mid feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
    pxs_vgg13    mid feature     multilabel/within_eps_majority        Always-NNGuide      -0.9941  0.0000
    pxs_vgg13    mid feature     multilabel/within_eps_majority           Always-fDBD      -4.8517  0.0000
    pxs_vgg13    mid feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13    mid feature     multilabel/within_eps_majority            Random-CSF     -14.8858  0.0000
    pxs_vgg13    mid feature         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
    pxs_vgg13    mid feature         multilabel/within_eps_rank        Always-NNGuide      -0.9212  0.0000
    pxs_vgg13    mid feature         multilabel/within_eps_rank           Always-fDBD      -3.3662  0.0000
    pxs_vgg13    mid feature         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13    mid feature         multilabel/within_eps_rank            Random-CSF     -13.8760  0.0000
    pxs_vgg13    mid feature          multilabel/within_eps_raw        Always-NNGuide      -0.6776  0.0000
    pxs_vgg13    mid feature          multilabel/within_eps_raw           Always-fDBD      -3.1328  0.0000
    pxs_vgg13    mid feature          multilabel/within_eps_raw            Random-CSF     -12.8521  0.0000
    pxs_vgg13    mid feature                         regression           Always-fDBD      -1.8711  0.0000
    pxs_vgg13    mid feature                         regression            Random-CSF     -11.7410  0.0000
    pxs_vgg13    mid    head                         regression            Always-MSR      -1.2672  0.0000
    pxs_vgg13    mid    head                         regression            Random-CSF      -3.9040  0.0000
    pxs_vgg13   near     all                  multilabel/clique            Always-CTM       0.0000  0.0000
    pxs_vgg13   near     all                  multilabel/clique         Always-Energy      -3.0532  0.0000
    pxs_vgg13   near     all                  multilabel/clique            Always-MLS      -2.3802  0.0000
    pxs_vgg13   near     all                  multilabel/clique            Always-MSR      -4.2846  0.0000
    pxs_vgg13   near     all                  multilabel/clique        Always-NNGuide      -1.8199  0.0000
    pxs_vgg13   near     all                  multilabel/clique           Always-fDBD      -3.8644  0.0000
    pxs_vgg13   near     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near     all                  multilabel/clique            Random-CSF     -14.3045  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority         Always-Energy      -3.3562  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority            Always-MLS      -2.7719  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority            Always-MSR      -4.6580  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority        Always-NNGuide      -2.1332  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority           Always-fDBD      -4.6289  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near     all     multilabel/within_eps_majority            Random-CSF     -16.4367  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank         Always-Energy      -2.8790  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank            Always-MLS      -2.1840  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank            Always-MSR      -3.9306  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank        Always-NNGuide      -1.7768  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank           Always-fDBD      -3.8279  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near     all         multilabel/within_eps_rank            Random-CSF     -14.2236  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw         Always-Energy      -2.9683  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw            Always-MLS      -2.3584  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw            Always-MSR      -4.2846  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw        Always-NNGuide      -1.8899  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw           Always-fDBD      -3.9748  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near     all          multilabel/within_eps_raw            Random-CSF     -14.2783  0.0000
    pxs_vgg13   near     all                         regression         Always-Energy      -1.4983  0.0000
    pxs_vgg13   near     all                         regression            Always-MLS      -0.3276  0.0000
    pxs_vgg13   near     all                         regression            Always-MSR      -1.6558  0.0000
    pxs_vgg13   near     all                         regression           Always-fDBD      -1.2722  0.0000
    pxs_vgg13   near     all                         regression            Random-CSF     -12.6935  0.0000
    pxs_vgg13   near feature                  multilabel/clique        Always-NNGuide      -1.3219  0.0000
    pxs_vgg13   near feature                  multilabel/clique           Always-fDBD      -2.6797  0.0000
    pxs_vgg13   near feature                  multilabel/clique            Random-CSF     -21.4378  0.0000
    pxs_vgg13   near feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
    pxs_vgg13   near feature     multilabel/within_eps_majority        Always-NNGuide      -1.5314  0.0000
    pxs_vgg13   near feature     multilabel/within_eps_majority           Always-fDBD      -4.2034  0.0000
    pxs_vgg13   near feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near feature     multilabel/within_eps_majority            Random-CSF     -22.6737  0.0000
    pxs_vgg13   near feature         multilabel/within_eps_rank        Always-NNGuide      -1.0385  0.0000
    pxs_vgg13   near feature         multilabel/within_eps_rank           Always-fDBD      -2.8398  0.0000
    pxs_vgg13   near feature         multilabel/within_eps_rank            Random-CSF     -21.2978  0.0000
    pxs_vgg13   near feature          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
    pxs_vgg13   near feature          multilabel/within_eps_raw        Always-NNGuide      -1.4366  0.0000
    pxs_vgg13   near feature          multilabel/within_eps_raw           Always-fDBD      -3.5936  0.0000
    pxs_vgg13   near feature          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
    pxs_vgg13   near feature          multilabel/within_eps_raw            Random-CSF     -21.7002  0.0000
    pxs_vgg13   near feature    multilabel/within_eps_unanimous            Random-CSF     -13.7361  0.0000
    pxs_vgg13   near feature                         regression           Always-fDBD      -1.2722  0.0000
    pxs_vgg13   near feature                         regression            Random-CSF     -20.6837  0.0000
    pxs_vgg13   near    head                         regression         Always-Energy      -1.3676  0.0000
    pxs_vgg13   near    head                         regression            Always-MSR      -0.4389  0.0000
    pxs_vgg13   near    head                         regression            Random-CSF      -6.2854  0.0000
 single_vgg13    far     all                  multilabel/clique         Always-Energy      -1.7273  0.0000
 single_vgg13    far     all                  multilabel/clique            Always-MLS      -2.2230  0.0000
 single_vgg13    far     all                  multilabel/clique            Always-MSR      -3.4542  0.0000
 single_vgg13    far     all                  multilabel/clique        Always-NNGuide      -3.2245  0.0000
 single_vgg13    far     all                  multilabel/clique           Always-fDBD      -3.4534  0.0000
 single_vgg13    far     all                  multilabel/clique            Random-CSF      -9.9301  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority            Always-CTM      -0.7775  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority         Always-Energy      -3.1893  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority            Always-MLS      -3.0630  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority            Always-MSR      -3.2345  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority        Always-NNGuide      -3.5073  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority           Always-fDBD      -4.8703  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -0.7775  0.0000
 single_vgg13    far     all     multilabel/within_eps_majority            Random-CSF     -11.6523  0.0000
 single_vgg13    far     all         multilabel/within_eps_rank            Random-CSF      -9.5985  0.0000
 single_vgg13    far     all          multilabel/within_eps_raw            Always-MSR      -2.7025  0.0000
 single_vgg13    far     all          multilabel/within_eps_raw        Always-NNGuide      -3.1474  0.0000
 single_vgg13    far     all          multilabel/within_eps_raw           Always-fDBD      -2.9113  0.0000
 single_vgg13    far     all          multilabel/within_eps_raw            Random-CSF     -11.2174  0.0000
 single_vgg13    far     all                         regression            Always-MSR      -1.3051  0.0000
 single_vgg13    far     all                         regression           Always-fDBD      -1.2319  0.0000
 single_vgg13    far     all                         regression            Random-CSF      -5.4355  0.0000
 single_vgg13    far     all          multilabel/within_eps_raw            Always-MLS      -2.6497  0.0001
 single_vgg13    far     all          multilabel/within_eps_raw         Always-Energy      -2.8410  0.0003
 single_vgg13    far     all                         regression         Always-Energy      -1.0590  0.0059
 single_vgg13    far     all         multilabel/within_eps_rank           Always-fDBD      -2.2677  0.0225
 single_vgg13    far     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0301
 single_vgg13    far     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0301
 single_vgg13    far feature     multilabel/within_eps_majority        Always-NNGuide      -2.7402  0.0000
 single_vgg13    far feature     multilabel/within_eps_majority           Always-fDBD      -3.8855  0.0000
 single_vgg13    far feature     multilabel/within_eps_majority            Random-CSF     -13.4567  0.0000
 single_vgg13    far feature          multilabel/within_eps_raw            Random-CSF     -10.5246  0.0000
 single_vgg13    far feature                         regression           Always-fDBD      -1.2319  0.0000
 single_vgg13    far feature                         regression            Random-CSF      -8.1269  0.0000
 single_vgg13    far feature         multilabel/within_eps_rank            Random-CSF      -6.9890  0.0100
 single_vgg13    far feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0197
 single_vgg13    far feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0197
 single_vgg13    far    head                         regression         Always-Energy      -0.3522  0.0000
 single_vgg13    far    head                         regression            Random-CSF      -3.3566  0.0000
 single_vgg13    far    head                         regression            Always-MSR      -0.0558  0.0367
 single_vgg13    mid     all                  multilabel/clique            Always-MLS      -1.7508  0.0000
 single_vgg13    mid     all                  multilabel/clique            Always-MSR      -5.0095  0.0000
 single_vgg13    mid     all                  multilabel/clique        Always-NNGuide      -1.4818  0.0000
 single_vgg13    mid     all                  multilabel/clique           Always-fDBD      -2.1915  0.0000
 single_vgg13    mid     all                  multilabel/clique            Random-CSF     -10.0518  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority            Always-CTM      -1.1188  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority         Always-Energy      -3.3656  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority            Always-MLS      -3.5909  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority            Always-MSR      -6.4326  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority        Always-NNGuide      -2.7612  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority           Always-fDBD      -5.6226  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority Oracle-on-train (CTM)      -1.1188  0.0000
 single_vgg13    mid     all     multilabel/within_eps_majority            Random-CSF     -13.4838  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank         Always-Energy      -2.2906  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank            Always-MLS      -3.0767  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank            Always-MSR      -5.7080  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank        Always-NNGuide      -2.2525  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank           Always-fDBD      -3.6743  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13    mid     all         multilabel/within_eps_rank            Random-CSF     -12.1621  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw         Always-Energy      -2.6055  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw            Always-MLS      -3.1767  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw            Always-MSR      -5.8376  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw        Always-NNGuide      -2.3250  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw           Always-fDBD      -3.7087  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13    mid     all          multilabel/within_eps_raw            Random-CSF     -12.1621  0.0000
 single_vgg13    mid     all                         regression         Always-Energy      -0.4291  0.0000
 single_vgg13    mid     all                         regression            Always-MLS      -0.5621  0.0000
 single_vgg13    mid     all                         regression            Always-MSR      -2.8639  0.0000
 single_vgg13    mid     all                         regression           Always-fDBD      -2.4940  0.0000
 single_vgg13    mid     all                         regression            Random-CSF      -8.2724  0.0000
 single_vgg13    mid     all                  multilabel/clique         Always-Energy       0.0000  0.0058
 single_vgg13    mid feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
 single_vgg13    mid feature     multilabel/within_eps_majority        Always-NNGuide      -1.3935  0.0000
 single_vgg13    mid feature     multilabel/within_eps_majority           Always-fDBD      -5.1529  0.0000
 single_vgg13    mid feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13    mid feature     multilabel/within_eps_majority            Random-CSF     -15.7509  0.0000
 single_vgg13    mid feature         multilabel/within_eps_rank        Always-NNGuide      -1.2521  0.0000
 single_vgg13    mid feature         multilabel/within_eps_rank           Always-fDBD      -2.9710  0.0000
 single_vgg13    mid feature         multilabel/within_eps_rank            Random-CSF     -13.6320  0.0000
 single_vgg13    mid feature          multilabel/within_eps_raw           Always-fDBD      -2.4889  0.0000
 single_vgg13    mid feature          multilabel/within_eps_raw            Random-CSF     -13.3828  0.0000
 single_vgg13    mid feature                         regression           Always-fDBD      -2.4940  0.0000
 single_vgg13    mid feature                         regression            Random-CSF     -11.5292  0.0000
 single_vgg13    mid feature          multilabel/within_eps_raw        Always-NNGuide      -1.1067  0.0002
 single_vgg13    mid    head                         regression            Always-MSR      -1.7844  0.0000
 single_vgg13    mid    head                         regression            Random-CSF      -4.2073  0.0000
 single_vgg13   near     all                  multilabel/clique            Always-CTM       0.0000  0.0000
 single_vgg13   near     all                  multilabel/clique         Always-Energy      -3.5626  0.0000
 single_vgg13   near     all                  multilabel/clique            Always-MLS      -2.7031  0.0000
 single_vgg13   near     all                  multilabel/clique            Always-MSR      -4.6944  0.0000
 single_vgg13   near     all                  multilabel/clique        Always-NNGuide      -2.1299  0.0000
 single_vgg13   near     all                  multilabel/clique           Always-fDBD      -4.3316  0.0000
 single_vgg13   near     all                  multilabel/clique Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near     all                  multilabel/clique            Random-CSF     -15.0946  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority         Always-Energy      -3.8845  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority            Always-MLS      -3.2496  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority            Always-MSR      -5.0538  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority        Always-NNGuide      -2.3768  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority           Always-fDBD      -4.6410  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near     all     multilabel/within_eps_majority            Random-CSF     -17.1764  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank            Always-CTM       0.0000  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank         Always-Energy      -2.7490  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank            Always-MLS      -2.0286  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank            Always-MSR      -3.8132  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank        Always-NNGuide      -1.7106  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank           Always-fDBD      -3.9359  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near     all         multilabel/within_eps_rank            Random-CSF     -14.4826  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw         Always-Energy      -3.2866  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw            Always-MLS      -2.5980  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw            Always-MSR      -4.6449  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw        Always-NNGuide      -2.0438  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw           Always-fDBD      -4.4361  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near     all          multilabel/within_eps_raw            Random-CSF     -15.6549  0.0000
 single_vgg13   near     all                         regression         Always-Energy      -1.4983  0.0000
 single_vgg13   near     all                         regression            Always-MLS      -0.4786  0.0000
 single_vgg13   near     all                         regression            Always-MSR      -2.0104  0.0000
 single_vgg13   near     all                         regression           Always-fDBD      -2.1486  0.0000
 single_vgg13   near     all                         regression            Random-CSF     -12.5626  0.0000
 single_vgg13   near feature                  multilabel/clique            Random-CSF     -20.1409  0.0000
 single_vgg13   near feature     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
 single_vgg13   near feature     multilabel/within_eps_majority        Always-NNGuide      -1.6049  0.0000
 single_vgg13   near feature     multilabel/within_eps_majority           Always-fDBD      -4.0790  0.0000
 single_vgg13   near feature     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near feature     multilabel/within_eps_majority            Random-CSF     -22.8799  0.0000
 single_vgg13   near feature         multilabel/within_eps_rank        Always-NNGuide      -0.8886  0.0000
 single_vgg13   near feature         multilabel/within_eps_rank           Always-fDBD      -2.9774  0.0000
 single_vgg13   near feature         multilabel/within_eps_rank            Random-CSF     -21.9478  0.0000
 single_vgg13   near feature          multilabel/within_eps_raw            Always-CTM       0.0000  0.0000
 single_vgg13   near feature          multilabel/within_eps_raw        Always-NNGuide      -1.3549  0.0000
 single_vgg13   near feature          multilabel/within_eps_raw           Always-fDBD      -3.9491  0.0000
 single_vgg13   near feature          multilabel/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
 single_vgg13   near feature          multilabel/within_eps_raw            Random-CSF     -22.5091  0.0000
 single_vgg13   near feature    multilabel/within_eps_unanimous            Random-CSF       5.8050  0.0000
 single_vgg13   near feature                         regression           Always-fDBD      -2.1486  0.0000
 single_vgg13   near feature                         regression            Random-CSF     -20.8948  0.0000
 single_vgg13   near    head                         regression         Always-Energy      -1.1157  0.0000
 single_vgg13   near    head                         regression            Always-MSR      -0.5820  0.0000
 single_vgg13   near    head                         regression            Random-CSF      -6.1212  0.0000
        xarch    far     all                  multilabel/clique         Always-Energy      -3.8371  0.0000
        xarch    far     all                  multilabel/clique            Always-MLS      -3.4141  0.0000
        xarch    far     all                  multilabel/clique            Always-MSR      -4.0413  0.0000
        xarch    far     all                  multilabel/clique        Always-NNGuide      -1.2790  0.0000
        xarch    far     all                  multilabel/clique           Always-fDBD      -3.0487  0.0000
        xarch    far     all                  multilabel/clique            Random-CSF      -8.8469  0.0000
        xarch    far     all     multilabel/within_eps_majority         Always-Energy      -3.2407  0.0000
        xarch    far     all     multilabel/within_eps_majority            Always-MLS      -2.8794  0.0000
        xarch    far     all     multilabel/within_eps_majority            Always-MSR      -3.2532  0.0000
        xarch    far     all     multilabel/within_eps_majority            Random-CSF      -8.1353  0.0000
        xarch    far     all per_csf_binary/within_eps_majority         Always-Energy      -4.8246  0.0000
        xarch    far     all per_csf_binary/within_eps_majority            Always-MLS      -4.2734  0.0000
        xarch    far     all per_csf_binary/within_eps_majority            Always-MSR      -5.4707  0.0000
        xarch    far     all per_csf_binary/within_eps_majority        Always-NNGuide      -2.1561  0.0000
        xarch    far     all per_csf_binary/within_eps_majority           Always-fDBD      -3.1936  0.0000
        xarch    far     all per_csf_binary/within_eps_majority            Random-CSF      -8.6405  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank         Always-Energy      -4.8246  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank            Always-MLS      -4.1961  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank            Always-MSR      -5.6192  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank        Always-NNGuide      -2.2704  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank           Always-fDBD      -3.9046  0.0000
        xarch    far     all     per_csf_binary/within_eps_rank            Random-CSF      -9.6636  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw         Always-Energy      -4.8246  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw            Always-MLS      -4.1961  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw            Always-MSR      -5.6192  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw        Always-NNGuide      -2.2704  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw           Always-fDBD      -3.9046  0.0000
        xarch    far     all      per_csf_binary/within_eps_raw            Random-CSF      -9.6636  0.0000
        xarch    far     all                         regression         Always-Energy      -1.8477  0.0000
        xarch    far     all                         regression            Always-MLS      -1.7422  0.0000
        xarch    far     all                         regression            Always-MSR      -2.2136  0.0000
        xarch    far     all                         regression            Random-CSF      -6.6817  0.0000
        xarch    far     all     multilabel/within_eps_majority           Always-fDBD      -1.8402  0.0004
        xarch    far     all              per_csf_binary/clique            Random-CSF      -7.9594  0.0009
        xarch    far     all     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0024
        xarch    far     all     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0024
        xarch    far     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0051
        xarch    far     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0051
        xarch    far     all              per_csf_binary/clique            Always-MSR      -4.2379  0.0070
        xarch    far     all              per_csf_binary/clique            Always-MLS      -3.8033  0.0111
        xarch    far     all     multilabel/within_eps_majority        Always-NNGuide      -0.9241  0.0153
        xarch    far     all per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0154
        xarch    far     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0154
        xarch    far     all              per_csf_binary/clique         Always-Energy      -4.0284  0.0491
        xarch    far feature     multilabel/within_eps_majority            Random-CSF      -7.1017  0.0000
        xarch    far feature per_csf_binary/within_eps_majority        Always-NNGuide      -2.1561  0.0000
        xarch    far feature per_csf_binary/within_eps_majority           Always-fDBD      -3.1936  0.0000
        xarch    far feature per_csf_binary/within_eps_majority            Random-CSF      -7.0157  0.0000
        xarch    far feature     per_csf_binary/within_eps_rank        Always-NNGuide      -2.1561  0.0000
        xarch    far feature     per_csf_binary/within_eps_rank           Always-fDBD      -3.8442  0.0000
        xarch    far feature     per_csf_binary/within_eps_rank            Random-CSF      -7.3213  0.0000
        xarch    far feature      per_csf_binary/within_eps_raw        Always-NNGuide      -2.1561  0.0000
        xarch    far feature      per_csf_binary/within_eps_raw           Always-fDBD      -3.8442  0.0000
        xarch    far feature      per_csf_binary/within_eps_raw            Random-CSF      -7.2107  0.0000
        xarch    far feature                         regression            Random-CSF      -6.7787  0.0000
        xarch    far feature     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0216
        xarch    far feature     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0216
        xarch    far feature      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0466
        xarch    far feature      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0466
        xarch    far    head                         regression            Random-CSF      -4.3912  0.0000
        xarch    mid     all                  multilabel/clique            Always-MLS      -1.7802  0.0000
        xarch    mid     all                  multilabel/clique            Always-MSR      -4.2763  0.0000
        xarch    mid     all                  multilabel/clique           Always-fDBD      -3.9944  0.0000
        xarch    mid     all                  multilabel/clique            Random-CSF     -10.0195  0.0000
        xarch    mid     all     multilabel/within_eps_majority            Always-CTM       0.0000  0.0000
        xarch    mid     all     multilabel/within_eps_majority         Always-Energy      -2.4391  0.0000
        xarch    mid     all     multilabel/within_eps_majority            Always-MLS      -2.3877  0.0000
        xarch    mid     all     multilabel/within_eps_majority            Always-MSR      -5.2011  0.0000
        xarch    mid     all     multilabel/within_eps_majority        Always-NNGuide      -1.1880  0.0000
        xarch    mid     all     multilabel/within_eps_majority           Always-fDBD      -5.2924  0.0000
        xarch    mid     all     multilabel/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid     all     multilabel/within_eps_majority            Random-CSF     -12.7417  0.0000
        xarch    mid     all         multilabel/within_eps_rank         Always-Energy      -1.8746  0.0000
        xarch    mid     all         multilabel/within_eps_rank            Always-MLS      -2.0204  0.0000
        xarch    mid     all         multilabel/within_eps_rank            Always-MSR      -4.3820  0.0000
        xarch    mid     all         multilabel/within_eps_rank           Always-fDBD      -4.3119  0.0000
        xarch    mid     all         multilabel/within_eps_rank            Random-CSF     -10.3741  0.0000
        xarch    mid     all          multilabel/within_eps_raw         Always-Energy      -1.9410  0.0000
        xarch    mid     all          multilabel/within_eps_raw            Always-MLS      -2.0427  0.0000
        xarch    mid     all          multilabel/within_eps_raw            Always-MSR      -4.7127  0.0000
        xarch    mid     all          multilabel/within_eps_raw           Always-fDBD      -4.4696  0.0000
        xarch    mid     all          multilabel/within_eps_raw            Random-CSF     -10.4186  0.0000
        xarch    mid     all              per_csf_binary/clique            Random-CSF      -9.3266  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority         Always-Energy      -2.4125  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority            Always-MLS      -2.6149  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority            Always-MSR      -4.8260  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority        Always-NNGuide      -1.4893  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority           Always-fDBD      -4.4044  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid     all per_csf_binary/within_eps_majority            Random-CSF     -10.7103  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank         Always-Energy      -2.0521  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank            Always-MLS      -2.0916  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank            Always-MSR      -4.5512  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank        Always-NNGuide      -1.1659  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank           Always-fDBD      -4.9851  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid     all     per_csf_binary/within_eps_rank            Random-CSF     -12.0981  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw         Always-Energy      -2.0521  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw            Always-MLS      -2.0916  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw            Always-MSR      -4.5512  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw           Always-fDBD      -4.8078  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid     all      per_csf_binary/within_eps_raw            Random-CSF     -12.3301  0.0000
        xarch    mid     all                         regression         Always-Energy      -0.9109  0.0000
        xarch    mid     all                         regression            Always-MLS      -1.3331  0.0000
        xarch    mid     all                         regression            Always-MSR      -3.4102  0.0000
        xarch    mid     all                         regression           Always-fDBD      -2.4010  0.0000
        xarch    mid     all                         regression            Random-CSF      -9.4384  0.0000
        xarch    mid     all                  multilabel/clique         Always-Energy      -1.1658  0.0002
        xarch    mid     all      per_csf_binary/within_eps_raw        Always-NNGuide      -1.1659  0.0002
        xarch    mid     all          multilabel/within_eps_raw        Always-NNGuide      -0.5732  0.0027
        xarch    mid     all              per_csf_binary/clique            Always-MSR      -4.0862  0.0070
        xarch    mid     all         multilabel/within_eps_rank        Always-NNGuide      -0.6323  0.0212
        xarch    mid feature     multilabel/within_eps_majority           Always-fDBD      -4.8813  0.0000
        xarch    mid feature     multilabel/within_eps_majority            Random-CSF     -10.5634  0.0000
        xarch    mid feature         multilabel/within_eps_rank            Random-CSF      -7.4139  0.0000
        xarch    mid feature per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0000
        xarch    mid feature per_csf_binary/within_eps_majority        Always-NNGuide      -1.2028  0.0000
        xarch    mid feature per_csf_binary/within_eps_majority           Always-fDBD      -4.2825  0.0000
        xarch    mid feature per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid feature per_csf_binary/within_eps_majority            Random-CSF     -10.7285  0.0000
        xarch    mid feature     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0000
        xarch    mid feature     per_csf_binary/within_eps_rank           Always-fDBD      -4.7301  0.0000
        xarch    mid feature     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid feature     per_csf_binary/within_eps_rank            Random-CSF     -10.7786  0.0000
        xarch    mid feature      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0000
        xarch    mid feature      per_csf_binary/within_eps_raw           Always-fDBD      -4.8078  0.0000
        xarch    mid feature      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0000
        xarch    mid feature      per_csf_binary/within_eps_raw            Random-CSF     -10.7786  0.0000
        xarch    mid feature                         regression           Always-fDBD      -2.4010  0.0000
        xarch    mid feature                         regression            Random-CSF     -12.1271  0.0000
        xarch    mid feature     per_csf_binary/within_eps_rank        Always-NNGuide      -1.1659  0.0021
        xarch    mid feature      per_csf_binary/within_eps_raw        Always-NNGuide      -1.1510  0.0033
        xarch    mid feature         multilabel/within_eps_rank           Always-fDBD      -3.9558  0.0041
        xarch    mid    head                         regression            Always-MSR      -1.3451  0.0000
        xarch    mid    head                         regression            Random-CSF      -5.8402  0.0000
        xarch   near     all                  multilabel/clique         Always-Energy      -3.2198  0.0000
        xarch   near     all                  multilabel/clique            Always-MLS      -2.4363  0.0000
        xarch   near     all                  multilabel/clique            Always-MSR      -3.8880  0.0000
        xarch   near     all                  multilabel/clique        Always-NNGuide      -1.8301  0.0000
        xarch   near     all                  multilabel/clique           Always-fDBD      -3.9687  0.0000
        xarch   near     all                  multilabel/clique            Random-CSF     -18.5388  0.0000
        xarch   near     all     multilabel/within_eps_majority         Always-Energy      -3.5110  0.0000
        xarch   near     all     multilabel/within_eps_majority            Always-MLS      -2.6348  0.0000
        xarch   near     all     multilabel/within_eps_majority            Always-MSR      -3.8880  0.0000
        xarch   near     all     multilabel/within_eps_majority        Always-NNGuide      -1.9452  0.0000
        xarch   near     all     multilabel/within_eps_majority           Always-fDBD      -3.5846  0.0000
        xarch   near     all     multilabel/within_eps_majority            Random-CSF     -19.9743  0.0000
        xarch   near     all          multilabel/within_eps_raw            Random-CSF     -17.8694  0.0000
        xarch   near     all              per_csf_binary/clique            Random-CSF     -15.5475  0.0000
        xarch   near     all per_csf_binary/within_eps_majority         Always-Energy      -4.3423  0.0000
        xarch   near     all per_csf_binary/within_eps_majority            Always-MLS      -3.0036  0.0000
        xarch   near     all per_csf_binary/within_eps_majority            Always-MSR      -4.2300  0.0000
        xarch   near     all per_csf_binary/within_eps_majority        Always-NNGuide      -2.3309  0.0000
        xarch   near     all per_csf_binary/within_eps_majority           Always-fDBD      -4.2764  0.0000
        xarch   near     all per_csf_binary/within_eps_majority            Random-CSF     -19.9384  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank         Always-Energy      -4.2962  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank            Always-MLS      -3.0036  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank            Always-MSR      -4.1819  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank        Always-NNGuide      -2.2520  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank           Always-fDBD      -4.2764  0.0000
        xarch   near     all     per_csf_binary/within_eps_rank            Random-CSF     -19.9384  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw         Always-Energy      -4.1653  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw            Always-MLS      -2.8851  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw            Always-MSR      -4.1256  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw        Always-NNGuide      -2.1612  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw           Always-fDBD      -4.1683  0.0000
        xarch   near     all      per_csf_binary/within_eps_raw            Random-CSF     -19.9384  0.0000
        xarch   near     all                         regression         Always-Energy      -1.1129  0.0000
        xarch   near     all                         regression            Always-MLS      -0.7279  0.0000
        xarch   near     all                         regression            Always-MSR      -1.5665  0.0000
        xarch   near     all                         regression            Random-CSF     -15.0383  0.0000
        xarch   near     all              per_csf_binary/clique           Always-fDBD      -2.2492  0.0001
        xarch   near     all                         regression           Always-fDBD      -1.4281  0.0001
        xarch   near     all         multilabel/within_eps_rank            Random-CSF     -12.7731  0.0002
        xarch   near     all per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0002
        xarch   near     all per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0002
        xarch   near     all     per_csf_binary/within_eps_rank            Always-CTM       0.0000  0.0046
        xarch   near     all     per_csf_binary/within_eps_rank Oracle-on-train (CTM)       0.0000  0.0046
        xarch   near     all      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0046
        xarch   near     all      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0046
        xarch   near     all          multilabel/within_eps_raw           Always-fDBD      -2.3212  0.0107
        xarch   near     all          multilabel/within_eps_raw            Always-MSR      -2.9480  0.0370
        xarch   near     all              per_csf_binary/clique            Always-MSR      -3.0852  0.0450
        xarch   near feature                  multilabel/clique            Random-CSF     -19.3867  0.0000
        xarch   near feature     multilabel/within_eps_majority           Always-fDBD      -2.7708  0.0000
        xarch   near feature     multilabel/within_eps_majority            Random-CSF     -23.7562  0.0000
        xarch   near feature          multilabel/within_eps_raw            Random-CSF     -15.0926  0.0000
        xarch   near feature per_csf_binary/within_eps_majority        Always-NNGuide      -2.0890  0.0000
        xarch   near feature per_csf_binary/within_eps_majority           Always-fDBD      -4.1302  0.0000
        xarch   near feature per_csf_binary/within_eps_majority            Random-CSF     -26.9122  0.0000
        xarch   near feature     per_csf_binary/within_eps_rank        Always-NNGuide      -2.0890  0.0000
        xarch   near feature     per_csf_binary/within_eps_rank           Always-fDBD      -4.0559  0.0000
        xarch   near feature     per_csf_binary/within_eps_rank            Random-CSF     -26.9122  0.0000
        xarch   near feature      per_csf_binary/within_eps_raw        Always-NNGuide      -2.0890  0.0000
        xarch   near feature      per_csf_binary/within_eps_raw           Always-fDBD      -4.1683  0.0000
        xarch   near feature      per_csf_binary/within_eps_raw            Random-CSF     -26.9122  0.0000
        xarch   near feature                         regression            Random-CSF     -22.4964  0.0000
        xarch   near feature              per_csf_binary/clique            Random-CSF     -10.5347  0.0001
        xarch   near feature                         regression           Always-fDBD      -1.4281  0.0001
        xarch   near feature         multilabel/within_eps_rank            Random-CSF      -9.4907  0.0005
        xarch   near feature     multilabel/within_eps_majority        Always-NNGuide      -1.3477  0.0095
        xarch   near feature                  multilabel/clique           Always-fDBD      -2.1430  0.0099
        xarch   near feature      per_csf_binary/within_eps_raw            Always-CTM       0.0000  0.0118
        xarch   near feature      per_csf_binary/within_eps_raw Oracle-on-train (CTM)       0.0000  0.0118
        xarch   near feature per_csf_binary/within_eps_majority            Always-CTM       0.0000  0.0167
        xarch   near feature per_csf_binary/within_eps_majority Oracle-on-train (CTM)       0.0000  0.0167
        xarch   near    head                         regression         Always-Energy      -0.9019  0.0000
        xarch   near    head                         regression            Random-CSF      -7.8567  0.0000
```

## Per-(split, predictor) tally of baselines beaten

```
        split                          predictor  n_baseline_wins
   lodo_vgg13     multilabel/within_eps_majority               26
   lodo_vgg13                  multilabel/clique               18
   lodo_vgg13                         regression               18
   lodo_vgg13         multilabel/within_eps_rank               10
   lodo_vgg13          multilabel/within_eps_raw               10
   lodo_vgg13 per_csf_binary/within_eps_majority                8
         lopo     multilabel/within_eps_majority               39
         lopo per_csf_binary/within_eps_majority               39
         lopo          multilabel/within_eps_raw               37
         lopo      per_csf_binary/within_eps_raw               37
         lopo                  multilabel/clique               32
         lopo     per_csf_binary/within_eps_rank               31
         lopo              per_csf_binary/clique               30
         lopo         multilabel/within_eps_rank               29
         lopo                         regression               28
lopo_cnn_only     multilabel/within_eps_majority               39
lopo_cnn_only          multilabel/within_eps_raw               33
lopo_cnn_only                  multilabel/clique               31
lopo_cnn_only                         regression               29
lopo_cnn_only         multilabel/within_eps_rank               25
    pxs_vgg13     multilabel/within_eps_majority               39
    pxs_vgg13         multilabel/within_eps_rank               27
    pxs_vgg13          multilabel/within_eps_raw               26
    pxs_vgg13                  multilabel/clique               25
    pxs_vgg13                         regression               21
    pxs_vgg13    multilabel/within_eps_unanimous                1
 single_vgg13     multilabel/within_eps_majority               39
 single_vgg13          multilabel/within_eps_raw               33
 single_vgg13                         regression               28
 single_vgg13         multilabel/within_eps_rank               25
 single_vgg13                  multilabel/clique               21
 single_vgg13    multilabel/within_eps_unanimous                1
        xarch      per_csf_binary/within_eps_raw               39
        xarch per_csf_binary/within_eps_majority               37
        xarch     per_csf_binary/within_eps_rank               37
        xarch     multilabel/within_eps_majority               26
        xarch                         regression               24
        xarch                  multilabel/clique               19
        xarch         multilabel/within_eps_rank               10
        xarch          multilabel/within_eps_raw               10
        xarch              per_csf_binary/clique               10
```
