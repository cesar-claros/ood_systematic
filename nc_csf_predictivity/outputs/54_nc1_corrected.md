# Corrected NC1 for every pool

**Source:** `nc_csf_predictivity/evaluation/nc1_corrected_panel.py`. Corrected NC1 = Tr(Sigma_W Sigma_B^+)/C with Sigma_W averaged over N (Papyan); legacy = the paper's value (corrected / C under the legacy pseudoinverse). ratio_to_c = corrected / (legacy x C): 1 when only the normalization differs.

Gates GP and GN passed.

```
      pool        source  models    C  legacy_median  corrected_median  corrected_min  corrected_max  ratio_to_c_median  ratio_to_c_max                            convention
     VGG13       cifar10      60   10         0.0094            0.0949         0.0593         0.1774             0.9970          1.0309                      pilot0 remeasure
     VGG13      cifar100      70  100         0.0085            0.8640         0.1275        25.3854             1.0021         16.3134                      pilot0 remeasure
     VGG13 supercifar100      90   20         0.0182            0.3433         0.1466       250.5125             0.9530          9.3313                      pilot0 remeasure
     VGG13  tinyimagenet      60  200         0.0051            2.6083         0.3647        21.6275             1.2950         25.4920                      pilot0 remeasure
  ResNet18       cifar10      12   10         0.0055            0.0544         0.0333         0.0877             0.9951          1.0161               D-R4 aug-view remeasure
  ResNet18      cifar100      14  100         0.0039            0.4151         0.1012        12.7581             1.0034         17.9282               D-R4 aug-view remeasure
  ResNet18 supercifar100      18   20         0.0059            0.1137         0.0603         0.6530             0.9592          0.9859               D-R4 aug-view remeasure
  ResNet18  tinyimagenet      12  200         0.0051            2.6153         0.3746        18.0609             1.2993         25.2117               D-R4 aug-view remeasure
       ViT       cifar10      10   10         0.0108            0.1078         0.0954         0.1321             1.0000          1.0000 legacy x C (approximate, legacy pinv)
       ViT      cifar100      10  100         0.0059            0.5911         0.5529         3.8591             1.0000          1.0000 legacy x C (approximate, legacy pinv)
       ViT supercifar100      10   20         0.0164            0.3283         0.2934         0.3525             1.0000          1.0000 legacy x C (approximate, legacy pinv)
       ViT  tinyimagenet      10  200         0.0520           10.4039         6.8004        14.1718             1.0000          1.0000 legacy x C (approximate, legacy pinv)
SSL probes       cifar10      10   10         0.0273            0.2732         0.1682         0.3790             1.0000          1.0000                    legacy x C (exact)
SSL probes      cifar100      10  100         0.0467            4.6692         1.3583         8.1098             1.0000          1.0000                    legacy x C (exact)
SSL probes supercifar100      10   20         0.0688            1.3769         0.5420         2.2725             1.0000          1.0000                    legacy x C (exact)
SSL probes  tinyimagenet      10  200         0.0537           10.7306         2.1233        19.4671             1.0000          1.0000                    legacy x C (exact)
  ImageNet      imagenet      47 1000         0.0061            6.0542         0.5319       150.7949             1.0000          1.0000                    legacy x C (exact)
```
