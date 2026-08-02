# E-F pilot summary: ASH / ReAct / Mahalanobis++ / NCI

Experiments: 320; rows: 295,970.

## Mean metrics per (arm, CSF, mode)

| arm            | csf      | mode                   |   n |   AUGRC |   AUROC_f |   FPR95 |
|:---------------|:---------|:-----------------------|----:|--------:|----------:|--------:|
| ASH (ash_s@65) | Energy   | iid_test               | 280 | 103.568 |     0.827 |   0.663 |
| ASH (ash_s@65) | Energy   | ood_nsncs_isun         | 280 | 203.653 |     0.881 |   0.509 |
| ASH (ash_s@65) | Energy   | ood_nsncs_lsun_cropped | 280 | 219.409 |     0.88  |   0.521 |
| ASH (ash_s@65) | Energy   | ood_nsncs_lsun_resize  | 280 | 218.926 |     0.882 |   0.508 |
| ASH (ash_s@65) | Energy   | ood_nsncs_places365    | 280 | 235.264 |     0.812 |   0.627 |
| ASH (ash_s@65) | Energy   | ood_nsncs_svhn         | 280 | 348.18  |     0.847 |   0.613 |
| ASH (ash_s@65) | Energy   | ood_nsncs_textures     | 280 | 156.707 |     0.835 |   0.611 |
| ASH (ash_s@65) | Energy   | ood_nsncs_ti           | 220 | 215.759 |     0.88  |   0.498 |
| ASH (ash_s@65) | Energy   | ood_sncs_c10           | 220 | 249.827 |     0.802 |   0.683 |
| ASH (ash_s@65) | Energy   | ood_sncs_c100          | 120 | 199.142 |     0.902 |   0.49  |
| ASH (ash_s@65) | GE       | iid_test               | 280 | 102.624 |     0.834 |   0.652 |
| ASH (ash_s@65) | GE       | ood_nsncs_isun         | 280 | 204.328 |     0.878 |   0.523 |
| ASH (ash_s@65) | GE       | ood_nsncs_lsun_cropped | 280 | 219.925 |     0.878 |   0.532 |
| ASH (ash_s@65) | GE       | ood_nsncs_lsun_resize  | 280 | 219.631 |     0.879 |   0.524 |
| ASH (ash_s@65) | GE       | ood_nsncs_places365    | 280 | 235.421 |     0.811 |   0.634 |
| ASH (ash_s@65) | GE       | ood_nsncs_svhn         | 280 | 348.189 |     0.847 |   0.614 |
| ASH (ash_s@65) | GE       | ood_nsncs_textures     | 280 | 156.877 |     0.835 |   0.614 |
| ASH (ash_s@65) | GE       | ood_nsncs_ti           | 220 | 216.595 |     0.876 |   0.518 |
| ASH (ash_s@65) | GE       | ood_sncs_c10           | 220 | 249.733 |     0.802 |   0.69  |
| ASH (ash_s@65) | GE       | ood_sncs_c100          | 120 | 199.028 |     0.903 |   0.495 |
| ASH (ash_s@65) | GEN      | iid_test               | 280 |  98.162 |     0.856 |   0.604 |
| ASH (ash_s@65) | GEN      | ood_nsncs_isun         | 280 | 205.618 |     0.873 |   0.517 |
| ASH (ash_s@65) | GEN      | ood_nsncs_lsun_cropped | 280 | 219.363 |     0.88  |   0.498 |
| ASH (ash_s@65) | GEN      | ood_nsncs_lsun_resize  | 280 | 221.296 |     0.872 |   0.518 |
| ASH (ash_s@65) | GEN      | ood_nsncs_places365    | 280 | 234.263 |     0.816 |   0.621 |
| ASH (ash_s@65) | GEN      | ood_nsncs_svhn         | 280 | 347.388 |     0.852 |   0.571 |
| ASH (ash_s@65) | GEN      | ood_nsncs_textures     | 280 | 157.255 |     0.833 |   0.613 |
| ASH (ash_s@65) | GEN      | ood_nsncs_ti           | 220 | 218.2   |     0.869 |   0.529 |
| ASH (ash_s@65) | GEN      | ood_sncs_c10           | 220 | 247.864 |     0.81  |   0.635 |
| ASH (ash_s@65) | GEN      | ood_sncs_c100          | 120 | 197.179 |     0.911 |   0.456 |
| ASH (ash_s@65) | GradNorm | iid_test               | 280 | 117.679 |     0.748 |   0.763 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_isun         | 280 | 230.03  |     0.771 |   0.68  |
| ASH (ash_s@65) | GradNorm | ood_nsncs_lsun_cropped | 280 | 247.588 |     0.761 |   0.677 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_lsun_resize  | 280 | 246.247 |     0.765 |   0.685 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_places365    | 280 | 252.346 |     0.739 |   0.738 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_svhn         | 280 | 366.082 |     0.734 |   0.756 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_textures     | 280 | 181.866 |     0.733 |   0.729 |
| ASH (ash_s@65) | GradNorm | ood_nsncs_ti           | 220 | 237.901 |     0.785 |   0.653 |
| ASH (ash_s@65) | GradNorm | ood_sncs_c10           | 220 | 267.794 |     0.723 |   0.794 |
| ASH (ash_s@65) | GradNorm | ood_sncs_c100          | 120 | 230.024 |     0.772 |   0.688 |
| ASH (ash_s@65) | MLS      | iid_test               | 280 | 101.51  |     0.837 |   0.63  |
| ASH (ash_s@65) | MLS      | ood_nsncs_isun         | 280 | 203.745 |     0.881 |   0.492 |
| ASH (ash_s@65) | MLS      | ood_nsncs_lsun_cropped | 280 | 218.698 |     0.883 |   0.491 |
| ASH (ash_s@65) | MLS      | ood_nsncs_lsun_resize  | 280 | 219.179 |     0.881 |   0.493 |
| ASH (ash_s@65) | MLS      | ood_nsncs_places365    | 280 | 234.361 |     0.816 |   0.607 |
| ASH (ash_s@65) | MLS      | ood_nsncs_svhn         | 280 | 347.335 |     0.852 |   0.58  |
| ASH (ash_s@65) | MLS      | ood_nsncs_textures     | 280 | 156.314 |     0.837 |   0.594 |
| ASH (ash_s@65) | MLS      | ood_nsncs_ti           | 220 | 216.101 |     0.878 |   0.49  |
| ASH (ash_s@65) | MLS      | ood_sncs_c10           | 220 | 248.316 |     0.808 |   0.647 |
| ASH (ash_s@65) | MLS      | ood_sncs_c100          | 120 | 197.931 |     0.907 |   0.454 |
| ASH (ash_s@65) | MSR      | iid_test               | 280 |  98.278 |     0.855 |   0.603 |
| ASH (ash_s@65) | MSR      | ood_nsncs_isun         | 280 | 206.16  |     0.871 |   0.513 |
| ASH (ash_s@65) | MSR      | ood_nsncs_lsun_cropped | 280 | 219.908 |     0.878 |   0.499 |
| ASH (ash_s@65) | MSR      | ood_nsncs_lsun_resize  | 280 | 221.85  |     0.87  |   0.515 |
| ASH (ash_s@65) | MSR      | ood_nsncs_places365    | 280 | 234.452 |     0.815 |   0.617 |
| ASH (ash_s@65) | MSR      | ood_nsncs_svhn         | 280 | 347.853 |     0.849 |   0.579 |
| ASH (ash_s@65) | MSR      | ood_nsncs_textures     | 280 | 157.689 |     0.831 |   0.604 |
| ASH (ash_s@65) | MSR      | ood_nsncs_ti           | 220 | 218.657 |     0.867 |   0.519 |
| ASH (ash_s@65) | MSR      | ood_sncs_c10           | 220 | 248.266 |     0.809 |   0.637 |
| ASH (ash_s@65) | MSR      | ood_sncs_c100          | 120 | 197.719 |     0.908 |   0.463 |
| ASH (ash_s@65) | PCE      | iid_test               | 280 |  98.584 |     0.854 |   0.61  |
| ASH (ash_s@65) | PCE      | ood_nsncs_isun         | 280 | 205.23  |     0.875 |   0.502 |
| ASH (ash_s@65) | PCE      | ood_nsncs_lsun_cropped | 280 | 219.313 |     0.881 |   0.494 |
| ASH (ash_s@65) | PCE      | ood_nsncs_lsun_resize  | 280 | 220.877 |     0.874 |   0.504 |
| ASH (ash_s@65) | PCE      | ood_nsncs_places365    | 280 | 234.22  |     0.816 |   0.615 |
| ASH (ash_s@65) | PCE      | ood_nsncs_svhn         | 280 | 347.654 |     0.85  |   0.58  |
| ASH (ash_s@65) | PCE      | ood_nsncs_textures     | 280 | 157.097 |     0.834 |   0.6   |
| ASH (ash_s@65) | PCE      | ood_nsncs_ti           | 220 | 217.666 |     0.871 |   0.506 |
| ASH (ash_s@65) | PCE      | ood_sncs_c10           | 220 | 248.063 |     0.809 |   0.637 |
| ASH (ash_s@65) | PCE      | ood_sncs_c100          | 120 | 197.442 |     0.909 |   0.462 |
| ASH (ash_s@65) | PE       | iid_test               | 280 |  99.815 |     0.848 |   0.625 |
| ASH (ash_s@65) | PE       | ood_nsncs_isun         | 280 | 204.102 |     0.88  |   0.499 |
| ASH (ash_s@65) | PE       | ood_nsncs_lsun_cropped | 280 | 218.837 |     0.883 |   0.501 |
| ASH (ash_s@65) | PE       | ood_nsncs_lsun_resize  | 280 | 219.627 |     0.879 |   0.5   |
| ASH (ash_s@65) | PE       | ood_nsncs_places365    | 280 | 234.137 |     0.817 |   0.616 |
| ASH (ash_s@65) | PE       | ood_nsncs_svhn         | 280 | 347.439 |     0.852 |   0.588 |
| ASH (ash_s@65) | PE       | ood_nsncs_textures     | 280 | 156.377 |     0.837 |   0.599 |
| ASH (ash_s@65) | PE       | ood_nsncs_ti           | 220 | 216.48  |     0.876 |   0.5   |
| ASH (ash_s@65) | PE       | ood_sncs_c10           | 220 | 248.235 |     0.809 |   0.653 |
| ASH (ash_s@65) | PE       | ood_sncs_c100          | 120 | 197.396 |     0.91  |   0.466 |
| ASH (ash_s@65) | REN      | iid_test               | 280 |  98.795 |     0.853 |   0.614 |
| ASH (ash_s@65) | REN      | ood_nsncs_isun         | 280 | 204.964 |     0.876 |   0.532 |
| ASH (ash_s@65) | REN      | ood_nsncs_lsun_cropped | 280 | 218.78  |     0.883 |   0.514 |
| ASH (ash_s@65) | REN      | ood_nsncs_lsun_resize  | 280 | 220.578 |     0.876 |   0.533 |
| ASH (ash_s@65) | REN      | ood_nsncs_places365    | 280 | 233.897 |     0.818 |   0.631 |
| ASH (ash_s@65) | REN      | ood_nsncs_svhn         | 280 | 346.43  |     0.858 |   0.57  |
| ASH (ash_s@65) | REN      | ood_nsncs_textures     | 280 | 156.6   |     0.836 |   0.624 |
| ASH (ash_s@65) | REN      | ood_nsncs_ti           | 220 | 217.688 |     0.872 |   0.55  |
| ASH (ash_s@65) | REN      | ood_sncs_c10           | 220 | 247.565 |     0.812 |   0.635 |
| ASH (ash_s@65) | REN      | ood_sncs_c100          | 120 | 196.505 |     0.913 |   0.458 |
| ReAct          | Energy   | iid_test               | 280 | 118.736 |     0.783 |   0.712 |
| ReAct          | Energy   | ood_nsncs_isun         | 280 | 210.203 |     0.865 |   0.518 |
| ReAct          | Energy   | ood_nsncs_lsun_cropped | 280 | 220.962 |     0.885 |   0.468 |
| ReAct          | Energy   | ood_nsncs_lsun_resize  | 280 | 225.519 |     0.865 |   0.512 |
| ReAct          | Energy   | ood_nsncs_places365    | 280 | 240.396 |     0.8   |   0.636 |
| ReAct          | Energy   | ood_nsncs_svhn         | 280 | 350.432 |     0.847 |   0.624 |
| ReAct          | Energy   | ood_nsncs_textures     | 280 | 161.661 |     0.826 |   0.607 |
| ReAct          | Energy   | ood_nsncs_ti           | 220 | 222.538 |     0.852 |   0.525 |
| ReAct          | Energy   | ood_sncs_c10           | 220 | 265.488 |     0.747 |   0.734 |
| ReAct          | Energy   | ood_sncs_c100          | 120 | 214.418 |     0.86  |   0.558 |
| ReAct          | GE       | iid_test               | 280 | 117.159 |     0.794 |   0.679 |
| ReAct          | GE       | ood_nsncs_isun         | 280 | 211.462 |     0.858 |   0.499 |
| ReAct          | GE       | ood_nsncs_lsun_cropped | 280 | 221.566 |     0.88  |   0.434 |
| ReAct          | GE       | ood_nsncs_lsun_resize  | 280 | 226.916 |     0.857 |   0.496 |
| ReAct          | GE       | ood_nsncs_places365    | 280 | 240.138 |     0.8   |   0.623 |
| ReAct          | GE       | ood_nsncs_svhn         | 280 | 349.718 |     0.848 |   0.577 |
| ReAct          | GE       | ood_nsncs_textures     | 280 | 162.709 |     0.821 |   0.592 |
| ReAct          | GE       | ood_nsncs_ti           | 220 | 224.371 |     0.841 |   0.527 |
| ReAct          | GE       | ood_sncs_c10           | 220 | 264.128 |     0.752 |   0.71  |
| ReAct          | GE       | ood_sncs_c100          | 120 | 209.755 |     0.88  |   0.501 |
| ReAct          | GEN      | iid_test               | 280 | 106.889 |     0.843 |   0.621 |
| ReAct          | GEN      | ood_nsncs_isun         | 280 | 209.246 |     0.872 |   0.515 |
| ReAct          | GEN      | ood_nsncs_lsun_cropped | 280 | 220.839 |     0.888 |   0.475 |
| ReAct          | GEN      | ood_nsncs_lsun_resize  | 280 | 225.201 |     0.87  |   0.515 |
| ReAct          | GEN      | ood_nsncs_places365    | 280 | 237.541 |     0.816 |   0.628 |
| ReAct          | GEN      | ood_nsncs_svhn         | 280 | 349.012 |     0.859 |   0.566 |
| ReAct          | GEN      | ood_nsncs_textures     | 280 | 160.65  |     0.832 |   0.615 |
| ReAct          | GEN      | ood_nsncs_ti           | 220 | 220.12  |     0.867 |   0.528 |
| ReAct          | GEN      | ood_sncs_c10           | 220 | 256.718 |     0.79  |   0.652 |
| ReAct          | GEN      | ood_sncs_c100          | 120 | 206.179 |     0.896 |   0.479 |
| ReAct          | GradNorm | iid_test               | 280 | 160.092 |     0.581 |   0.851 |
| ReAct          | GradNorm | ood_nsncs_isun         | 280 | 281.573 |     0.561 |   0.783 |
| ReAct          | GradNorm | ood_nsncs_lsun_cropped | 280 | 290.46  |     0.582 |   0.761 |
| ReAct          | GradNorm | ood_nsncs_lsun_resize  | 280 | 296.269 |     0.555 |   0.782 |
| ReAct          | GradNorm | ood_nsncs_places365    | 280 | 291.58  |     0.577 |   0.83  |
| ReAct          | GradNorm | ood_nsncs_svhn         | 280 | 392.373 |     0.563 |   0.837 |
| ReAct          | GradNorm | ood_nsncs_textures     | 280 | 218.803 |     0.594 |   0.821 |
| ReAct          | GradNorm | ood_nsncs_ti           | 220 | 262.292 |     0.682 |   0.724 |
| ReAct          | GradNorm | ood_sncs_c10           | 220 | 323.461 |     0.489 |   0.89  |
| ReAct          | GradNorm | ood_sncs_c100          | 120 | 315.532 |     0.42  |   0.852 |
| ReAct          | MLS      | iid_test               | 280 | 111.561 |     0.817 |   0.655 |
| ReAct          | MLS      | ood_nsncs_isun         | 280 | 208.126 |     0.876 |   0.482 |
| ReAct          | MLS      | ood_nsncs_lsun_cropped | 280 | 219.282 |     0.894 |   0.429 |
| ReAct          | MLS      | ood_nsncs_lsun_resize  | 280 | 223.697 |     0.876 |   0.479 |
| ReAct          | MLS      | ood_nsncs_places365    | 280 | 237.688 |     0.814 |   0.604 |
| ReAct          | MLS      | ood_nsncs_svhn         | 280 | 348.847 |     0.861 |   0.561 |
| ReAct          | MLS      | ood_nsncs_textures     | 280 | 159.281 |     0.837 |   0.581 |
| ReAct          | MLS      | ood_nsncs_ti           | 220 | 219.71  |     0.868 |   0.501 |
| ReAct          | MLS      | ood_sncs_c10           | 220 | 259.666 |     0.776 |   0.671 |
| ReAct          | MLS      | ood_sncs_c100          | 120 | 210.673 |     0.877 |   0.494 |
| ReAct          | MSR      | iid_test               | 280 | 107.546 |     0.84  |   0.62  |
| ReAct          | MSR      | ood_nsncs_isun         | 280 | 211.169 |     0.864 |   0.518 |
| ReAct          | MSR      | ood_nsncs_lsun_cropped | 280 | 221.963 |     0.883 |   0.464 |
| ReAct          | MSR      | ood_nsncs_lsun_resize  | 280 | 226.987 |     0.862 |   0.517 |
| ReAct          | MSR      | ood_nsncs_places365    | 280 | 238.572 |     0.811 |   0.621 |
| ReAct          | MSR      | ood_nsncs_svhn         | 280 | 349.938 |     0.852 |   0.566 |
| ReAct          | MSR      | ood_nsncs_textures     | 280 | 162.019 |     0.826 |   0.606 |
| ReAct          | MSR      | ood_nsncs_ti           | 220 | 221.431 |     0.861 |   0.526 |
| ReAct          | MSR      | ood_sncs_c10           | 220 | 258.53  |     0.781 |   0.664 |
| ReAct          | MSR      | ood_sncs_c100          | 120 | 208.427 |     0.886 |   0.493 |
| ReAct          | PCE      | iid_test               | 280 | 107.935 |     0.838 |   0.625 |
| ReAct          | PCE      | ood_nsncs_isun         | 280 | 209.737 |     0.869 |   0.499 |
| ReAct          | PCE      | ood_nsncs_lsun_cropped | 280 | 220.64  |     0.888 |   0.439 |
| ReAct          | PCE      | ood_nsncs_lsun_resize  | 280 | 225.516 |     0.868 |   0.497 |
| ReAct          | PCE      | ood_nsncs_places365    | 280 | 237.884 |     0.813 |   0.611 |
| ReAct          | PCE      | ood_nsncs_svhn         | 280 | 349.213 |     0.857 |   0.551 |
| ReAct          | PCE      | ood_nsncs_textures     | 280 | 161.058 |     0.829 |   0.595 |
| ReAct          | PCE      | ood_nsncs_ti           | 220 | 220.587 |     0.864 |   0.517 |
| ReAct          | PCE      | ood_sncs_c10           | 220 | 257.992 |     0.783 |   0.656 |
| ReAct          | PCE      | ood_sncs_c100          | 120 | 207.229 |     0.891 |   0.475 |
| ReAct          | PE       | iid_test               | 280 | 110.347 |     0.826 |   0.649 |
| ReAct          | PE       | ood_nsncs_isun         | 280 | 209.07  |     0.87  |   0.49  |
| ReAct          | PE       | ood_nsncs_lsun_cropped | 280 | 219.805 |     0.89  |   0.424 |
| ReAct          | PE       | ood_nsncs_lsun_resize  | 280 | 224.696 |     0.869 |   0.488 |
| ReAct          | PE       | ood_nsncs_places365    | 280 | 237.737 |     0.813 |   0.61  |
| ReAct          | PE       | ood_nsncs_svhn         | 280 | 348.575 |     0.86  |   0.55  |
| ReAct          | PE       | ood_nsncs_textures     | 280 | 160.657 |     0.83  |   0.589 |
| ReAct          | PE       | ood_nsncs_ti           | 220 | 220.817 |     0.86  |   0.518 |
| ReAct          | PE       | ood_sncs_c10           | 220 | 259.022 |     0.777 |   0.673 |
| ReAct          | PE       | ood_sncs_c100          | 120 | 206.799 |     0.893 |   0.469 |
| ReAct          | REN      | iid_test               | 280 | 107.219 |     0.842 |   0.632 |
| ReAct          | REN      | ood_nsncs_isun         | 280 | 208.523 |     0.875 |   0.538 |
| ReAct          | REN      | ood_nsncs_lsun_cropped | 280 | 221.479 |     0.886 |   0.505 |
| ReAct          | REN      | ood_nsncs_lsun_resize  | 280 | 224.415 |     0.874 |   0.539 |
| ReAct          | REN      | ood_nsncs_places365    | 280 | 238.263 |     0.813 |   0.649 |
| ReAct          | REN      | ood_nsncs_svhn         | 280 | 348.28  |     0.864 |   0.567 |
| ReAct          | REN      | ood_nsncs_textures     | 280 | 160.705 |     0.831 |   0.635 |
| ReAct          | REN      | ood_nsncs_ti           | 220 | 219.013 |     0.872 |   0.548 |
| ReAct          | REN      | ood_sncs_c10           | 220 | 255.577 |     0.795 |   0.652 |
| ReAct          | REN      | ood_sncs_c100          | 120 | 204.873 |     0.901 |   0.482 |
| base           | CTM      | iid_test               | 320 |  93.281 |     0.822 |   0.61  |
| base           | CTM      | ood_nsncs_isun         | 320 | 196.934 |     0.889 |   0.425 |
| base           | CTM      | ood_nsncs_lsun_cropped | 320 | 212.864 |     0.886 |   0.418 |
| base           | CTM      | ood_nsncs_lsun_resize  | 320 | 212.377 |     0.889 |   0.422 |
| base           | CTM      | ood_nsncs_places365    | 320 | 227.488 |     0.823 |   0.558 |
| base           | CTM      | ood_nsncs_svhn         | 320 | 338.817 |     0.882 |   0.496 |
| base           | CTM      | ood_nsncs_textures     | 320 | 144.5   |     0.867 |   0.496 |
| base           | CTM      | ood_nsncs_ti           | 250 | 211.357 |     0.878 |   0.453 |
| base           | CTM      | ood_sncs_c10           | 250 | 243.391 |     0.803 |   0.583 |
| base           | CTM      | ood_sncs_c100          | 140 | 197.736 |     0.893 |   0.38  |
| base           | CTM      | ood_sncs_sc100         |  70 | 233.323 |     0.931 |   0.356 |
| base           | Energy   | iid_test               | 320 |  93.816 |     0.828 |   0.653 |
| base           | Energy   | ood_nsncs_isun         | 320 | 199.678 |     0.877 |   0.499 |
| base           | Energy   | ood_nsncs_lsun_cropped | 320 | 213.63  |     0.883 |   0.498 |
| base           | Energy   | ood_nsncs_lsun_resize  | 320 | 214.8   |     0.878 |   0.498 |
| base           | Energy   | ood_nsncs_places365    | 320 | 227.125 |     0.824 |   0.59  |
| base           | Energy   | ood_nsncs_svhn         | 320 | 342.359 |     0.856 |   0.577 |
| base           | Energy   | ood_nsncs_textures     | 320 | 148.184 |     0.852 |   0.554 |
| base           | Energy   | ood_nsncs_ti           | 250 | 211.553 |     0.877 |   0.487 |
| base           | Energy   | ood_sncs_c10           | 250 | 241.402 |     0.812 |   0.649 |
| base           | Energy   | ood_sncs_c100          | 140 | 194.489 |     0.905 |   0.456 |
| base           | Energy   | ood_sncs_sc100         |  70 | 238.602 |     0.907 |   0.515 |
| base           | MLS      | iid_test               | 320 |  91.824 |     0.839 |   0.62  |
| base           | MLS      | ood_nsncs_isun         | 320 | 199.639 |     0.877 |   0.481 |
| base           | MLS      | ood_nsncs_lsun_cropped | 320 | 212.979 |     0.886 |   0.471 |
| base           | MLS      | ood_nsncs_lsun_resize  | 320 | 214.889 |     0.878 |   0.483 |
| base           | MLS      | ood_nsncs_places365    | 320 | 226.386 |     0.828 |   0.574 |
| base           | MLS      | ood_nsncs_svhn         | 320 | 341.573 |     0.861 |   0.547 |
| base           | MLS      | ood_nsncs_textures     | 320 | 147.885 |     0.854 |   0.54  |
| base           | MLS      | ood_nsncs_ti           | 250 | 211.793 |     0.876 |   0.479 |
| base           | MLS      | ood_sncs_c10           | 250 | 239.95  |     0.818 |   0.615 |
| base           | MLS      | ood_sncs_c100          | 140 | 193.409 |     0.91  |   0.424 |
| base           | MLS      | ood_sncs_sc100         |  70 | 236.757 |     0.915 |   0.455 |
| base           | MSR      | iid_test               | 320 |  88.503 |     0.859 |   0.586 |
| base           | MSR      | ood_nsncs_isun         | 320 | 201.321 |     0.87  |   0.5   |
| base           | MSR      | ood_nsncs_lsun_cropped | 320 | 214     |     0.881 |   0.479 |
| base           | MSR      | ood_nsncs_lsun_resize  | 320 | 216.804 |     0.869 |   0.502 |
| base           | MSR      | ood_nsncs_places365    | 320 | 226.841 |     0.826 |   0.587 |
| base           | MSR      | ood_nsncs_svhn         | 320 | 342.019 |     0.858 |   0.546 |
| base           | MSR      | ood_nsncs_textures     | 320 | 149.376 |     0.847 |   0.553 |
| base           | MSR      | ood_nsncs_ti           | 250 | 214.021 |     0.867 |   0.506 |
| base           | MSR      | ood_sncs_c10           | 250 | 239.975 |     0.818 |   0.609 |
| base           | MSR      | ood_sncs_c100          | 140 | 193.193 |     0.911 |   0.433 |
| base           | MSR      | ood_sncs_sc100         |  70 | 236.934 |     0.914 |   0.443 |
| base           | Maha     | iid_test               | 320 | 132.918 |     0.645 |   0.855 |
| base           | Maha     | ood_nsncs_isun         | 320 | 246.555 |     0.677 |   0.808 |
| base           | Maha     | ood_nsncs_lsun_cropped | 320 | 274.701 |     0.618 |   0.85  |
| base           | Maha     | ood_nsncs_lsun_resize  | 320 | 259.475 |     0.681 |   0.809 |
| base           | Maha     | ood_nsncs_places365    | 320 | 260.63  |     0.676 |   0.754 |
| base           | Maha     | ood_nsncs_svhn         | 320 | 372.166 |     0.658 |   0.813 |
| base           | Maha     | ood_nsncs_textures     | 320 | 183.967 |     0.707 |   0.7   |
| base           | Maha     | ood_nsncs_ti           | 250 | 241.384 |     0.743 |   0.758 |
| base           | Maha     | ood_sncs_c10           | 250 | 307.212 |     0.522 |   0.922 |
| base           | Maha     | ood_sncs_c100          | 140 | 258.899 |     0.627 |   0.739 |
| base           | Maha     | ood_sncs_sc100         |  70 | 358.662 |     0.366 |   0.977 |
| base           | MahaPP   | iid_test               | 320 | 108.172 |     0.762 |   0.758 |
| base           | MahaPP   | ood_nsncs_isun         | 320 | 221.583 |     0.781 |   0.677 |
| base           | MahaPP   | ood_nsncs_lsun_cropped | 320 | 251.528 |     0.717 |   0.753 |
| base           | MahaPP   | ood_nsncs_lsun_resize  | 320 | 235.903 |     0.782 |   0.677 |
| base           | MahaPP   | ood_nsncs_places365    | 320 | 235.486 |     0.785 |   0.656 |
| base           | MahaPP   | ood_nsncs_svhn         | 320 | 359.098 |     0.741 |   0.727 |
| base           | MahaPP   | ood_nsncs_textures     | 320 | 162.336 |     0.794 |   0.624 |
| base           | MahaPP   | ood_nsncs_ti           | 250 | 221.442 |     0.828 |   0.604 |
| base           | MahaPP   | ood_sncs_c10           | 250 | 275.809 |     0.658 |   0.845 |
| base           | MahaPP   | ood_sncs_c100          | 140 | 232.01  |     0.743 |   0.684 |
| base           | NCI      | iid_test               | 320 |  89.24  |     0.851 |   0.604 |
| base           | NCI      | ood_nsncs_isun         | 320 | 197.779 |     0.885 |   0.466 |
| base           | NCI      | ood_nsncs_lsun_cropped | 320 | 211.444 |     0.892 |   0.436 |
| base           | NCI      | ood_nsncs_lsun_resize  | 320 | 213.034 |     0.886 |   0.464 |
| base           | NCI      | ood_nsncs_places365    | 320 | 226.136 |     0.828 |   0.574 |
| base           | NCI      | ood_nsncs_svhn         | 320 | 338.841 |     0.879 |   0.501 |
| base           | NCI      | ood_nsncs_textures     | 320 | 145.069 |     0.865 |   0.517 |
| base           | NCI      | ood_nsncs_ti           | 250 | 210.462 |     0.882 |   0.483 |
| base           | NCI      | ood_sncs_c10           | 250 | 240.925 |     0.814 |   0.603 |
| base           | NCI      | ood_sncs_c100          | 140 | 192.052 |     0.916 |   0.396 |
| base           | NNGuide  | iid_test               | 320 |  92.32  |     0.838 |   0.626 |
| base           | NNGuide  | ood_nsncs_isun         | 320 | 197.845 |     0.885 |   0.458 |
| base           | NNGuide  | ood_nsncs_lsun_cropped | 320 | 212.463 |     0.887 |   0.45  |
| base           | NNGuide  | ood_nsncs_lsun_resize  | 320 | 213.103 |     0.885 |   0.459 |
| base           | NNGuide  | ood_nsncs_places365    | 320 | 226.786 |     0.825 |   0.575 |
| base           | NNGuide  | ood_nsncs_svhn         | 320 | 341.074 |     0.863 |   0.534 |
| base           | NNGuide  | ood_nsncs_textures     | 320 | 147.076 |     0.857 |   0.526 |
| base           | NNGuide  | ood_nsncs_ti           | 250 | 210.213 |     0.883 |   0.468 |
| base           | NNGuide  | ood_sncs_c10           | 250 | 240.205 |     0.817 |   0.614 |
| base           | NNGuide  | ood_sncs_c100          | 140 | 192.619 |     0.913 |   0.409 |
| base           | NNGuide  | ood_sncs_sc100         |  70 | 236.298 |     0.917 |   0.432 |
| base           | NeCo     | iid_test               | 320 |  97.072 |     0.815 |   0.672 |
| base           | NeCo     | ood_nsncs_isun         | 320 | 204.72  |     0.854 |   0.545 |
| base           | NeCo     | ood_nsncs_lsun_cropped | 320 | 217.809 |     0.864 |   0.534 |
| base           | NeCo     | ood_nsncs_lsun_resize  | 320 | 220.243 |     0.852 |   0.55  |
| base           | NeCo     | ood_nsncs_places365    | 320 | 228.253 |     0.818 |   0.601 |
| base           | NeCo     | ood_nsncs_svhn         | 320 | 344.059 |     0.843 |   0.599 |
| base           | NeCo     | ood_nsncs_textures     | 320 | 150     |     0.844 |   0.561 |
| base           | NeCo     | ood_nsncs_ti           | 250 | 214.89  |     0.859 |   0.512 |
| base           | NeCo     | ood_sncs_c10           | 250 | 247.765 |     0.781 |   0.704 |
| base           | NeCo     | ood_sncs_c100          | 140 | 199.676 |     0.883 |   0.507 |
| base           | NeCo     | ood_sncs_sc100         |  70 | 249.082 |     0.859 |   0.624 |
| base           | Residual | iid_test               | 320 | 135.455 |     0.626 |   0.872 |
| base           | Residual | ood_nsncs_isun         | 320 | 252.024 |     0.655 |   0.813 |
| base           | Residual | ood_nsncs_lsun_cropped | 320 | 282.985 |     0.584 |   0.869 |
| base           | Residual | ood_nsncs_lsun_resize  | 320 | 264.067 |     0.663 |   0.814 |
| base           | Residual | ood_nsncs_places365    | 320 | 265.486 |     0.656 |   0.767 |
| base           | Residual | ood_nsncs_svhn         | 320 | 378.057 |     0.623 |   0.839 |
| base           | Residual | ood_nsncs_textures     | 320 | 190.338 |     0.681 |   0.723 |
| base           | Residual | ood_nsncs_ti           | 250 | 244.784 |     0.73  |   0.76  |
| base           | Residual | ood_sncs_c10           | 250 | 312.634 |     0.5   |   0.936 |
| base           | Residual | ood_sncs_c100          | 140 | 268.185 |     0.589 |   0.766 |
| base           | Residual | ood_sncs_sc100         |  70 | 371.469 |     0.309 |   0.986 |
| base           | ViM      | iid_test               | 320 | 107.064 |     0.768 |   0.77  |
| base           | ViM      | ood_nsncs_isun         | 320 | 213.473 |     0.815 |   0.672 |
| base           | ViM      | ood_nsncs_lsun_cropped | 320 | 236.595 |     0.781 |   0.738 |
| base           | ViM      | ood_nsncs_lsun_resize  | 320 | 227.509 |     0.818 |   0.667 |
| base           | ViM      | ood_nsncs_places365    | 320 | 235.125 |     0.787 |   0.677 |
| base           | ViM      | ood_nsncs_svhn         | 320 | 351.855 |     0.79  |   0.71  |
| base           | ViM      | ood_nsncs_textures     | 320 | 158.317 |     0.81  |   0.618 |
| base           | ViM      | ood_nsncs_ti           | 250 | 218.674 |     0.84  |   0.619 |
| base           | ViM      | ood_sncs_c10           | 250 | 267.544 |     0.693 |   0.845 |
| base           | ViM      | ood_sncs_c100          | 140 | 221.619 |     0.788 |   0.67  |
| base           | ViM      | ood_sncs_sc100         |  70 | 291.523 |     0.667 |   0.899 |
| base           | fDBD     | iid_test               | 320 |  96.912 |     0.815 |   0.623 |
| base           | fDBD     | ood_nsncs_isun         | 320 | 204.201 |     0.854 |   0.482 |
| base           | fDBD     | ood_nsncs_lsun_cropped | 320 | 218.178 |     0.859 |   0.457 |
| base           | fDBD     | ood_nsncs_lsun_resize  | 320 | 219.356 |     0.854 |   0.48  |
| base           | fDBD     | ood_nsncs_places365    | 320 | 230.706 |     0.806 |   0.575 |
| base           | fDBD     | ood_nsncs_svhn         | 320 | 343.743 |     0.842 |   0.528 |
| base           | fDBD     | ood_nsncs_textures     | 320 | 153.82  |     0.828 |   0.53  |
| base           | fDBD     | ood_nsncs_ti           | 250 | 218.898 |     0.839 |   0.505 |
| base           | fDBD     | ood_sncs_c10           | 250 | 246.284 |     0.786 |   0.619 |
| base           | fDBD     | ood_sncs_c100          | 140 | 196.168 |     0.899 |   0.412 |
| base           | fDBD     | ood_sncs_sc100         |  70 | 236.37  |     0.917 |   0.415 |

## New-method AUGRC rank among all base-arm CSFs

| csf    | mode                   |   rank |   of | best_in_mode            |   best_AUGRC |
|:-------|:-----------------------|-------:|-----:|:------------------------|-------------:|
| MahaPP | iid_test               |     73 |   85 | GEN_global              |       88.311 |
| MahaPP | ood_nsncs_isun         |     71 |   85 | CTM_global              |      196.548 |
| MahaPP | ood_nsncs_lsun_cropped |     74 |   85 | NCI                     |      211.444 |
| MahaPP | ood_nsncs_lsun_resize  |     71 |   85 | CTM_global              |      211.894 |
| MahaPP | ood_nsncs_places365    |     65 |   85 | PCA_RecError_class_pred |      225.204 |
| MahaPP | ood_nsncs_svhn         |     73 |   85 | CTM_oc_mean             |      338.376 |
| MahaPP | ood_nsncs_textures     |     63 |   85 | PCA_RecError_class_pred |      143.104 |
| MahaPP | ood_nsncs_ti           |     60 |   85 | NNGuide                 |      210.213 |
| MahaPP | ood_sncs_c10           |     75 |   85 | PCE_global              |      239.683 |
| MahaPP | ood_sncs_c100          |     72 |   85 | NCI                     |      192.052 |
| NCI    | iid_test               |     13 |   85 | GEN_global              |       88.311 |
| NCI    | ood_nsncs_isun         |      4 |   85 | CTM_global              |      196.548 |
| NCI    | ood_nsncs_lsun_cropped |      1 |   85 | NCI                     |      211.444 |
| NCI    | ood_nsncs_lsun_resize  |      4 |   85 | CTM_global              |      211.894 |
| NCI    | ood_nsncs_places365    |      2 |   85 | PCA_RecError_class_pred |      225.204 |
| NCI    | ood_nsncs_svhn         |      4 |   85 | CTM_oc_mean             |      338.376 |
| NCI    | ood_nsncs_textures     |      5 |   85 | PCA_RecError_class_pred |      143.104 |
| NCI    | ood_nsncs_ti           |      4 |   85 | NNGuide                 |      210.213 |
| NCI    | ood_sncs_c10           |     28 |   85 | PCE_global              |      239.683 |
| NCI    | ood_sncs_c100          |      1 |   85 | NCI                     |      192.052 |
