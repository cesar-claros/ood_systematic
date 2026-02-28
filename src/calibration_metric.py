# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchmetrics import CalibrationError
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.isotonic import IsotonicRegression
from optbinning import OptimalBinning
import seaborn as sns
import pandas as pd
import scipy
from sklearn.utils.validation import column_or_1d
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import _check_pos_label_consistency
from scipy.stats import binom, binomtest, norm, t, beta as beta_dist



def ece(preds, labels, n_bin=20, mode='l1', savepath=False): 
    bin_preds, bin_count, bin_total, bins = calibration_summary(preds, labels, "uniform", n_bin=n_bin)
    prob_pred = np.array([ elem.mean() if len(elem) > 0 else 0.0 for elem in bin_preds ])
    prob_data = np.zeros(len(bin_total))
    prob_data[bin_total!=0] = bin_count[bin_total!=0] / bin_total[bin_total!=0]
    
    val = 0
    if mode == 'l1':
        val = np.sum( np.abs( prob_data - prob_pred ) * bin_total ) / np.sum(bin_total)
    elif mode == 'l2':
        val = np.sum( ( np.abs( prob_data - prob_pred ) ** 2 ) * bin_total ) / np.sum(bin_total)
    elif mode == 'inf':
        val = np.max( np.abs( prob_data - prob_pred ) )
    else:
        assert False, 'no correct mode specified: (l1, l2, inf)'
        
    if savepath != False:
        plot_reliability_diagram(prob_pred, prob_data, bin_total, preds, bins, savepath)
        
    return val



def ace(preds, labels, n_bin=10, mode='l1', savepath=False): 
    bin_preds, bin_count, bin_total, bins = calibration_summary(preds, labels, "quantile", n_bin=n_bin)
    prob_pred = np.array([ elem.mean() if len(elem) > 0 else 0.0 for elem in bin_preds ])
    prob_data = np.zeros(len(bin_total))
    prob_data[bin_total!=0] = bin_count[bin_total!=0] / bin_total[bin_total!=0]
    
    val = 0
    if mode == 'l1':
        val = np.sum( np.abs( prob_data - prob_pred ) * bin_total ) / np.sum(bin_total)
    elif mode == 'l2':
        val = np.sum( ( np.abs( prob_data - prob_pred ) ** 2 ) * bin_total ) / np.sum(bin_total)
    elif mode == 'inf':
        val = np.max( np.abs( prob_data - prob_pred ) )
    else:
        assert False, 'no correct mode specified: (l1, l2, inf)'
        
    # if savepath != False:
    plot_reliability_diagram(prob_pred, prob_data, bin_total, preds, bins, savepath)
        
    return val



def lce(preds, labels, n_min=10, n_max=1000, mode='l1', savepath=False): 
    bin_preds, bin_count, bin_total, bins = calibration_summary(preds, labels, "pavabc", n_min=n_min, n_max=n_max)
    prob_pred = np.array([ elem.mean() if len(elem) > 0 else 0.0 for elem in bin_preds ])
    prob_data = np.zeros(len(bin_total))
    prob_data[bin_total!=0] = bin_count[bin_total!=0] / bin_total[bin_total!=0]
    
    val = 0
    if mode == 'l1':
        val = np.sum( np.abs( prob_data - prob_pred ) * bin_total ) / np.sum(bin_total)
    elif mode == 'l2':
        val = np.sum( ( np.abs( prob_data - prob_pred ) ** 2 ) * bin_total ) / np.sum(bin_total)
    elif mode == 'linf':
        val = np.max( np.abs( prob_data - prob_pred ) )
    else:
        assert False, 'no correct mode specified: (l1, l2, inf)'
        
    if savepath != False:
        plot_reliability_diagram(prob_pred, prob_data, bin_total, preds, bins, savepath)
    
    return val



def tce(preds, labels, siglevel=0.05, strategy='pavabc', n_min=10, n_max=1000, n_bin=10, savepath=False, ymax=None, optb_kwargs=None):
    assert labels.shape[0] != n_min, "The minimum bin size equals to the data size. No binning needed."
    bin_preds, bin_count, bin_total, _ = calibration_summary(preds, labels, strategy, n_min=n_min, n_max=n_max, n_bin=n_bin, optb_kwargs=optb_kwargs)
    
    bin_rnum = np.zeros(len(bin_count))
    for i in range(len(bin_rnum)):
        pvals = np.array([ binomtest(bin_count[i], bin_total[i], p=p).pvalue for p in bin_preds[i] ])
        bin_rnum[i] = sum((pvals <= siglevel))
    
    if savepath != False:
        plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath, ymax)
    
    return 100 * np.sum(bin_rnum) / np.sum(bin_total)
    #return np.sum(bin_rnum) / np.sum(bin_total), bin_rnum, bin_preds, bin_count, bin_total

    

def tce_ttest(preds, labels, siglevel=0.05, strategy='pavabc', n_min=10, n_max=1000, n_bin=10, savepath=False, ymax=None, optb_kwargs=None):
    assert labels.shape[0] != n_min, "The minimum bin size equals to the data size. No binning needed."
    
    bin_preds, bin_count, bin_total, _ = calibration_summary(preds, labels, strategy, n_min=n_min, n_max=n_max, n_bin=n_bin, optb_kwargs=optb_kwargs)
    
    bin_rnum = np.zeros(len(bin_count))
    for i in range(len(bin_rnum)):
        ni = bin_total[i]
        mu = bin_count[i] / ni
        sd = mu * ( 1 - mu )
        if sd == 0:
            bin_rnum[i] = len(bin_preds[i])
        else:
            pvals = np.array([ 2.0*t.sf(np.sqrt(ni)*np.abs(mu-p)/sd, ni-1) for p in bin_preds[i] ])
            bin_rnum[i] = sum((pvals <= siglevel))
    
    # if savepath != False:
    plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath, ymax)
    
    return 100 * np.sum(bin_rnum) / np.sum(bin_total)
    #return np.sum(bin_rnum) / np.sum(bin_total), bin_rnum, bin_preds, bin_count, bin_total
    
    
    
def plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath=False, ymax=None):
    ### Prepare values (start) ###
    bin_prob = np.zeros(len(bin_total))
    bin_prob[bin_total!=0] = bin_count[bin_total!=0] / bin_total[bin_total!=0]
    
    width = 1 / ( bin_total.shape[0] + 1 )
    positions = np.linspace(0.0, 1.0, bin_total.shape[0]+1)[:-1] + width / 2.0
    if ymax == None:
        ymax = np.maximum( max(bin_prob), np.array([ elem.mean() for elem in bin_preds ]).max() )
        ymax = 1.25 * ymax if ymax < 0.7 else 1.0
    ### Prepare values (end) ###
        
    ### Plot (start) ###
    ratio = 4
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [ratio, 1], 'height_ratios': [ratio, 1]})
        
    axs[1, 1].set_visible(False)
    
    axs[0, 1].hist(np.concatenate(bin_preds), bins=30, orientation="horizontal")
    axs[0, 1].set_box_aspect(ratio/1)
    axs[0, 1].set_ylim(0, ymax)
    axs[0, 1].set_yticklabels([])
    axs[0, 1].xaxis.set_label_position("top")
    axs[0, 1].xaxis.tick_top()
    axs[0, 1].tick_params(axis='x', labelsize=12)
    axs[0, 1].set_xlabel("Count", fontsize=14)
    
    axs[1, 0].bar(positions, bin_total, width=width, color="grey", alpha=0.5, linewidth=3)
    axs[1, 0].bar(positions, bin_rnum, width=width, color="red", alpha=0.5, linewidth=3)
    axs[1, 0].set_box_aspect(1/ratio)
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].set_xticks(positions)
    axs[1, 0].set_xticklabels(["{:d}".format(i+1) for i in range(positions.shape[0])])
    axs[1, 0].tick_params(axis='x', labelsize=12)
    axs[1, 0].tick_params(axis='y', labelsize=12)
    axs[1, 0].set_xlabel("Bin ID", fontsize=14)
    axs[1, 0].set_ylabel("Count", fontsize=14)
    
    conf_plt = axs[0, 0].violinplot(bin_preds, positions, widths=width*0.8, vert=True, showmeans=True, showextrema=True, showmedians=False, bw_method=None)
    accr_plt = axs[0, 0].hlines(bin_prob, positions-(0.8*width/2.0), positions+(0.8*width/2.0), linestyle="-", linewidth=3, color="red", label="Empirical Probability")
    axs[0, 0].set_box_aspect(1)
    axs[0, 0].set_xlim(0, 1.0)
    axs[0, 0].set_ylim(0, ymax)
    axs[0, 0].set_xticks(positions)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].xaxis.set_label_position("top")
    axs[0, 0].set_xlabel(" ", fontsize=14)
    axs[0, 0].set_ylabel(r"$P_\theta(y=1 \mid x)$", fontsize=14)
    axs[0, 0].set_title(r"Estimates vs Predictions", fontsize=14)
    axs[0, 0].tick_params(axis='y', labelsize=12)
    axs[0, 0].legend(handles=[accr_plt], loc='upper left', fontsize=12)
    ### Plot (end) ###
    
    ### Save (start) ###
    if not savepath == False:
        fig.savefig(savepath, dpi=288)
        plt.close(fig)
    ### Plot (end) ###


    
def plot_reliability_diagram(prob_pred, prob_data, bin_total, preds, bins, savepath=False):
    #
    width = 1 / bins.shape[0]
    positions = np.linspace(0.0, 1.0, bins.shape[0])[:-1] + width / 2.0
    ymax = np.maximum( max(prob_data), preds.max() )
    ymax = 1.25 * ymax if ymax < 0.7 else 1.0
    
    ratio = 4
    fig, axs = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw={'width_ratios': [ratio, 1], 'height_ratios': [ratio, 1]})
        
    axs[1, 1].set_visible(False)
        
    axs[0, 1].hist(preds, bins=30, orientation="horizontal")
    axs[0, 1].set_box_aspect(ratio/1)
    axs[0, 1].set_ylim(0, ymax)
    axs[0, 1].set_yticklabels([])
    axs[0, 1].xaxis.set_label_position("top")
    axs[0, 1].xaxis.tick_top()
    axs[0, 1].tick_params(axis='x', labelsize=12)
    axs[0, 1].set_xlabel("Count", fontsize=14)
    
    axs[1, 0].bar(positions, bin_total, width=width, color="grey", alpha=0.5, linewidth=3)
    axs[1, 0].set_box_aspect(1/ratio)
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].set_xticks(positions)
    axs[1, 0].set_xticklabels(["{:d}".format(i+1) for i in range(positions.shape[0])])
    axs[1, 0].tick_params(axis='x', labelsize=12)
    axs[1, 0].tick_params(axis='y', labelsize=12)
    axs[1, 0].set_xlabel("Bin ID", fontsize=14)
    axs[1, 0].set_ylabel("Count", fontsize=14)
    
    conf_plt = axs[0, 0].bar(positions, prob_pred, width=width, color="blue", alpha=0.5, linewidth=3, label="Confidence")
    accr_plt = axs[0, 0].bar(positions, prob_data, width=width, color="red", alpha=0.5, linewidth=3, label="Accuracy")
    line_plt = axs[0, 0].hlines(bins[0:-1], positions-(0.8*width/2.0), positions+(0.8*width/2.0), linestyle="dotted", linewidth=1, color="black", label="Bin Boundary")
    axs[0, 0].hlines(bins[1:], positions-(0.8*width/2.0), positions+(0.8*width/2.0), linestyle="dotted", linewidth=1, color="black")
    axs[0, 0].set_box_aspect(1)
    axs[0, 0].set_xlim(0, 1.0)
    axs[0, 0].set_ylim(0, ymax)
    axs[0, 0].set_xticks(positions)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].xaxis.set_label_position("top")
    axs[0, 0].set_xlabel(" ", fontsize=14)
    axs[0, 0].set_ylabel(r"$P_\theta(y = 1 \mid x)$", fontsize=14)
    axs[0, 0].set_title(r"Accuracies vs Confidences", fontsize=14)
    axs[0, 0].tick_params(axis='y', labelsize=12)
    axs[0, 0].legend(handles=[accr_plt, conf_plt, line_plt], loc='upper left', fontsize=12)
        
    if not savepath == False:
        fig.savefig(savepath, dpi=288)
        plt.close(fig)

    
    
def calibration_summary(preds, labels, strategy='pavabc', n_min=10, n_max=1000, n_bin=10, optb_kwargs=None):
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0), "Prediction Out of Range [0, 1]"
    assert np.all((labels == 0) | (labels == 1)), "Label Not 0 or 1"

    if strategy == 'pavabc':
        bin_preds, bin_count, bin_total, bins = _pavabc(preds, labels, n_min=n_min, n_max=n_max)
    elif strategy == 'pava':
        bin_preds, bin_count, bin_total, bins = _pavabc(preds, labels, n_min=0, n_max=len(preds)+1)
    elif strategy == 'uniform':
        bin_preds, bin_count, bin_total, bins = _calibration_process(preds, labels, strategy, n_bin)
    elif strategy == 'quantile':
        bin_preds, bin_count, bin_total, bins = _calibration_process(preds, labels, strategy, n_bin)
    elif strategy == 'optbinning':
        bin_preds, bin_count, bin_total, bins = _optbinning_process(preds, labels, optb_kwargs=optb_kwargs)
    else:
        assert False, 'no correct strategy specified: (uniform, quantile, pava, pavabc, optbinning)'
    
    return bin_preds, bin_count, bin_total, bins



def _pavabc(x, y, n_min=0, n_max=10000):
    ### Sort (Start) ###
    order = np.argsort(x)
    xsort = x[order]
    ysort = y[order]
    num_y = len(ysort)
    ### Sort (End) ###
    
    def _condition(y0, y1, w0, w1):
        condition1 = ( w0 + w1 <= n_min )
        condition2 = ( w0 + w1 <= n_max )
        condition3 = ( y0 / w0 >= y1 / w1 )
        return condition1 or (condition2 and condition3)
    
    ### PAVA with Number Constraint (Start) ###
    count = -1
    iso_y = []
    iso_w = []
    for i in range(num_y - n_min):
        count += 1
        iso_y.append(ysort[i])
        iso_w.append(1)
        while count > 0 and _condition(iso_y[count-1], iso_y[count], iso_w[count-1], iso_w[count]):
            iso_y[count-1] += iso_y[count]
            iso_w[count-1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    if n_min > 0:
        count += 1
        iso_y.append(sum(ysort[num_y-n_min:num_y]))
        iso_w.append(n_min)
        if iso_w[-1] + n_min <= n_max:
            iso_y[count-1] += iso_y[count]
            iso_w[count-1] += iso_w[count]
            iso_y.pop()
            iso_w.pop()
            count -= 1
    ### PAVA with Number Constraint (End) ###
    
    ### Process return values (Start) ###
    index = np.r_[0, np.cumsum(iso_w)]
    bins = np.r_[0.0, [(xsort[index[j]-1]+xsort[index[j]])/2.0 for j in range(1, len(index)-1)], 1.0]
    bin_count = np.array(iso_y)
    bin_total = np.array(iso_w)
    bin_preds = [ xsort[ index[j]:index[j+1] ] for j in range(len(index)-1) ]
    ### Process return values (End) ###
    
    return bin_preds, bin_count, bin_total, bins



def _calibration_process(preds, labels, strategy="uniform", n_bin=10):
    if strategy == 'uniform':
        bins = np.linspace(0.0, 1.0, n_bin+1)
        bins[-1] = 1.1 #trick to include 'pred=1.0' in the final bin
        indices = np.digitize(preds, bins, right=False) - 1
        bins[-1] = 1.0 #put it back to 1.0
        bin_count = np.array([ sum(labels[indices==i]) for i in range(bins.shape[0]-1) ]).astype(int)
        bin_total = np.array([ len(labels[indices==i]) for i in range(bins.shape[0]-1) ]).astype(int)
        bin_preds = [ preds[indices==i] for i in range(bins.shape[0]-1) ]
        return bin_preds, bin_count, bin_total, bins
    
    elif strategy == 'quantile':
        quantile = np.linspace(0, 1, n_bin+1)
        #bins = np.percentile(preds, quantile * 100)
        #bins[0] = 0.0
        #bins[-1] = 1.0
        sortedindices = np.argsort(preds)
        sortedlabels = labels[sortedindices]
        sortedpreds = preds[sortedindices]
        idpartition = ( quantile * len(labels) ).astype(int)
        bin_count = np.array([ sum(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:]) ]).astype(int)
        bin_total = np.array([ len(sortedlabels[s:e]) for s, e in zip(idpartition, idpartition[1:]) ]).astype(int)
        bin_preds = [ sortedpreds[s:e] for s, e in zip(idpartition, idpartition[1:]) ]
        bins = np.array([ 0.0 ] + [ ( sortedpreds[e-1] + sortedpreds[e] ) / 2.0 for e in idpartition[1:-1] ] + [ 1.0 ])
        return bin_preds, bin_count, bin_total, bins
        
    else:
        assert False, 'no correct strategy specified: (uniform, quantile)'



def _optbinning_process(preds, labels, optb_kwargs=None):
    """Compute calibration bins using OptimalBinning with Hellinger divergence.
    
    Any extra keyword arguments for OptimalBinning can be passed via optb_kwargs
    and will override the defaults (dtype, solver, divergence).
    """
    defaults = {
        "dtype": "numerical",
        "solver": "mip",
        "divergence": "hellinger",
    }
    if optb_kwargs is not None:
        defaults.update(optb_kwargs)
    optb = OptimalBinning(**defaults)
    optb.fit(preds, labels)

    # Build bin edges: prepend 0.0 and append 1.0 around the optimal splits
    splits = optb.splits
    bins = np.concatenate([[0.0], splits, [1.0]])

    # Digitize predictions into bins (right=False so left-closed intervals)
    # Trick the last edge so that preds==1.0 falls in the final bin
    bins_dig = bins.copy()
    bins_dig[-1] = 1.0 + 1e-8
    indices = np.digitize(preds, bins_dig, right=False) - 1

    n_bins = len(bins) - 1
    bin_count = np.array([int(labels[indices == i].sum()) for i in range(n_bins)])
    bin_total = np.array([int((indices == i).sum()) for i in range(n_bins)])
    bin_preds = [preds[indices == i] for i in range(n_bins)]

    return bin_preds, bin_count, bin_total, bins


# Modification of calibration_curve function in scikit-learn
# License: BSD 3 clause
# ==================================================
'''
BSD 3-Clause License

Copyright (c) 2007-2023 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# ==================================================
def _calibration_curve(y_true, y_prob, *, pos_label=None, normalize="deprecated", n_bins=5, strategy="uniform"):
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)
    
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_data = np.zeros(bin_true.shape)
    prob_data[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = np.zeros(bin_sums.shape)
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]
    
    #prob_data = bin_true[nonzero] / bin_total[nonzero]
    #prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return prob_data[:-1], prob_pred[:-1], bin_total[:-1], bins


class TCEBinner:
    def __init__(
        self,
        siglevel: float = 0.05,
        optb_kwargs: dict | None = None,
        check_range: bool = True,
        eps_last_edge: float = 1e-8,
        boundary_tol: float = 1e-6,
        boundary_max_iter: int = 60,
        z_test_use_cc: bool = False,   # continuity correction option
    ):
        self.siglevel = siglevel
        self.check_range = check_range
        self.eps_last_edge = eps_last_edge
        self.boundary_tol = boundary_tol
        self.boundary_max_iter = boundary_max_iter
        self.z_test_use_cc = z_test_use_cc

        self.optb_kwargs = optb_kwargs or {
            "solver": "mip",
            "divergence": "neg_brier",
            "max_n_prebins": 20,
            "max_bin_size": 0.6,
            "monotonic_trend": "auto",
            "max_pvalue": 0.05,
        }

        self.optb_ = None
        self.bins_ = None
        self._boundary_cache = {}

    def fit(self, preds_train, labels_train):
        preds_train = np.asarray(preds_train, dtype=float)
        labels_train = np.asarray(labels_train)

        if self.check_range:
            if not (np.all(preds_train >= 0.0) and np.all(preds_train <= 1.0)):
                raise ValueError("Training preds out of range [0, 1].")
            if not np.all((labels_train == 0) | (labels_train == 1)):
                raise ValueError("Training labels must be 0/1.")

        optb = OptimalBinning(dtype="numerical", **self.optb_kwargs)
        optb.fit(preds_train, labels_train)

        self.bins_ = np.concatenate([[0.0], optb.splits, [1.0]])
        self.optb_ = optb
        return self

    def _digitize(self, preds):
        if self.bins_ is None:
            raise RuntimeError("Call fit(...) first.")

        preds = np.asarray(preds, dtype=float)
        if self.check_range and not (np.all(preds >= 0.0) and np.all(preds <= 1.0)):
            raise ValueError("Preds out of range [0, 1].")

        bins_dig = self.bins_.copy()
        bins_dig[-1] = 1.0 + self.eps_last_edge
        idx = np.digitize(preds, bins_dig, right=False) - 1
        return np.clip(idx, 0, len(self.bins_) - 2)

    def calibration_summary(self, preds, labels):
        preds = np.asarray(preds, dtype=float)
        labels = np.asarray(labels)

        if self.check_range and not np.all((labels == 0) | (labels == 1)):
            raise ValueError("Labels must be 0/1.")

        idx = self._digitize(preds)
        n_bins = len(self.bins_) - 1

        bin_total = np.bincount(idx, minlength=n_bins).astype(int)
        y = labels.astype(int)
        bin_count = np.bincount(idx, weights=y, minlength=n_bins).astype(int)

        bin_preds = [preds[idx == i] for i in range(n_bins)]
        return bin_preds, bin_count, bin_total, self.bins_.copy()

    # ---------- Test backends ----------

    def _pvalue_binom(self, k, n, p):
        p = float(np.clip(p, 0.0, 1.0))
        return binomtest(int(k), int(n), p=p, alternative="two-sided").pvalue

    def _find_acceptance_interval_binom_fast(self, k: int, n: int, alpha: float):
        key = (int(k), int(n), float(alpha))
        if key in self._boundary_cache:
            return self._boundary_cache[key]

        if n <= 0:
            interval = (0.0, 1.0)
            self._boundary_cache[key] = interval
            return interval

        p_hat = k / n
        pv_hat = self._pvalue_binom(k, n, p_hat)
        if pv_hat <= alpha:
            interval = (p_hat, p_hat)
            self._boundary_cache[key] = interval
            return interval

        def bisect(lo, hi, side):
            for _ in range(self.boundary_max_iter):
                mid = 0.5 * (lo + hi)
                pv = self._pvalue_binom(k, n, mid)

                if pv > alpha:
                    if side == "left":
                        hi = mid
                    else:
                        lo = mid
                else:
                    if side == "left":
                        lo = mid
                    else:
                        hi = mid

                if abs(hi - lo) < self.boundary_tol:
                    break
            return 0.5 * (lo + hi)

        # left
        if k == 0:
            p_low = 0.0
        else:
            pv0 = self._pvalue_binom(k, n, 0.0)
            p_low = 0.0 if pv0 > alpha else bisect(0.0, p_hat, "left")

        # right
        if k == n:
            p_high = 1.0
        else:
            pv1 = self._pvalue_binom(k, n, 1.0)
            p_high = 1.0 if pv1 > alpha else bisect(p_hat, 1.0, "right")

        interval = (float(p_low), float(p_high))
        self._boundary_cache[key] = interval
        return interval

    def _acceptance_interval_wilson(self, k: int, n: int, alpha: float):
        # Wilson score CI for Bernoulli proportion
        if n <= 0:
            return (0.0, 1.0)
        z = norm.ppf(1 - alpha / 2)
        phat = k / n
        denom = 1 + (z * z) / n
        center = (phat + (z * z) / (2 * n)) / denom
        half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
        return (float(max(0.0, center - half)), float(min(1.0, center + half)))

    def _acceptance_interval_clopper_pearson(self, k: int, n: int, alpha: float):
        # Exact CI from Beta distribution quantiles
        if n <= 0:
            return (0.0, 1.0)
        if k == 0:
            low = 0.0
        else:
            low = beta_dist.ppf(alpha / 2, k, n - k + 1)
        if k == n:
            high = 1.0
        else:
            high = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
        return (float(low), float(high))

    def _pvals_ztest(self, p_arr, k: int, n: int, alpha: float):
        """
        Very fast approx: two-sided z-test per p, using SE under H0: p(1-p)/n.
        (Optional continuity correction is heuristic here; you can ignore it.)
        """
        if n <= 1:
            return np.ones_like(p_arr)

        mu = k / n
        p_arr = np.asarray(p_arr, dtype=float)

        se = np.sqrt(np.maximum(p_arr * (1 - p_arr) / n, 1e-12))
        z = np.abs(mu - p_arr) / se
        # two-sided p-value
        pvals = 2 * (1 - norm.cdf(z))
        return pvals

    # ---------- Public evaluation ----------

    def tce(self, preds, labels, test: str = "binom_fast", savepath=False, ymax=None):
        """
        test in {"binom", "binom_fast", "wilson", "clopper_pearson", "ztest"}
        """
        bin_preds, bin_count, bin_total, bins = self.calibration_summary(preds, labels)
        alpha = self.siglevel

        bin_rnum = np.zeros(len(bin_count), dtype=int)

        for i in range(len(bin_count)):
            n = int(bin_total[i])
            if n == 0:
                continue
            k = int(bin_count[i])
            p_arr = bin_preds[i]

            if test == "binom":
                pvals = np.array([self._pvalue_binom(k, n, p) for p in p_arr], dtype=float)
                bin_rnum[i] = int(np.sum(pvals <= alpha))

            elif test == "binom_fast":
                p_low, p_high = self._find_acceptance_interval_binom_fast(k, n, alpha)
                bin_rnum[i] = int(np.sum((p_arr < p_low) | (p_arr > p_high)))

            elif test == "wilson":
                p_low, p_high = self._acceptance_interval_wilson(k, n, alpha)
                bin_rnum[i] = int(np.sum((p_arr < p_low) | (p_arr > p_high)))

            elif test == "clopper_pearson":
                p_low, p_high = self._acceptance_interval_clopper_pearson(k, n, alpha)
                bin_rnum[i] = int(np.sum((p_arr < p_low) | (p_arr > p_high)))

            elif test == "ztest":
                # close to what your "ttest" is trying to approximate, but more standard for proportions
                pvals = self._pvals_ztest(p_arr, k, n, alpha)
                bin_rnum[i] = int(np.sum(pvals <= alpha))

            else:
                raise ValueError(f"Unknown test={test}. Choose from binom, binom_fast, wilson, clopper_pearson, ztest.")

        if savepath is not False:
            plot_tce_diagram(bin_rnum, bin_preds, bin_count, bin_total, savepath, ymax)

        tce_value = 100.0 * np.sum(bin_rnum) / max(1, np.sum(bin_total))
        return tce_value