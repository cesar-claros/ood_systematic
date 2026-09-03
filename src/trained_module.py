import os
import copy
import numpy as np
import torch
import faiss
from torch.autograd import Variable
from torch_pca import PCA
from fd_shifts import logger
from tqdm import tqdm
from torch.nn import functional as F
from typing import Any, Callable, TypeVar, List, Tuple, Optional
import pandas as pd
from src.rc_stats import RiskCoverageStats
from src import utils
from src import scores_funcs
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
#%%
ArrayType = torch.Tensor
T = TypeVar(
    "T", Callable[[ArrayType], ArrayType], Callable[[ArrayType, ArrayType], ArrayType]
)

def _l2normalize(v, eps=1e-10):
    return v / (torch.norm(v,dim=2,keepdim=True) + eps)

#Power Iteration as SVD substitute for accleration
def power_iteration(A, iter=100):
    u = torch.FloatTensor(1, A.size(1)).normal_(0, 1).view(1,1,A.size(1)).repeat(A.size(0),1,1).to(A)
    v = torch.FloatTensor(A.size(2),1).normal_(0, 1).view(1,A.size(2),1).repeat(A.size(0),1,1).to(A)
    for _ in range(iter):
      v = _l2normalize(u.bmm(A)).transpose(1,2)
      u = _l2normalize(A.bmm(v).transpose(1,2))
    sigma = u.bmm(A).bmm(v)
    sub = sigma * u.transpose(1,2).bmm(v.transpose(1,2))
    return sub
class TrainedModule:
    """TrainedModule
    """
    def __init__(self, module, study_name:str, cf, rank_weight:bool=False, rank_feat:bool=False, ash_method:str|None=None, use_cuda=False):
        self.module = copy.deepcopy(module)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        # if use_cuda:
        self.module.to(self.device)    
        self.module.eval()
        self.study_name = study_name
        if self.study_name=='vit':
            self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name, return_model=False)
            self.module.disable_dropout()
        else:    
            self.model, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
            if hasattr(self.model.encoder, 'features'):
                self.maxpool_layers_name = [name for name,module in self.model.encoder.features.named_children() if 'MaxPool2d' in str(module)]
                self.conv_layers_name = [name for name,module in self.model.encoder.features.named_children() if 'Conv2d' in str(module)]
            else:
                # Non-sequential encoders (svhn_small_conv, resnet50): the
                # layer lists only serve the RankWeight/RankFeat/ASH
                # surgeries, which slice a .features Sequential. The plain
                # forward path (encoder(x)) never touches them.
                self.maxpool_layers_name = []
                self.conv_layers_name = []
                if rank_weight or rank_feat or (ash_method is not None):
                    raise NotImplementedError(
                        'RankWeight/RankFeat/ASH require an encoder with a '
                        '.features Sequential; unavailable for this backbone')
            if self.study_name == 'confidnet':
                self.model.encoder.disable_dropout()
                self.network = self.module.network
                self.network.encoder.disable_dropout()
            elif self.study_name in ('dg', 'devries', 'intervention', 'ce'):
                self.model.encoder.disable_dropout()
                self.network = None
        
        # if self.study_name!='vit':
        

        #     # self.module.backbone.encoder.disable_dropout()
        # elif (self.study_name == 'devries') or (self.study_name == 'dg'):
        #     self.model = self.module.model
        #     # self.module.model.encoder.disable_dropout()
        self.ext_confid_name = cf.eval.ext_confid_name
        self.query_confids = cf.eval.confidence_measures
        self.test_mcd_samples = cf.model.test_mcd_samples
        self.dataset = cf.data.dataset
        self.rank_weight = rank_weight
        self.rank_feat = rank_feat
        self.ash_method = ash_method
        # self.ct_method = ct_method

        
        if self.rank_weight:
            logger.info(f'Applying RankWeight...')
            if self.study_name == 'vit':
                weight = self.module.model.head.weight.data.clone()
                weight_svd = self.rank_weight_svd(weight)
                self.module.model.head.weight.data = weight_svd
            else:
                weight = self.model.encoder.features[int(self.conv_layers_name[-1])].weight.data.clone()# model = module.backbone
                weight_svd = self.rank_weight_svd(weight)
                self.model.encoder.features[int(self.conv_layers_name[-1])].weight.data = weight_svd

        
        if self.rank_feat:
            logger.info(f'RankFeat enabled...')
        if self.ash_method is not None:
            logger.info(f'{self.ash_method} enabled...')
        # if self.ct_method:
        #     logger.info(f'Curvature Tuning enabled...')
    #     if self.study_name == 'confidnet':
    #         self.maxpool_layers_name = [name for name,module in self.module.backbone.encoder.features.named_children() if 'MaxPool2d' in str(module)]
    #         self.conv_layers_name = [name for name,module in self.module.backbone.encoder.features.named_children() if 'Conv2d' in str(module)]
    #         if self.rank_weight:
    #             weight = self.module.backbone.encoder.features[int(self.conv_layers_name[-1])].weight.data.clone()# model = module.backbone
    #             weight_svd = self.rank_weight_svd(weight)
    #             self.module.backbone.encoder.features[int(self.conv_layers_name[-1])].weight.data = weight_svd
    #     elif (self.study_name == 'devries') or (self.study_name == 'dg'):
    #         self.maxpool_layers_name = [name for name,module in self.module.model.encoder.features.named_children() if 'MaxPool2d' in str(module)]
    #         self.conv_layers_name = [name for name,module in self.module.model.encoder.features.named_children() if 'Conv2d' in str(module)]
    #         if self.rank_weight:
    #             weight = self.module.model.encoder.features[int(self.conv_layers_name[-1])].weight.data.clone() # last conv layer
    #             weight_svd = self.rank_weight_svd(weight)
    #             self.module.model.encoder.features[int(self.conv_layers_name[-1])].weight.data = weight_svd
    #     else:
    #         raise NotImplementedError
    
    def rank_weight_svd(self, weight:ArrayType):
        weight_svd = weight
        if self.study_name=='vit':
            assert weight_svd.dim() == 2,'Weight matrix must have 2 dimensions'
        else:
            assert weight_svd.dim() == 4,'Weight matrix must have 4 dimensions'
            B, C, H, W = weight_svd.size()
            weight_svd = weight_svd.view(B, C * H * W)
        weight_sub = power_iteration(weight_svd.unsqueeze(0), iter=100)
        weight_svd = weight_svd - weight_sub.squeeze()
        if self.study_name=='vit':
            return weight_svd
        else:
            return weight_svd.view(B, C, H, W)

    def rank_feat_svd(self, x:ArrayType):
        feat1 = x.clone()
        if self.study_name=='vit':
            assert feat1.dim() == 3
        else:
            assert feat1.dim() == 4
            B, C, H, W = feat1.size()
            feat1 = feat1.view(B, C, H * W)
        u,s,v = torch.linalg.svd(feat1, full_matrices=False)
        feat1 = feat1 - s[:,0:1].unsqueeze(2)*u[:,:,0:1].bmm(v[:,0:1,:])
        if self.study_name=='vit':
            return feat1
        else:
            feat1 = feat1.view(B,C,H,W)
            return feat1

    def ash_b(self, x_act, percentile=65):
        assert x_act.dim() == 4
        assert 0 <= percentile <= 100
        x = x_act.clone()
        b, c, h, w = x.shape
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x

    def ash_p(self, x_act, percentile=65):
        assert x_act.dim() == 4
        assert 0 <= percentile <= 100
        x = x_act.clone()
        b, c, h, w = x.shape
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)
        return x

    def ash_s(self, x_act, percentile=65):
        assert x_act.dim() == 4
        assert 0 <= percentile <= 100
        x = x_act.clone()
        b, c, h, w = x.shape
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)
        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])
        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])
        return x

    def ash_rand(self, x, percentile=65, r1=0, r2=10):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        v = v.uniform_(r1, r2)
        t.zero_().scatter_(dim=1, index=i, src=v)
        return x

    def react(self, x, threshold):
        x = x.clip(max=threshold)
        return x

    def react_and_ash(self, x, clip_threshold, pruning_percentile):
        x = x.clip(max=clip_threshold)
        x = self.ash_s(x, pruning_percentile)
        return x

    def apply_ash(self, x):
        if self.ash_method.startswith('react_and_ash@'):
            fn, t, p = self.ash_method.split('@')
            return eval(f'self.{fn}')(x, float(t), int(p))
        elif self.ash_method.startswith('react@'):
            fn, t = self.ash_method.split('@')
            return eval(f'self.{fn}')(x, float(t))
        elif self.ash_method.startswith('ash'):
            fn, p = self.ash_method.split('@')
            return eval(f'self.{fn}')(x, int(p))
        return x
    
    def forward_features(self, x:ArrayType):
        if (not self.rank_feat) and (self.ash_method is None):
            encoded = self.model.encoder(x)
        else: 
            x_svd = self.model.encoder.features[:int(self.maxpool_layers_name[-1])](x)
            if self.rank_feat:
                x_svd = self.rank_feat_svd(x_svd)
            if self.ash_method is not None:
                x_svd = self.apply_ash(x_svd)
            encoded = self.model.encoder.features[int(self.maxpool_layers_name[-1]):](x_svd)
        return encoded

    def forward_features_vit(self, x:ArrayType):
        if (not self.rank_feat) and (self.ash_method is None):
            encoded = self.module.model.forward_features(x)
            return encoded
        else:
            x = self.module.model.patch_embed(x)
            cls_token = self.module.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.module.model.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.module.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.module.model.pos_drop(x + self.module.model.pos_embed)
            x_svd = self.module.model.blocks(x)
            # x_svd = self.model.encoder.features[:int(self.maxpool_layers_name[-1])](x)
            if self.rank_feat:
                x_svd = self.rank_feat_svd(x_svd)
            if self.ash_method is not None:
                x_svd = self.apply_ash(x_svd)
            #
            x = self.module.model.norm(x_svd)
            if self.dist_token is None:
                return self.module.model.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]
            # encoded = self.model.encoder.features[int(self.maxpool_layers_name[-1]):](x_svd)
        # return encoded
        # if self.study_name == 'confidnet':
        #     x_svd = self.module.backbone.encoder.features[:int(self.self.maxpool_layers_name[-1])](x)
        #     if self.rank_feat:
        #         x_svd = self.rank_feat_svd(x_svd)
        #     encoded = self.module.backbone.encoder.features[int(self.self.maxpool_layers_name[-1]):](x_svd)
        # elif (self.study_name == 'devries') or (self.study_name == 'dg'):
        #     x_svd = self.module.model.encoder.features[:int(self.self.maxpool_layers_name[-1])](x)
        #     if self.rank_feat:
        #         x_svd = self.rank_feat_svd(x_svd)
        #     encoded = self.module.model.encoder.features[int(self.self.maxpool_layers_name[-1]):](x_svd)
    
    def mcd_eval_forward(self, x:ArrayType, n_samples:int):
        if self.study_name=='vit':
            self.module.enable_dropout()
        else:
            self.model.encoder.enable_dropout()
            if self.ext_confid_name == "tcp":
                self.network.encoder.enable_dropout()
        encoded_list = []
        logits_list = []
        conf_list = []
        for _ in range(n_samples - len(logits_list)):
            # print(p)
            if self.study_name=='vit':
                z = self.forward_features_vit(x)
            else:
                z = self.forward_features(x)
            encoded_list.append(z.unsqueeze(2))
            if self.ext_confid_name == "devries":
                logits, confidence = self.model.head(z)
                confidence = torch.sigmoid(confidence).squeeze(1)
                logits_list.append(logits.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))
            elif self.ext_confid_name == "dg":
                outputs = self.model.head(z)
                logits = outputs[:, :-1].clone()
                outputs_prob = F.softmax(outputs, dim=1)
                _, reservation = outputs_prob[:, :-1], outputs_prob[:, -1].clone()
                confidence = 1 - reservation
                logits_list.append(logits.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))
            elif self.ext_confid_name == "tcp":
                logits = self.model.head(z)
                _, confidence = self.network(x)
                confidence = torch.sigmoid(confidence).squeeze(1)
                logits_list.append(logits.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))
            elif self.ext_confid_name == "maha":
                # if any("ext" in cfd for cfd in self.query_confids.test):
                #     zm = z[:, None, :] - self.module.mean.to(self.device)
                #     maha = -(torch.einsum("inj,jk,ink->in", zm, self.module.icov.to(self.device), zm))
                #     maha = maha.max(dim=1)[0].type_as(x)
                logits = self.module.model.head(z)
                maha = torch.zeros((logits.shape[0]))
                logits_list.append(logits.unsqueeze(2))
                conf_list.append(maha.unsqueeze(1))
            elif self.ext_confid_name is None:
                logits = self.model.head(z)
                confidence = F.softmax(logits, dim=1).max(dim=1).values
                logits_list.append(logits.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))

        if self.study_name=='vit':
            self.module.disable_dropout()
        else:
            self.model.encoder.disable_dropout()
            if self.ext_confid_name == "tcp":
                self.network.encoder.disable_dropout()

        return torch.cat(encoded_list, dim=2), torch.cat(logits_list, dim=2), torch.cat(conf_list, dim=1)

    
    def __call__(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        if self.study_name=='vit':
            z = self.forward_features_vit(x)
        else:
            z = self.forward_features(x)
        # z = self.forward_features(x)

        if self.ext_confid_name == "devries":
            logits, confidence = self.model.head(z)
            confidence = torch.sigmoid(confidence).squeeze(1)
        elif self.ext_confid_name == "dg":
            outputs = self.model.head(z)
            logits = outputs[:, :-1].clone()
            outputs_prob = F.softmax(outputs, dim=1)
            _, reservation = outputs_prob[:, :-1], outputs_prob[:, -1].clone()
            confidence = 1 - reservation
        elif self.ext_confid_name == "tcp":
            logits = self.model.head(z)
            _, confidence = self.network(x)
            confidence = torch.sigmoid(confidence).squeeze(1)
        elif self.ext_confid_name == "maha":
            # if any("ext" in cfd for cfd in self.query_confids.test):
            #     zm = z[:, None, :] - self.module.mean.to(self.device)
            #     maha = -(torch.einsum("inj,jk,ink->in", zm, self.module.icov.to(self.device), zm))
            #     maha = maha.max(dim=1)[0].type_as(x)
            logits = self.module.model.head(z)
            maha = torch.zeros((logits.shape[0]))
            confidence = maha
        elif self.ext_confid_name is None:
            # CE and intervention paradigms have no external confidence
            # head; maximum softmax is the recorded confidence.
            logits = self.model.head(z)
            confidence = torch.softmax(logits, dim=1).max(dim=1).values
        else:
            raise NotImplementedError

        encoded_dist = None
        logits_dist = None
        confid_dist = None
        if any("mcd" in cfd for cfd in self.query_confids.test):
            encoded_dist, logits_dist, confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        return {
            "encoded": z.detach().cpu().data,
            "logits": logits.detach().cpu().data,
            "confid": confidence.detach().cpu().data,
            "encoded_dist": encoded_dist.detach().cpu().data if logits_dist is not None else None,
            "logits_dist": logits_dist.detach().cpu().data if logits_dist is not None else None,
            "confid_dist": confid_dist.detach().cpu().data if logits_dist is not None else None,
            "labels": y.detach().cpu().data,
        }

# %%
