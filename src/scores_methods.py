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

#%%
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

def cov(tensor:ArrayType, centered:bool=False, rowvar:bool=True, bias:bool=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor.clone()
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    if not centered:
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()
#%%
# RankWeight
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
            self.maxpool_layers_name = [name for name,module in self.model.encoder.features.named_children() if 'MaxPool2d' in str(module)]
            self.conv_layers_name = [name for name,module in self.model.encoder.features.named_children() if 'Conv2d' in str(module)]
            if self.study_name == 'confidnet':
                self.model.encoder.disable_dropout()
                self.network = self.module.network
                self.network.encoder.disable_dropout()
            elif (self.study_name == 'dg') or (self.study_name == 'devries'):
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
class ClassTypicalMatching:

    def __init__(self, module, study_name:str, cf, mode:str='global', alpha:float = 1.0):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        # self.ext_confid_name = self.cf.eval.ext_confid_name
        self.study_name = study_name
        # self.query_confids = self.cf.eval.confidence_measures
        # self.test_mcd_samples = self.cf.model.test_mcd_samples
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        self.mode = mode
        self.alpha = alpha
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.class_means = None
    
    def compute_CTM_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType,
                                    only_correct: bool = False ):
        logger.info("Class Typical Matching: Fitting parameters...")
        labels_train = labels_train.clone()
        if self.mode == 'global':
            self.class_means = []
            activations_train = activations_train.clone()
            for c in range(self.num_classes):
                activations_per_class_tensor = activations_train[labels_train==c]
                if only_correct:
                    labels_per_class_tensor = labels_train[labels_train==c]
                    logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
                else:
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
        elif self.mode == 'class':
            self.class_means = []
            for c in range(self.num_classes):
                activations_per_class_tensor = activations_train[c][labels_train==c]
                if only_correct:
                    labels_per_class_tensor = labels_train[labels_train==c]
                    logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
                else:
                    self.class_means.append( activations_per_class_tensor.mean(dim=0) )
    
    def get_scores( self, backprojected_activations_eval: ArrayType|List[ArrayType], similarity:str='weight', batch_size=25):
        logger.info(f"Class Typical Matching: Computing scores using similarity={similarity} and mode={self.mode}...")
        
        if self.mode == 'global':
            X_back_projected = backprojected_activations_eval
            if X_back_projected.dim()==2:
                feat_data_loader = torch.utils.data.DataLoader(X_back_projected, batch_size=batch_size, shuffle=False)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    print("Warning: CUDA not available, using CPU. This will be slower.")
                if similarity == 'weight':
                    weights_expanded = self.w.T.unsqueeze(0).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append(F.cosine_similarity(x, weights_expanded, dim=1).amax(dim=1))
                elif similarity == 'mean' and self.class_means is not None:
                    class_means_ = torch.stack(self.class_means, dim=0).T.unsqueeze(0).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append(F.cosine_similarity(x, class_means_, dim=1).amax(dim=1))
                cosine_similarity_eval = torch.cat(sim, dim=0).cpu()
                return cosine_similarity_eval
            elif X_back_projected.dim()==3:
                logger.info(f"Class Typical Matching: Operating on distribution of activations...")
                feat_data_loader = torch.utils.data.DataLoader(X_back_projected, batch_size=batch_size, shuffle=False)
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    print("Warning: CUDA not available, using CPU. This will be slower.")
                
                if similarity == 'weight':
                    weights_expanded = self.w.T.unsqueeze(0).unsqueeze(3).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append( F.cosine_similarity(x, weights_expanded, dim=1).amax(dim=1) )
                elif similarity == 'mean' and self.class_means is not None:
                    class_means_ = torch.stack(self.class_means, dim=0).T.unsqueeze(0).unsqueeze(3).contiguous().to(device)
                    sim = []
                    for x in feat_data_loader:
                        x = x.unsqueeze(2).contiguous()
                        x = x.to(device)
                        sim.append( F.cosine_similarity(x, class_means_, dim=1).amax(dim=1) )
                
                cosine_similarity_eval = torch.cat(sim, dim=0).cpu()
                return cosine_similarity_eval.mean(dim=1)

        elif self.mode == 'class':
            self.n_components = []
            cosine_similarity_eval_list = []
            X_back_projected_list = backprojected_activations_eval
            for c in range(self.num_classes):
                X_back_projected = X_back_projected_list[c]
                if similarity == 'weight': 
                    cosine_similarity_eval_list.append(F.cosine_similarity(X_back_projected,self.w.T[:,c]))
                elif similarity == 'mean' and self.class_means is not None:
                    cosine_similarity_eval_list.append(F.cosine_similarity(X_back_projected,self.class_means[c]))
            cosine_similarity_eval = torch.stack(cosine_similarity_eval_list, dim=1)
        return cosine_similarity_eval.amax(dim=1)

    def save_params(self, path:str|None=None, filename:str='CTM_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.class_means is not None, 'Class means constant have not been computed...'
        # assert self.alpha is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'alpha': self.alpha,
                        'class_means': self.class_means,
                        # 'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'CTM: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='CTM_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'CTM: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.alpha = params_dict['alpha']
        self.class_means = params_dict['class_means']

# %%
class GeometricComplexity:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = cf.data.num_classes
        self.study_name = study_name
        self.model, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name,return_model=True)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.encoder = self.model.encoder
        self.h_grad_x = None
        self.g_grad_x = None
        # self.fc = torch.nn.Linear(*self.w.shape[::-1])
        # self.fc.weight.data[...] = self.w
        # self.fc.bias.data[...] = self.b
        # self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
    
    def _rademacher(self, shape, device, dtype):
        # ±1 with prob 1/2, returned as dtype
        return (torch.randint(0, 2, shape, device=device, dtype=torch.int8).to(dtype) * 2 - 1)

    @torch.no_grad()
    def _check_shapes(self, encoder, W_L, x):
        h = encoder(x)
        assert h.ndim == 2, f"encoder(x) must be [B, D], got {tuple(h.shape)}"
        assert W_L.ndim == 2, f"W_L must be [C, D], got {tuple(W_L.shape)}"
        assert W_L.shape[1] == h.shape[1], f"W_L.shape[1]={W_L.shape[1]} must equal D={h.shape[1]}"

    def get_grad_frob_norms(
        self,
        # encoder,
        # W_L: torch.Tensor,                  # [C, D]
        x: torch.Tensor,                    # [B, ...]
        n_probes_h: int = 8,
        n_probes_logits: int = 8,
        rademacher: bool = True,
        recompute_forward: bool = False,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        frob_h:      [B] estimate of || d h(x) / d x ||_F
        frob_logits: [B] estimate of || d (W_L h(x)+b) / d x ||_F
        """
        self.encoder.eval()
        W_L = self.w.to(device=x.device)

        # Quick shape sanity check (runs a forward once, no grad)
        self._check_shapes(self.encoder, W_L, x)

        x = x.requires_grad_(True)
        B = x.shape[0]

        acc_h = torch.zeros(B, device=x.device, dtype=x.dtype)
        acc_g = torch.zeros(B, device=x.device, dtype=x.dtype)

        if not recompute_forward:
            h = self.encoder(x)  # [B, D]
            D = h.shape[1]
            C = W_L.shape[0]

            total_grads = n_probes_h + n_probes_logits
            done = 0

            # --- Hutchinson for h(x): E_v ||J_h^T v||^2
            for _ in range(n_probes_h):
                v = self._rademacher((B, D), x.device, h.dtype) if rademacher else torch.randn(B, D, device=x.device, dtype=h.dtype)
                s = (h * v).sum()  # sum_b <v_b, h_b>
                done += 1
                retain = (done < total_grads)
                (gx,) = torch.autograd.grad(s, x, retain_graph=retain, create_graph=create_graph)
                acc_h += gx.reshape(B, -1).pow(2).sum(dim=1)

            # --- Hutchinson for logits g(x): E_u ||J_g^T u||^2, with J_g^T u = J_h^T (u W_L)
            for _ in range(n_probes_logits):
                u = self._rademacher((B, C), x.device, h.dtype) if rademacher else torch.randn(B, C, device=x.device, dtype=h.dtype)
                v = u @ W_L  # [B, D]  (this is W_L^T u, in batch row-vector convention)
                s = (h * v).sum()
                done += 1
                retain = (done < total_grads)
                (gx,) = torch.autograd.grad(s, x, retain_graph=retain, create_graph=create_graph)
                acc_g += gx.reshape(B, -1).pow(2).sum(dim=1)

        else:
            # Recompute h each probe (less graph retention, more forward passes)
            for _ in range(n_probes_h):
                h = self.encoder(x)
                D = h.shape[1]
                v = self._rademacher((B, D), x.device, h.dtype) if rademacher else torch.randn(B, D, device=x.device, dtype=h.dtype)
                s = (h * v).sum()
                (gx,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=create_graph)
                acc_h += gx.reshape(B, -1).pow(2).sum(dim=1)

            C = W_L.shape[0]
            for _ in range(n_probes_logits):
                h = self.encoder(x)
                u = self._rademacher((B, C), x.device, h.dtype) if rademacher else torch.randn(B, C, device=x.device, dtype=h.dtype)
                v = u @ W_L
                s = (h * v).sum()
                (gx,) = torch.autograd.grad(s, x, retain_graph=False, create_graph=create_graph)
                acc_g += gx.reshape(B, -1).pow(2).sum(dim=1)

        frob_h = (acc_h / max(n_probes_h, 1))
        frob_g = (acc_g / max(n_probes_logits, 1))

        if not create_graph:
            frob_h = frob_h.detach()
            frob_g = frob_g.detach()

        return frob_h, frob_g
    
    def compute_GC_params( self, datamodule ):
        logger.info('Geometric Complexity: Computing gradients...')
        dataloaders = datamodule.train_dataloader()
        frob_norm_list_set = [ self.get_grad_frob_norms( batch[0] ) for i,batch in enumerate(tqdm(dataloaders, position=0, leave=True)) ]
        h_grad_x = torch.concat([h  for h,_ in frob_norm_list_set])
        g_grad_x = torch.concat([g  for _,g in frob_norm_list_set])
        self.h_grad_x = h_grad_x.mean()
        self.g_grad_x = g_grad_x.mean()
    
    def save_params(self, path:str|None=None, filename:str='GC_params'):
        assert self.h_grad_x is not None, 'h_grad_x has not been computed'
        assert self.g_grad_x is not None, 'h_grad_x has not been computed'
        params_dict = {
                    'h_grad_x': self.h_grad_x,
                    'g_grad_x': self.g_grad_x
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Saving GC parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='GC_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Loading GC params from {path}')
        params_dict = torch.load(path)
        self.h_grad_x = params_dict['h_grad_x']
        self.g_grad_x = params_dict['g_grad_x']
        # logger.info(f'temperature={self.temperature:.3f}')


# %%
class GradNorm:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.num_classes = cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.fc = torch.nn.Linear(*self.w.shape[::-1])
        self.fc.weight.data[...] = self.w
        self.fc.bias.data[...] = self.b
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def get_scores( self, activations_eval: ArrayType, temperature:float=1.0, use_cuda=True ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.fc.to(self.device)
        self.logsoftmax.to(self.device)
        logger.info('GradNorm: Computing scores...')
        confs = []
        activations_eval = activations_eval.clone()
        for x in tqdm(activations_eval):
            targets = torch.ones((1, self.num_classes))
            input_var = Variable(x, requires_grad=True)
            targets = targets.to(self.device)
            input_var = input_var.to(self.device)    
            self.fc.zero_grad()
            logits = self.fc(input_var[None])
            # if self.study_name == 'dg':
            #     logits = logits[:,:-1]
            logits = logits/temperature
            loss = torch.mean(
                torch.sum( -targets * self.logsoftmax( logits ), dim=-1)
                )
            loss.backward()
            layer_grad_norm = torch.sum( torch.abs(self.fc.weight.grad.data) ).detach().cpu()
            confs.append(layer_grad_norm)
        return torch.Tensor(confs)

#%%

class TemperatureScaling:
    def __init__(self,cf):
        self.cf = cf
        self.temperature = None

    def compute_temperature(self, 
                    logits_val: ArrayType, 
                    labels_val: ArrayType):
        logits_val = logits_val.clone()
        labels_val = labels_val.clone()
        logger.info("Fitting temperature to logits in the validation set...")
        self.temperature = torch.ones(1).requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=2000)

        def _eval():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits_val / self.temperature, labels_val)
            loss.backward()
            return loss

        optimizer.step(_eval)  # type: ignore

        self.temperature = self.temperature.item()
        logger.info(f'Temperature={self.temperature:.3f}')

    def get_scaled_softmax(self, logits_eval: ArrayType) -> ArrayType:
        # import torch
        logger.info('Temperature scale: Computing scores...')
        logits_eval = logits_eval.clone()
        softmax_rescaled = torch.softmax(logits_eval / self.temperature, dim=1, dtype=torch.float64)
        return softmax_rescaled
    
    def save_params(self, path:str|None=None, filename:str='Temperature_params'):
        assert self.temperature is not None, 'Temperature has not been computed'
        params_dict = {
                    'temperature': self.temperature,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Saving temperature scaling parameter in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='Temperature_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        params_dict = torch.load(path)
        self.temperature = params_dict['temperature']
        logger.info(f'Loading temperature scale from {path}')
        logger.info(f'temperature={self.temperature:.3f}')

# %%
class MahalanobisDistance:

    def __init__(self,cf):
        self.cf = cf
        self.precision = None
        self.unique_labels = None
        self.means = None

    def compute_MahaDist_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType, ):
        logger.info("MahalanobisDistance: Fitting parameters...")
        activations_train = activations_train.clone()
        labels_train = labels_train.clone()
        self.unique_labels = torch.unique(labels_train)
        self.means = [
            activations_train[ labels_train == i ].mean(dim=0) for i in self.unique_labels 
            ]
        class_centered_features = torch.concat(
            [ activations_train[labels_train == i] - self.means[i] for i in self.unique_labels ],
            dim=0)
        covariance = cov(class_centered_features, centered=True, rowvar=False)
        try:
            self.precision = torch.linalg.pinv(covariance, hermitian=True, rtol=1e-6)
        except RuntimeError as e:
            if "The algorithm failed to converge" in str(e):
                print("Caught a convergence error with torch.linalg.eigh:", e)
                self.precision = torch.linalg.pinv(covariance.double() + 1e-6*torch.eye(covariance.shape[0]), hermitian=True, rtol=1e-6)
                self.precision = self.precision.float()
            else:
                raise e

        if torch.isnan(self.precision).any() or torch.isinf(self.precision).any():
            self.precision = torch.zeros_like(covariance)
    
    def save_params(self, path:str|None=None, filename:str='MahaDist_params'):
        assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.means is not None, 'Class means constant have not been computed...'
        assert self.unique_labels is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'precision': self.precision,
                        'means': self.means,
                        'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'MahalanobisDistance: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='MahaDist_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'MahalanobisDistance: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.precision = params_dict['precision']
        self.means = params_dict['means']
        self.unique_labels = params_dict['unique_labels']
    
    def get_scores( self, activations_eval: ArrayType, batch_size=1000 ):
        logger.info('MahalanobisDistance: Computing scores...')
        # activations_eval = activations_eval.clone()
        means_ = torch.stack(self.means, dim=0).T.unsqueeze(0).contiguous()
        class_centered_evaluations_ = (activations_eval.unsqueeze(2) - means_).contiguous()
        precision_ = self.precision.unsqueeze(0).contiguous()
        feat_data_loader = torch.utils.data.DataLoader(class_centered_evaluations_, batch_size=batch_size, shuffle=False)
        # scores_eval = (-1 * class_centered_evaluations_ * torch.matmul(precision_,class_centered_evaluations_)).sum(dim=1).amax(dim=1)
        # Check if CUDA (GPU support) is available
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Warning: CUDA not available, using CPU. This will be slower.")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision_ = precision_.to(device)
        score_list = []    
        for x in feat_data_loader:
            x = x.to(device)
            score_list.append( (-1 * x * torch.matmul(precision_,x)).sum(dim=1).amax(dim=1) )
        scores_eval = torch.cat(score_list, dim=0).cpu()
        return scores_eval
#%%
class KLMatching:

    def __init__(self,cf):
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        # self.precision = None
        # self.unique_labels = None
        self.means = None
    
    def pairwise_kl_divergence(self, p: ArrayType, q: ArrayType) -> ArrayType:
        """
        Compute the pairwise KL divergence between two sets of distributions.

        Args:
            p (torch.Tensor): A tensor of shape (N, D), where N is the batch size and D is the dimensionality of the distributions.
            q (torch.Tensor): A tensor of shape (M, D), where M is the batch size and D is the dimensionality of the distributions.

        Returns:
            torch.Tensor: A tensor of shape (N, M) containing the pairwise KL divergences.
        """

        N, D = p.shape
        M, _ = q.shape

        p = p.unsqueeze(1).expand(N, M, D)
        q = q.unsqueeze(0).expand(N, M, D)

        kl = torch.sum( torch.where(p != 0, p * torch.log(p / q), 0), dim=-1)

        return kl

    def min_pairwise_kl_divergence(self, p: ArrayType, q: ArrayType) -> ArrayType:
        """
        Find the argmin of the pairwise KL divergence between two sets of distributions.

        Args:
            p (torch.Tensor): A tensor of shape (N, D), where N is the batch size and D is the dimensionality of the distributions.
            q (torch.Tensor): A tensor of shape (M, D), where M is the batch size and D is the dimensionality of the distributions.

        Returns:
            torch.Tensor: A tensor of shape (N,) containing the indices of the minimum KL divergence for each distribution in p.
        """

        kl = self.pairwise_kl_divergence(p, q)
        return torch.min(kl, dim=1).values

    def compute_KLMatching_params(self, softmax_train: ArrayType,):
        logger.info("KLMatching: Fitting parameters...")
        softmax_train = softmax_train.clone()
        # n, n_classes = softmax_train.shape
        predicted_labels = softmax_train.max(dim=1).indices
        self.means = torch.vstack([
            softmax_train[ predicted_labels == i ].mean(dim=0) for i in range(self.num_classes) 
            ])
    
    def save_params(self, path:str|None=None, filename:str='KLMatching_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.means is not None, 'Class means constant have not been computed...'
        # assert self.unique_labels is not None, 'Unique labels have not been computed...'
        params_dict = {
                        # 'precision': self.precision,
                        'means': self.means,
                        # 'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'KLMatching: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='KLMatching_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'KLMatching: Loading parameters from {path}')
        params_dict = torch.load(path)
        # self.precision = params_dict['precision']
        self.means = params_dict['means']
        # self.unique_labels = params_dict['unique_labels']
    
    def get_scores( self, softmax_eval: ArrayType ) -> ArrayType:
        logger.info(f'KLMatching: Computing scores...')
        softmax_eval = softmax_eval.clone()
        scores_eval = -self.min_pairwise_kl_divergence(softmax_eval, self.means)
        return scores_eval

# %%
class ViMScore:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.ext_confid_name = self.cf.eval.ext_confid_name
        self.study_name = study_name
        self.query_confids = self.cf.eval.confidence_measures
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        
        self.u = -torch.linalg.pinv(self.w) @ self.b
        self.residual = None
        self.alpha = None
    
    def compute_ViM_params(self, activations_train: ArrayType,
                                D: int|None = None ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        logger.info('ViM Score: Fitting parameters...')
        activations_train = activations_train.clone()
        logit_train = activations_train @ self.w.T + self.b
        if D is None:
            if activations_train.shape[1] >= 2048:
                self.D = 1000
            elif activations_train.shape[1] >= 768:
                self.D = 512
            else:
                self.D = activations_train.shape[1] // 2
        else:
            self.D = D
        
        X_train = (activations_train - self.u)
        covariance = cov(X_train, centered=True, rowvar=False)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        sorted_indices = torch.argsort(eigenvalues.real, descending=True)
        self.residual = eigenvectors[:, sorted_indices[self.D:]].real
        virtual_logit_norm_train = torch.linalg.norm(X_train @ self.residual, dim=1)
        self.alpha = logit_train.max(dim=1).values.mean() / virtual_logit_norm_train.mean()
    
    def save_params(self, path:str|None=None, filename:str='ViM_params'):
        assert self.residual is not None, 'Residual matrix has not been computed'
        assert self.alpha is not None, 'alpha constant has not been computed'
        params_dict = {
                    'residual': self.residual,
                    'alpha': self.alpha,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'ViM Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='ViM_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'ViM Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.residual = params_dict['residual']
        self.alpha = params_dict['alpha']

    
    def get_scores( self, activations_eval: ArrayType ):
        # logger.info('ViM: Computing scores')
        logger.info(f'ViM Score: Computing scores...')
        activations_eval = activations_eval.clone()
        X_eval = (activations_eval - self.u)    
        logit_eval = activations_eval @ self.w.T + self.b

        virtual_logit_eval = torch.linalg.norm(X_eval @ self.residual, dim=1) * self.alpha
        energy_eval = torch.logsumexp(logit_eval, dim=1)
        scores_eval = -virtual_logit_eval + energy_eval
        
        return scores_eval

# %%
class ResidualScore:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.ext_confid_name = self.cf.eval.ext_confid_name
        self.study_name = study_name
        self.query_confids = self.cf.eval.confidence_measures
        self.test_mcd_samples = self.cf.model.test_mcd_samples
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        
        self.u = -torch.linalg.pinv(self.w) @ self.b
        self.residual = None
        self.alpha = None
    
    def compute_Residual_params(self, activations_train: ArrayType,
                                D: int|None = None ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        logger.info('Residual Score: Fitting parameters...')
        activations_train = activations_train.clone()
        logit_train = activations_train @ self.w.T + self.b
        if D is None:
            if activations_train.shape[1] >= 2048:
                self.D = 1000
            elif activations_train.shape[1] >= 768:
                self.D = 512
            else:
                self.D = activations_train.shape[1] // 2
        else:
            self.D = D
        
        X_train = (activations_train - self.u)
        covariance = cov(X_train, centered=True, rowvar=False)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        sorted_indices = torch.argsort(eigenvalues.real, descending=True)
        self.residual = eigenvectors[:, sorted_indices[self.D:]].real
    
    def save_params(self, path:str|None=None, filename:str='Residual_params'):
        assert self.residual is not None, 'Residual matrix has not been computed'
        params_dict = {
                    'residual': self.residual,
                    # 'alpha': self.alpha,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Residual Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='Residual_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Residual Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.residual = params_dict['residual']
    
    def get_scores( self, activations_eval: ArrayType ):
        logger.info(f'Residual Score: Computing scores...')
        activations_eval = activations_eval.clone()
        X_eval = (activations_eval - self.u)    
        logit_eval = activations_eval @ self.w.T + self.b

        scores_eval = -torch.linalg.norm(X_eval @ self.residual, dim=1)
        
        return scores_eval
#%%
class TorchStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    def fit(self, x, threshold=1.0):
        x = x.clone()
        self.mean_ = x.mean(0, keepdim=True)
        self.std_ = x.std(0, unbiased=False, keepdim=True)
        condition = self.std_< threshold
        if torch.any(condition):
            logger.info(f'Standard deviation of {condition.sum()} variables is less than {threshold} with average value={self.std_.mean():.4f}. Only centering is applied...')
            # self.mean_ = torch.zeros_like(self.mean_)
            self.std_ = torch.ones_like(self.std_)
    def transform(self, x, tol=1e-12):
        assert self.mean_ is not None, 'Mean has not been computed...'
        assert self.std_ is not None, 'Standard deviation has not been computed...' 
        x = x.clone()
        x -= self.mean_
        # x /= (self.std_ + tol)
        return x
    def inverse_transform(self, x, tol=1e-12):
        assert self.mean_ is not None, 'Mean has not been computed...'
        assert self.std_ is not None, 'Standard deviation has not been computed...' 
        x = x.clone()
        # x *= (self.std_ + tol)
        x += self.mean_
        return x

#%%
class NeCo:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        self.pca_estimator = None
        self.tss = None

    def compute_NeCo_params(self, activations_train: ArrayType,
                                D: int|None = None ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        # Scaling
        logger.info('NeCo Score: Fitting parameters...')
        activations_train = activations_train.clone()
        dimensions = activations_train.shape[1]
        self.tss = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
        self.tss.fit(activations_train)
        self.activations_scaled = self.tss.transform(activations_train)
        # Principal Compoment Analysis 
        self.pca_estimator = PCA(n_components=None, svd_solver='full')
        self.pca_estimator.fit(self.activations_scaled)


    def get_scores( self, activations_eval: ArrayType, neco_dim:int|None=100):
        # logger.info('ViM: Computing scores')
        logger.info(f'NeCo Score: Computing scores...')
        activations_eval = activations_eval.clone()
        if self.cf.model.network.backbone is None:
            if 'ViT' in self.cf.model.network.name:
                X_eval = activations_eval
            else:
                X_eval = self.tss.transform(activations_eval)
        else:
            if 'ViT' in self.cf.model.network.backbone:
                X_eval = activations_eval
            else:
                X_eval = self.tss.transform(activations_eval)
        X_projected_full = self.pca_estimator.transform(X_eval)
        X_projected_reduced = X_projected_full[:,:neco_dim]
        logit_eval = activations_eval @ self.w.T + self.b
        logit_eval_max = logit_eval.max(dim=-1).values
        confs = []
        # activations_eval = activations_eval
        for i in tqdm(range(activations_eval.shape[0])):
            norm_full = torch.linalg.norm(X_eval[i, :])
            norm_reduced = torch.linalg.norm(X_projected_reduced[i, :])
            score = norm_reduced/norm_full
            confs.append(score)
        scores_eval = torch.Tensor(confs)
        if self.cf.model.network.backbone is None:
            if 'resnet' not in self.cf.model.network.name:
                scores_eval = scores_eval*logit_eval_max
        else:
            if 'resnet' not in self.cf.model.network.backbone:
                scores_eval = scores_eval*logit_eval_max
        
        return scores_eval

    def save_params(self, path:str|None=None, filename:str='NeCo_params'):
        assert self.tss is not None, 'Residual matrix has not been computed'
        assert self.pca_estimator is not None, 'alpha constant has not been computed'
        params_dict = {
                    'tss': self.tss,
                    'pca_estimator': self.pca_estimator,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'NeCo Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='NeCo_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'NeCo Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.tss = params_dict['tss']
        self.pca_estimator = params_dict['pca_estimator']


#%%
class ProjectionFiltering:

    def __init__(self, module, study_name:str, cf, mode:str='global'):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = self.cf.data.num_classes 
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.mode = mode
        self.condition_transform = ('ViT' in [self.cf.model.network.name, self.cf.model.network.backbone]) 
        logger.info('Using non-standardized inputs' if self.condition_transform else 'Standardizing inputs...')
        self.pca_estimator = None
        self.tss = None
        self.variance_explained = None
        

    def compute_ProjectionFiltering_params(self, activations_train: ArrayType,
                                labels:ArrayType|None = None, only_correct: bool = False,
                                variance_explained:float|None=0.90, ):
                                # last_layer: tuple[npt.NDArray[Any], ...],
        if self.variance_explained is None:
            logger.info(f'Projection Filtering: Setting minimum explained variance to {variance_explained}...')
            self.variance_explained = variance_explained
            assert 0<self.variance_explained<=1, f'Required explained variance (variance_explained={variance_explained}) should be between 0 and 1.'

        logger.info(f'Projection Filtering with {self.mode} mode: Fitting parameters...')
        activations_train = activations_train.clone()
        # dimensions = activations_train.shape[1]
        
        if 'global' in self.mode:
            if only_correct:
                self.labels = labels.clone() if labels is not None else None
                assert self.labels is not None, "Labels have not been provided..." 
                logits_train = activations_train@self.w.T + self.b
                softmax_train = F.softmax(logits_train, dim=1, dtype=torch.float64)
                predictions = softmax_train.max(dim=1).indices
                activations_train = activations_train[predictions==self.labels]
            self.tss = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
            self.tss.fit(activations_train)
            self.activations_scaled = self.tss.transform(activations_train) 
            # Principal Compoment Analysis 
            self.pca_estimator = PCA(n_components=None, svd_solver='full')
            self.pca_estimator.fit(self.activations_scaled)
        elif 'class' in self.mode:
            assert (labels is not None) and (len(labels)==len(activations_train)) , f'Labels are required to compute class subspaces.\n N_labels={len(labels)}, N_activations={len(activations_train)}'
            self.labels = labels.clone()
            # self.classes = self.labels.unique().to(int).numpy()
            self.pca_estimator = []
            self.tss = []
            for c in range(self.num_classes):
                labels_per_class_tensor = self.labels[self.labels==c]
                activations_per_class_tensor = activations_train[self.labels==c]
                logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                if only_correct:
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The PCA for this class uses all the activations.')
                tss_class = TorchStandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
                tss_class.fit(activations_per_class_tensor)
                # activations_per_class_tensor_scaled = activations_per_class_tensor if self.condition_transform else tss_class.transform(activations_per_class_tensor)
                activations_per_class_tensor_scaled = tss_class.transform(activations_per_class_tensor)
                pca_estimator_sigma_W_class = PCA(n_components=None, svd_solver='full')
                pca_estimator_sigma_W_class.fit(activations_per_class_tensor_scaled)
                self.tss.append(tss_class)
                self.pca_estimator.append(pca_estimator_sigma_W_class)

    def get_projection(self, inputs:ArrayType, components:ArrayType, d:int|None=None, proj_mode:str='projection'):
        # self.proj_mode = proj_mode
        D = components.shape[0]
        if d is None:
            d = D
        elif d==0:
            logger.info(f'Reduced dimension cannot be d={d}. Reassign d=1...')
            d = 1
        # print(f'The number of components (d={d}) cannot be bigger that the dimensionality of the input (D={D}).')
        assert (d<=D) and (d>0) 
        components = (
                            components.to(torch.float16)
                            if inputs.dtype == torch.float16
                            else components
                    )
        components = components[:d,:] # = RankFeat?
        # components = components[:d,:] # dimensionality reduction
        if proj_mode == 'projection':
            return inputs @ components.T
        elif proj_mode == 'back-projection':
            return inputs @ components.T @ components
        else:
            raise ValueError(f"Unknown projection mode {proj_mode}")

    def get_backprojection(self, activations_eval: ArrayType,  ):
        
        logger.info(f'Projection Filtering: Computing scores...')
        # activations_eval = activations_eval.clone()
        # condition = 'ViT' in [self.cf.model.network.name, self.cf.model.network.backbone]
        if self.mode == 'global':
            # X_eval = activations_eval if self.condition_transform else self.tss.transform(activations_eval)
            X_eval = self.tss.transform(activations_eval)
            self.n_components = (self.pca_estimator.explained_variance_ratio_.cumsum(0)<=self.variance_explained).sum().item() + 1
            X_back_projected = self.get_projection(X_eval, self.pca_estimator.components_, d=self.n_components, proj_mode='back-projection')
            # X_back_projected = X_back_projected if self.condition_transform else self.tss.inverse_transform(X_back_projected)
            X_back_projected = self.tss.inverse_transform(X_back_projected)
            return X_back_projected
        elif self.mode == 'class':
            self.n_components = []
            X_back_projected_list = []
            for c in range(self.num_classes):
                # X_eval = activations_eval if self.condition_transform else self.tss[c].transform(activations_eval)
                X_eval = self.tss[c].transform(activations_eval)
                n_components_class = (self.pca_estimator[c].explained_variance_ratio_.cumsum(0)<=self.variance_explained).sum().item() + 1
                X_back_projected = self.get_projection(X_eval, self.pca_estimator[c].components_, d=n_components_class, proj_mode='back-projection')
                # X_back_projected = X_back_projected if self.condition_transform else self.tss[c].inverse_transform(X_back_projected)
                X_back_projected = self.tss[c].inverse_transform(X_back_projected)
                X_back_projected_list.append( X_back_projected )
                self.n_components.append( n_components_class )
            return X_back_projected_list
    
    def get_combined_backprojection(self, X_back_projected_list:List, combine:str|None=None, preds:ArrayType|None=None):
        assert self.mode == 'class', f'Combined backrpojection not defined for mode={self.mode}'
        assert isinstance(combine,str) , f'Combine has to be a string'
        N = X_back_projected_list[0].shape[0]
        if combine == 'prediction':
            X_back_projected = torch.vstack([X_back_projected_list[preds[t]][t] for t in range(N)])
        elif combine == 'average':
            # Backprojections for class averaged
            X_back_projected = []
            for t in range(N):
                avg_sampled = []
                for c in range(self.num_classes):
                    avg_sampled.append(X_back_projected_list[c][t])
                X_back_projected.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
            X_back_projected = torch.stack(X_back_projected, dim=0)
        logits_eval = X_back_projected @ self.w.T + self.b
        return X_back_projected, logits_eval

    def get_logits( self, activations_eval: ArrayType):
        logger.info(f'Projection Filtering: Computing logits...')
        activations_eval = activations_eval.clone()
        if self.mode == 'global':
            X_back_projected = self.get_backprojection(activations_eval)
            logits_eval = X_back_projected @ self.w.T + self.b
        elif self.mode == 'class':
            logit_eval_list = []
            X_back_projected = self.get_backprojection(activations_eval)
            for c in range(self.num_classes):
                logit_class_eval = X_back_projected[c] @ self.w.T[:,c] + self.b[c]
                logit_eval_list.append(logit_class_eval)
            logits_eval = torch.stack(logit_eval_list, dim=1)
        return logits_eval
    
    def get_scores( self, activations_eval:ArrayType, 
                            predictions_eval:ArrayType|None=None,
                            X_back_projected_eval:ArrayType|List|None=None,):
        logger.info(f'Projection Filtering: Computing scores...')
        if isinstance(activations_eval,ArrayType):
            activations_eval = activations_eval.clone()
        if self.mode == 'global':
            if X_back_projected_eval is None:
                X_back_projected = self.get_backprojection(activations_eval)
            else:
                X_back_projected = X_back_projected_eval
            scores = -1 * torch.norm(activations_eval - X_back_projected, p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)
        elif self.mode == 'class':
            scores_list = []
            if X_back_projected_eval is None:
                X_back_projected = self.get_backprojection(activations_eval)
            else:
                X_back_projected = X_back_projected_eval
            if isinstance(X_back_projected,List):            # X_back_projected = self.get_backprojection(activations_eval)
                for c in range(self.num_classes):
                    score = -1 * torch.norm(activations_eval - X_back_projected[c], p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)
                    scores_list.append(score)
                scores = torch.stack(scores_list, dim=1)
                if predictions_eval is None:
                    scores = scores.amax(dim=1) # pick best reconstruction
                else:
                    scores = torch.gather(scores, 1, predictions_eval.reshape(-1,1)).squeeze() # pick reconstruction guided by prediction
            else:
                scores = -1 * torch.norm(activations_eval - X_back_projected, p=2, dim=1)/torch.norm(activations_eval, p=2, dim=1)

        return scores
    
    def tune_hyperparameters( self, activations_train: ArrayType,
                                        activations_val: ArrayType, 
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType = None,
                                        only_correct: bool = False, 
                                        var_bounds:tuple=(0.85,0.99),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Projection Filtering: Tuning variance explained required to minimize AUGRC using the validation set...')
        activations_val = activations_val.clone()
        self.compute_ProjectionFiltering_params(activations_train, labels=labels_train, only_correct=only_correct)
        def _get_metric(explained_variance):
            self.variance_explained = explained_variance
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={'explained_variance': var_bounds,},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.variance_explained = best_params['explained_variance'] 
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
        

    def save_params(self, path:str|None=None, filename:str='ProjectionFiltering_params'):
        assert self.tss is not None, 'Standardizer has not been computed'
        assert self.pca_estimator is not None, 'PCA estimator has not been computed'
        assert self.variance_explained is not None, 'Variance explained has not been computed'
        params_dict = {
                        'tss': self.tss,
                        'pca_estimator': self.pca_estimator,
                        'variance_explained': self.variance_explained, }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'ProjectionFiltering Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='ProjectionFiltering_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'ProjectionFiltering Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.tss = params_dict['tss']
        self.pca_estimator = params_dict['pca_estimator']
        self.variance_explained = params_dict['variance_explained']


#%%
class pNML:
# Predictive Normalized Maximum Likelihood

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.p_parallel = None
        self.p_bot = None
        # self.probs = None
    
    def softmax(self, logits: ArrayType) -> ArrayType:
        """Compute softmax values for each sets of scores in x."""
        logits = logits.clone()
        logits_max = torch.max(logits, dim=1, keepdims=True).values
        e_x = torch.exp(logits - logits_max)
        return e_x / e_x.sum(dim=1, keepdims=True)
    
    def transform_activations(self, activations: ArrayType) -> ArrayType:
        activations = activations.clone()
        n, m = activations.shape
        norm = torch.linalg.norm(activations, dim=1, keepdims=True)
        activations = activations / norm
        ones = torch.ones((n, 1))
        return torch.hstack((ones,activations))

    def compute_pNML_params(self, activations_train: ArrayType) -> ArrayType:
        # Calc x_Bot
        logger.info('pNML Score: Fitting parameters...')
        activations_train = activations_train.clone()
        activations_aug_train = self.transform_activations(activations_train) # add bias term
        p = activations_aug_train.T @ activations_aug_train
        self.p_parallel = torch.linalg.pinv(p)
        self.p_bot = torch.eye(activations_aug_train.shape[1]) - self.p_parallel @ p

    def get_scores(self, activations_eval: ArrayType, output_predictions:bool=False) -> ArrayType:
        """
        Calculate the genie probability
        :param activations_eval: the dataset to evaluate: (n,m)
        :param probs: The model probability of the dataset: (n,)
        :param p_parallel: projection matrix, the parrallel component
        :param p_bot: projection matrix, the orthogonal component
        :return:
        """
        logger.info('pNML Score: Computing scores...')
        activations_eval = activations_eval.clone()
        bias_weight = torch.hstack( (self.b[:,None], self.w) )
        activations_aug_eval = self.transform_activations(activations_eval) # add bias term
        logits_eval = activations_aug_eval @ bias_weight.T
        probs = self.softmax(logits_eval)
        n, n_classes = probs.shape

        # Calc energy of each component
        # x_parallel_square = np.array([x @ self.p_parallel @ x.T for x in activations_aug])
        x_parallel_square = torch.sum((activations_aug_eval @ self.p_parallel) * activations_aug_eval, dim=1)
        # x_bot_square = np.array([x @ self.p_bot @ x.T for x in activations_aug])
        x_bot_square = torch.sum((activations_aug_eval @ self.p_bot) * activations_aug_eval, dim=1)
        # x_t_g = np.maximum(x_bot_square, x_parallel_square / (1 + x_parallel_square))
        x_t_g = torch.maximum(x_bot_square, x_parallel_square / (1 + x_parallel_square))
        # x_t_g = np.expand_dims(x_t_g, -1)
        x_t_g_repeated = x_t_g[:,None].repeat(1,self.num_classes)

        # Genie prediction
        genie_predictions = probs / (probs + (1 - probs) * (probs ** x_t_g_repeated))

        # Regret
        nfs = genie_predictions.sum(dim=1)
        regrets = np.log(nfs) / np.log(self.num_classes)

        # pNML probability assignment
        # pnml_prediction = genie_predictions / np.repeat(
        #     np.expand_dims(nfs, -1), n_classes, axis=1
        # )
        if output_predictions:
            pnml_prediction = genie_predictions / nfs[:,None].repeat(1,self.num_classes)
            return -regrets, pnml_prediction
        else:
            return -regrets
    
    def save_params(self, path:str|None=None, filename:str='pNML_params'):
        assert self.p_parallel is not None, 'p_parallel matrix has not been computed'
        assert self.p_bot is not None, 'p_bot matrix has not been computed'
        params_dict = {
                    'p_parallel': self.p_parallel,
                    'p_bot': self.p_bot,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'pNML Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='pNML_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'pNML Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.p_parallel = params_dict['p_parallel']
        self.p_bot = params_dict['p_bot']
#%%

class EntropyScores:
    def __init__(self, cf, confid_func_name:str='renyi'):
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.confid_func_name = confid_func_name
        if confid_func_name == 'renyi':
            self.confid_func = scores_funcs.renyi_entropy
            # self.gamma_range =  np.arange(0.0, 1.0, 0.1)
        elif confid_func_name == 'generalized':
            self.confid_func = scores_funcs.generalized_entropy
            # self.gamma_range = np.arange(0.0, 1.0, 0.1)
        elif confid_func_name == 'tsallis':
            self.confid_func = scores_funcs.tsallis_entropy
            # self.gamma_range =  np.arange(0.0, 1.0, 0.1)
        else:
            raise NotImplementedError
        self.M = None
        self.gamma = None

    def compute_entropy_params(self, softmax:ArrayType, 
                                        residuals:ArrayType,
                                        gamma_bounds:tuple=(1e-6,0.999999),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Entropy Score ({self.confid_func_name}): Fitting parameters ...')
        softmax = softmax.clone()
        residuals = residuals.clone()
        # n, n_classes = softmax.shape
        M_bounds = (1, self.num_classes)
        def _get_metric(M, gamma):
            confidence = self.confid_func(softmax, gamma=gamma, M=int(M))
            stats_val_ = RiskCoverageStats(confids =  confidence, residuals = residuals)
            return -stats_val_.augrc
        
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'M': M_bounds, 
                                                'gamma': gamma_bounds,
                                                },
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.M = int(best_params['M'])
        self.gamma = best_params['gamma']
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
    
    def get_scores(self, softmax_eval:ArrayType) -> ArrayType:
        logger.info(f'Entropy Score: Computing scores ...')
        softmax_eval = softmax_eval.clone()
        return self.confid_func(softmax_eval, gamma=self.gamma, M=self.M)

    
    def save_params(self, path:str|None=None, filename:str='EntropyScore_params'):
        assert self.M is not None, 'M has not been computed'
        assert self.gamma is not None, 'gamma has not been computed'
        params_dict = {
                    'M': self.M,
                    'gamma': self.gamma,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Entropy Score ({self.confid_func_name}): Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='EntropyScore_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Entropy Score ({self.confid_func_name}): Loading parameters from {path}')
        params_dict = torch.load(path)
        self.M = params_dict['M']
        self.gamma = params_dict['gamma']

#%%
class KernelPCA:
    def __init__(self, module, study_name:str, cf, mode:str='global'):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = self.cf.data.num_classes 
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.mode = mode
        self.condition_transform = ('ViT' in [self.cf.model.network.name, self.cf.model.network.backbone]) 
        logger.info('Using non-standardized inputs' if self.condition_transform else 'Standardizing inputs...')
        self.X_ref, self.K_c, self.normalization, self.u_q = None, None, None, None
        self.explained_variance = None
        self.kernel = None
        self.gamma = None
    
    def _feat_normalization(self, X):
        X = X / (torch.norm(X, p=2, dim=-1, keepdim=True) + 1e-12)
        X = X.contiguous()
        return X
    
    def _rbf_kernel(self, X, Y, gamma=None):
        """
        RBF (Gaussian) kernel between X (n, d) and Y (m, d).
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        X_norm = (X**2).sum(dim=1).view(-1, 1)
        Y_norm = (Y**2).sum(dim=1).view(1, -1)
        K = X_norm + Y_norm - 2 * X @ Y.t()
        return torch.exp(-gamma * K)
    
    def _center_kernel(self, K, K_ref=None, center_on='all'):
        """
        Center the kernel matrix.
        - K: (n, m), kernel between data and reference.
        - K_ref: (m, m), kernel between reference and reference.
        - center_on: 'none', 'all', or 'mixed'.
        """
        n, m = K.shape
        if center_on == 'none':
            return K
        if center_on == 'all':
            # Center with respect to all data (classic double centering)
            K_mean_rows = K.mean(dim=1, keepdim=True)      # (n, 1)
            K_mean_cols = K.mean(dim=0, keepdim=True)      # (1, m)
            K_mean_all  = K.mean()
            # return (K_mean_rows + K_mean_cols - K_mean_all)
            return (K_mean_rows,K_mean_cols,K_mean_all)
        elif center_on == 'mixed':
            # Mixed centering: center K_{nm} using K_ref
            # K: (n, m), K_ref: (m, m)
            K_mean_rows = K.mean(dim=1, keepdim=True)      # (n, 1)
            K_ref_mean_cols = K_ref.mean(dim=0, keepdim=True)  # (1, m)
            K_ref_mean = K_ref.mean()
            # return (K_mean_rows + K_ref_mean_cols - K_ref_mean)
            return (K_mean_rows,K_ref_mean_cols,K_ref_mean)
        else:
            raise ValueError("center_on must be 'none', 'all', or 'mixed'.")

    def _kernel(self, X, Y):
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y, gamma=self.gamma)
        else:
            raise NotImplementedError("Only 'rbf' kernel implemented.")

    def _get_eigendecomposition(self, M):
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     eigvals, eigvecs = torch.linalg.eigh(M.to(device))
        #     eigvals, eigvecs = eigvals.cpu(), eigvecs.cpu()
        # else:
        #     eigvals, eigvecs = torch.linalg.eigh(M)
        # Returns eigenvalues (descending), eigenvectors
        try:
            eigvals, eigvecs = torch.linalg.eigh(M)
        except:
            n,m = M.shape
            epsilon = 1e-6
            M_p = M + epsilon*torch.eye(min(m,n),device=self.device) 
            eigvals, eigvecs = torch.linalg.eigh(M_p)
        idx = torch.argsort(eigvals, descending=True)
        return eigvals[idx], eigvecs[:, idx]
    
    def KPCA(self, X, ref_indices=None, m_samples=None, center_on=None, verbose=False):
        X = self._feat_normalization(X)
        
        if ref_indices is None:
            if m_samples is None:
                n = X.shape[0]
                raise ValueError("Specify either m_samples or ref_indices.")
            ref_indices = torch.randperm(n)[:m_samples]
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        X = X.to(self.device)
        X_ref = X[ref_indices]
        # Compute K_mm and K_nm
        K_mm = self._kernel(X_ref, X_ref)  # (m, m)
        # Eigendecomposition (only top n_components)
        evals, evecs = self._get_eigendecomposition(K_mm)
        evals = torch.maximum(evals, torch.tensor(1e-12))
        normalization = (evecs / torch.sqrt(evals))@ evecs.T
        # ---- low-rank projection
        K_nm = self._kernel(X, X_ref) @ normalization.T
        # K_c = self._center_kernel(K_nm, K_mm, center_on=center_on)
        K_c = K_nm.mean(dim=0)
        # # Explained variance selection
        # sigma = (K_nm-K_c).T@((K_nm-K_c))
        sigma = cov(K_nm-K_c, centered=True, rowvar=False)
        evals_full, evecs_full = self._get_eigendecomposition(sigma)
        evals_full_accuml = evals_full.cumsum(0)
        explained_variance_ratio_ = evals_full_accuml/evals_full_accuml[-1]
        n_components = (explained_variance_ratio_<= self.explained_variance).sum().item() + 1
        u_q = evecs_full[:,:n_components]
        if verbose:
            logger.info(f"n_components = {n_components}, variance ratio = {explained_variance_ratio_[n_components - 1]}")
        return X_ref.cpu(), K_c.cpu(), normalization.cpu(), u_q.cpu()

    def compute_KernelPCA_params(self, activations_train: ArrayType,
                                        labels:ArrayType|None = None, 
                                        only_correct: bool = False, 
                                        temperature:float|None=None, n_landmarks:int|None=None,
                                        explained_variance:float|None=None, gamma:float|None=None,
                                        center_on:str|None=None, kernel:str|None=None,
                                        verbose:bool=False):
        """
        X: (n, d) torch tensor.
        m_samples: number of landmark points for Nystrom.
        ref_indices: (m,) landmark indices. If None, random sample.
        """
        if verbose:
            logger.info(f'Kernel PCA with {self.mode} mode: Fitting parameters...')
        activations_train = activations_train.clone()
        self.explained_variance = 0.95 if explained_variance is None else explained_variance
        self.gamma = 0.2 if gamma is None else gamma 
        self.temperature = 1.0 if temperature is None else temperature
        self.n_landmarks = activations_train.shape[0]/self.num_classes if n_landmarks is None else n_landmarks 
        self.center_on = 'all' if center_on is None else center_on
        self.kernel = 'rbf' if kernel is None else kernel
        if 'global' in self.mode:
            if only_correct:
                assert self.labels is not None, "Labels have not been provided..."
                self.labels = labels.clone()
                logits_train = activations_train@self.w.T + self.b
                softmax_train = F.softmax(logits_train, dim=1, dtype=torch.float64)
                predictions = softmax_train.max(dim=1).indices
                correct_idx = predictions==self.labels
                activations_train = activations_train[correct_idx]
                energy_train = scores_funcs.energy(logits_train[correct_idx], temperature=self.temperature)
                energy_idx = energy_train.argsort()[:self.n_landmarks]
                self.X_ref, self.K_c, self.normalization, self.u_q = self.KPCA(activations_train, ref_indices=energy_idx, center_on=self.center_on)
            else:
                logits_train = activations_train@self.w.T + self.b
                energy_train = scores_funcs.energy(logits_train, temperature=self.temperature)
                energy_idx = energy_train.argsort()[:self.n_landmarks]
                self.X_ref, self.K_c, self.normalization, self.u_q = self.KPCA(activations_train, ref_indices=energy_idx, center_on=self.center_on, verbose=verbose)
        elif 'class' in self.mode:
            assert (labels is not None) and (len(labels)==len(activations_train)) , f'Labels are required to compute class subspaces.\n N_labels={len(labels)}, N_activations={len(activations_train)}'
            self.labels = labels.clone()
            self.X_ref = []
            self.K_c = []
            self.normalization = []
            self.u_q = []
            for c in range(self.num_classes):
                labels_per_class_tensor = self.labels[self.labels==c]
                activations_per_class_tensor = activations_train[self.labels==c]
                logits_per_class_tensor = activations_per_class_tensor@self.w.T + self.b
                if only_correct:
                    predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                    correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                    if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                        activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                        logits_per_class_tensor = logits_per_class_tensor[correct_idx]
                    else:
                        logger.info(f'No correct predictions for class {c} in the training set. The PCA for this class uses all the activations.')
                energy_per_class_tensor = scores_funcs.energy(logits_per_class_tensor, temperature=self.temperature)
                energy_per_class_tensor_idx = energy_per_class_tensor.argsort()[:self.n_landmarks]
                X_ref, K_c, normalization, u_q = self.KPCA(activations_per_class_tensor, ref_indices=energy_per_class_tensor_idx, center_on=self.center_on, verbose=verbose)
                self.X_ref.append(X_ref)
                self.K_c.append(K_c)
                self.normalization.append(normalization)
                self.u_q.append(u_q)

    def get_scores(self, activations_eval,
                            predictions_eval:ArrayType|None=None,
                            combine:str|None=None):
        logger.info(f'Kernel PCA: Computing scores...')
        X = activations_eval.clone()
        X = self._feat_normalization(X)
        N = X.shape[0]
        if self.mode == 'global':
            K_nm = self._kernel(X, self.X_ref) @ self.normalization.T
            # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
            K_nm_backprojected = (self.u_q@self.u_q.T@(K_nm-self.K_c).T).T
            scores = -1 * torch.linalg.norm(K_nm_backprojected - (K_nm-self.K_c), ord=2, dim=1)
        elif self.mode == 'class':
            if combine=='average':
                K_nm_backprojected_list = []
                K_nm_centered_list = []
                for c in range(self.num_classes):
                    K_nm = self._kernel(X, self.X_ref[c]) @ self.normalization[c].T
                    # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
                    K_nm_centered = K_nm - self.K_c[c]
                    K_nm_backprojected = (self.u_q[c]@self.u_q[c].T@(K_nm_centered).T).T
                    K_nm_backprojected_list.append(K_nm_backprojected)
                    K_nm_centered_list.append(K_nm_centered)
                K_nm_backprojected_avg = []
                for t in range(N):
                    avg_sampled = []
                    for c in range(self.num_classes):
                        avg_sampled.append(K_nm_backprojected_list[c][t])
                    K_nm_backprojected_avg.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
                K_nm_backprojected_avg = torch.stack(K_nm_backprojected_avg, dim=0)

                K_nm_centered_avg = []
                for t in range(N):
                    avg_sampled = []
                    for c in range(self.num_classes):
                        avg_sampled.append(K_nm_centered_list[c][t])
                    K_nm_centered_avg.append(torch.stack(avg_sampled,dim=0).mean(dim=0))
                K_nm_centered_avg = torch.stack(K_nm_centered_avg, dim=0)
                scores = -1 * torch.linalg.norm(K_nm_backprojected_avg - K_nm_centered_avg, ord=2, dim=1)
            
            else:
                scores_list = []
                for c in range(self.num_classes):
                    K_nm = self._kernel(X, self.X_ref[c]) @ self.normalization[c].T
                    # K_nm = self._center_kernel(K_nm, self.K_mm, center_on=self.center_on)
                    K_nm_backprojected = (self.u_q[c]@self.u_q[c].T@(K_nm-self.K_c[c]).T).T
                    score = -1 * torch.linalg.norm(K_nm_backprojected - (K_nm-self.K_c[c]), ord=2, dim=1)
                    scores_list.append(score)
                scores = torch.stack(scores_list, dim=1)
                if predictions_eval is None:
                    scores = scores.amax(dim=1) # pick best reconstruction
                else:
                    scores = torch.gather(scores, 1, predictions_eval.reshape(-1,1)).squeeze() # pick reconstruction guided by prediction

        return scores

    def tune_hyperparameters( self, activations_train: ArrayType,
                                        activations_val: ArrayType, 
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType = None, 
                                        only_correct: bool = False, 
                                        temperature:float|None=None,
                                        center_on:str = 'all',
                                        kernel:str = 'rbf',
                                        var_bounds:tuple = (0.85, 0.99),
                                        gamma_bounds:tuple = (0.10, 1.0),
                                        landmarks_bounds:tuple = (2000, 5000),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'Kernel PCA: Tuning hyper-parameters required to minimize AUGRC using the validation set...')
        activations_val = activations_val.clone()
        def _get_metric(explained_variance, gamma, n_landmarks):
            self.compute_KernelPCA_params(activations_train,
                                        labels=labels_train, 
                                        only_correct=False, 
                                        temperature=temperature,
                                        n_landmarks=int(n_landmarks),
                                        explained_variance=explained_variance, 
                                        gamma=gamma,
                                        center_on=center_on,
                                        kernel=kernel)
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'explained_variance': var_bounds, 
                                                'gamma': gamma_bounds, 
                                                'n_landmarks': landmarks_bounds},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.compute_KernelPCA_params(activations_train,
                                        labels=labels_train, 
                                        only_correct=False, 
                                        temperature=temperature,
                                        n_landmarks=int(best_params['n_landmarks']),
                                        explained_variance=best_params['explained_variance'], 
                                        gamma=best_params['gamma'],
                                        center_on=center_on,
                                        kernel=kernel,
                                        verbose=True)
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")
    
    def save_params(self, path:str|None=None, filename:str='KernelPCA_params'):
        assert self.X_ref is not None, 'X_ref has not been computed'
        assert self.K_c is not None, 'K_c has not been computed'
        assert self.normalization is not None, 'normalization has not been computed'
        assert self.u_q is not None, 'u_q has not been computed'
        assert self.kernel is not None, 'kernel has not been defined'
        assert self.gamma is not None, 'gamma has not been defined'
        params_dict = {
                        'X_ref': self.X_ref,
                        'K_c': self.K_c,
                        'normalization': self.normalization,
                        'u_q':self.u_q,
                        'kernel':self.kernel,
                        'gamma':self.gamma,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'KernelPCA Score: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='KernelPCA_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'KernelPCA Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.X_ref = params_dict['X_ref']
        self.K_c = params_dict['K_c']
        self.normalization = params_dict['normalization']
        self.u_q = params_dict['u_q']
        self.kernel = params_dict['kernel']
        self.gamma = params_dict['gamma']


#%%
class NNGuide:
# Nearest Neighbor Guidance

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.normalizer = lambda x: x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-10)
        self.bank_guide = None
        self.proportion = None

    def tune_hyperparameters(self,activations_train: ArrayType,
                                        activations_val: ArrayType,
                                        residuals_val:ArrayType,
                                        labels_train:ArrayType|None = None,
                                        logits_train:ArrayType|None = None,
                                        k_clusters_bounds:tuple=(10,500),
                                        proportion_bounds:tuple=(0.1,0.5),
                                        n_iters:int=80,
                                        n_init:int=20,):
        logger.info(f'NNGuide Score: Tunung hyper-parameters required to minimize AUGRC using the validation set...')
        def _get_metric(k_clusters, proportion):
            self.compute_NNGuide_params(activations_train,
                                        labels_train=labels_train,
                                        logits_train=logits_train,
                                        proportion=proportion,
                                        k_clusters=int(k_clusters) )
            scores_val = self.get_scores(activations_val)
            stats_val_ = RiskCoverageStats(confids =  scores_val, residuals = residuals_val)
            return -stats_val_.augrc
        bo = BayesianOptimization(
                                    f=_get_metric,
                                    pbounds={   'k_clusters': k_clusters_bounds, 
                                                'proportion': proportion_bounds,},
                                    verbose=0,
                                    random_state=1,
                                )
        # Perform the optimization
        bo.maximize(init_points=n_init, n_iter=n_iters)
        # Best hyperparameters and corresponding accuracy
        best_params = bo.max['params']
        best_augrc = bo.max['target']
        self.compute_NNGuide_params(activations_train,
                                        labels_train=labels_train,
                                        logits_train=logits_train,
                                        proportion=best_params['proportion'],
                                        k_clusters=int(best_params['k_clusters']) )
        logger.info(f"Best Hyperparameters: {best_params}")
        logger.info(f"Best AUGRC: {-1*best_augrc}")

    def compute_NNGuide_params(self, activations_train: ArrayType,
                                        labels_train:ArrayType|None = None,
                                        logits_train:ArrayType|None = None,
                                        proportion:float=0.1,
                                        k_clusters:int=10):
        self.k_clusters = 10 if k_clusters is None else k_clusters
        self.proportion = 0.1 if proportion is None else proportion
        logger.info(f'NNGuide Score: Fitting parameters with alpha={self.proportion}...')
        activations_train = activations_train.clone()
        if logits_train is None:
            logits_train = activations_train @ self.w.T + self.b
        assert labels_train is not None, f'Labels need to be provided to sample features by classes...'
        labels_train = labels_train.clone()
        # proportion = 0.1 # Proportion of samples
        size_per_class = int((len(labels_train)/len(labels_train.unique())*proportion))
        logger.info(f'NNGuide Score: Each class is represented by {size_per_class} samples...')
        np.random.seed(12345) # For consistency
        idx_train = set([ x for y in (labels_train).unique() for x in (np.random.choice(np.argwhere(labels_train==y).squeeze(),size=size_per_class,replace=False)) ])
        sampled_activations_train = torch.stack([activations_train[idx] for idx in idx_train])
        bank_features = self.normalizer(sampled_activations_train)
        bank_logits = torch.stack([logits_train[idx] for idx in idx_train])
        bank_energy = torch.logsumexp(bank_logits, dim=1)
        self.bank_guide = bank_features * bank_energy[:, None]
        # activations_aug_train = self.transform_activations(activations_train) # add bias term
        # p = activations_aug_train.T @ activations_aug_train
        # self.p_parallel = torch.linalg.pinv(p)
        # self.p_bot = torch.eye(activations_aug_train.shape[1]) - self.p_parallel @ p
    
    def get_scores( self, activations_eval: ArrayType ):
        # logger.info('ViM: Computing scores')
        logger.info(f'NNGuide Score: Computing scores...')
        activations_eval = activations_eval.clone()
        logits_eval = activations_eval @ self.w.T + self.b
        energy_eval = torch.logsumexp(logits_eval, dim=1)
        features_eval = self.normalizer(activations_eval)
        # conf_eval = knn_score(self.bank_guide, features_eval, k=10)
        
        feat_data_loader = torch.utils.data.DataLoader(features_eval, batch_size=35, shuffle=False)
        # conf_eval = torch.Tensor(np.concatenate([self.knn_score(self.bank_guide, x, k=k_clusters) for x in feat_data_loader]))
        conf_eval = torch.cat([self.knn_score(self.bank_guide, x, k=self.k_clusters) for x in feat_data_loader])
        scores_eval = conf_eval * energy_eval
        
        return scores_eval
    
    def knn_score(self,bankfeas, queryfeas, k=100, min=False):

        inner_prod_topk = torch.topk(torch.mm(queryfeas, bankfeas.T), k=k)
        if min:
            # scores = np.array(D.min(axis=1))
            scores = inner_prod_topk.values.amin(dim=1)
        else:
            # scores = np.array(D.mean(axis=1))
            scores = inner_prod_topk.values.mean(dim=1)
        
        return scores
    
    def save_params(self, path:str|None=None, filename:str='NNGuide_params'):
        assert self.bank_guide is not None, 'bank_guide has not been computed'
        assert self.proportion is not None, 'proportion has not been computed'
        assert self.k_clusters is not None, 'k_clusters has not been computed'
        params_dict = {
                    'bank_guide': self.bank_guide,
                    'proportion': self.proportion,
                    'k_clusters': self.k_clusters,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'NNGuide Score: Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='NNGuide_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'NNGuide Score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.bank_guide = params_dict['bank_guide']
        self.proportion = params_dict['proportion']
        self.k_clusters = params_dict['k_clusters']

#%%
class fDBD:
# Fast Decision Boundary based Out-of-Distribution Detector

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        # self.module.model.encoder.disable_dropout()
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        # self.model.encoder.disable_dropout()
        if self.study_name == 'confidnet':
            self.network = self.module.network
            self.network.encoder.disable_dropout()
        else:
            self.network = None
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.normalizer = lambda x: x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-10)
        self.denominator_matrix = None
        self.mean_global_train = None
        # self.proportion = None

    def compute_fDBD_params(self, activations_train: ArrayType,):
        
        # self.proportion = proportion
        logger.info(f'fDBD Score: Fitting parameters...')
        activations_train = activations_train.clone()
        self.mean_global_train = activations_train.mean(dim=0)
        self.denominator_matrix = torch.zeros((self.num_classes, self.num_classes))
        for p in range(self.num_classes):
            w_p = self.w - self.w[p, :]
            denominator = torch.linalg.norm(w_p, dim=1)
            denominator[p] = 1
            self.denominator_matrix[p, :] = denominator
            
    def get_scores( self, activations_eval: ArrayType, logits_eval: ArrayType|None=None, normalizer='distance' ):
        # logger.info('ViM: Computing scores')
        logger.info(f'fDBD: Computing scores...')
        activations_eval = activations_eval.clone()
        if logits_eval is None:
            logits_eval = activations_eval @ self.w.T + self.b
        else:
            logits_eval = logits_eval.clone()
            logits_eval = logits_eval[:,:self.num_classes]
        mls, mls_idx = logits_eval.max(dim=1)
        logits_abs_sub = torch.abs(logits_eval - mls[:,None])
        if normalizer=='distance':
            scores_eval = 1/(self.num_classes-1) \
                                * torch.sum(logits_abs_sub / self.denominator_matrix[mls_idx], dim=1)\
                                * 1/(torch.norm(activations_eval - self.mean_global_train, dim=1))
        elif normalizer=='activation':
            scores_eval = 1/(self.num_classes-1) \
                                * torch.sum(logits_abs_sub / self.denominator_matrix[mls_idx], dim=1)\
                                * 1/(torch.norm(activations_eval, dim=1))
        else:
            raise NotImplementedError
        
        return scores_eval
    
    
    def save_params(self, path:str|None=None, filename:str='fDBD_params'):
        assert self.mean_global_train is not None, 'mean_global_train has not been computed'
        assert self.denominator_matrix is not None, 'denominator_matrix has not been computed'
        params_dict = {
                    'mean_global_train': self.mean_global_train,
                    'denominator_matrix': self.denominator_matrix,
                    }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'fDBD score: Saving parameters in {path}')
        torch.save(params_dict, path)
    
    def load_params(self, path:str|None=None, filename:str='fDBD_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'fDBD score: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.mean_global_train = params_dict['mean_global_train']
        self.denominator_matrix = params_dict['denominator_matrix']


#%%
class NeuralCollapseMetrics:

    def __init__(self, module, study_name:str, cf):
        self.module = copy.deepcopy(module)
        self.cf = cf
        self.num_classes = self.cf.data.num_classes
        self.study_name = study_name
        _, self.w, self.b = utils.get_model_and_last_layer(self.module, self.study_name)
        if self.study_name == 'dg':
            self.w = self.w[:self.num_classes,:]
            self.b = self.b[:self.num_classes]
        self.class_means = None
        self.global_mean = None
        self.nc_metrics = None
        self.class_variance = None
    
    def equiangular(self, C:ArrayType):
        n = C.shape[0]
        diagonal_mask = torch.eye(n, dtype=torch.bool)
        non_diagonal_mask = ~diagonal_mask
        row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
        # il = torch.tril_indices(row=len(C), col=len(C), offset=-1)
        equi_angle = torch.std(C[row_indices,col_indices])
        max_angle = torch.mean(torch.abs( C[row_indices,col_indices] + 1/(self.num_classes-1) )) 
        return equi_angle, max_angle
    
    def cdnv(self, class_means, class_variance):
        mu_dist = pairwise_euclidean_distance(torch.vstack(class_means), zero_diagonal=False)
        variances = torch.stack(class_variance)
        var_sum = variances.unsqueeze(0) + variances.unsqueeze(-1)
        cdnv_matrix = var_sum/(2*mu_dist)

        n = cdnv_matrix.shape[0]
        diagonal_mask = torch.eye(n, dtype=torch.bool)
        non_diagonal_mask = ~diagonal_mask
        row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
        return torch.mean(cdnv_matrix[row_indices,col_indices])
    
    # def equiangular(self, C:ArrayType):
    #     n = matrix.shape[0]
    #     diagonal_mask = torch.eye(n, dtype=torch.bool)
    #     non_diagonal_mask = ~diagonal_mask
    #     row_indices, col_indices = torch.nonzero(non_diagonal_mask, as_tuple=True)
    #     # il = torch.tril_indices(row=len(C), col=len(C), offset=-1)
    #     # return torch.std(C[il[0],il[1]])
    #     return torch.std(C[row_indices,col_indices])

    # def max_angular(self, C:ArrayType):
    #     # il = torch.tril_indices(row=len(C), col=len(C), offset=0)
    #     # return torch.mean( torch.abs( C[il[0],il[1]] + 1/(self.num_classes-1) ) )
    #     return torch.mean( torch.abs( C + 1/(self.num_classes-1) ) )
    
    def etf_simplex_difference(self, A:ArrayType, B:ArrayType|None=None):
        etf_simplex = (1/np.sqrt(self.num_classes-1)) * (torch.eye(self.num_classes)-(1/self.num_classes)*torch.ones((self.num_classes,self.num_classes)))
        if B is None:
            ABt = A@A.T
        else:
            ABt = A@B.T
        # torch.linalg.norm(, ord='fro')
        diff = (ABt)/torch.linalg.norm(ABt, ord='fro') - etf_simplex
        return torch.linalg.norm(diff, ord='fro')
    
    def compute_NeuralCollapse_params(self, activations_train: ArrayType,
                                    labels_train:ArrayType,
                                    only_correct: bool = False ):
        logger.info("Neural Collapse Metrics: Computing global, class means, and metrics...")
        labels_train = labels_train.clone()
        activations_train = activations_train.clone()
        self.global_mean = activations_train.mean(dim=0)
        # self.unique_labels = torch.unique(labels_train)
        dim_B = self.global_mean.shape[0]
        sigma_B = torch.zeros(dim_B, dim_B)
        sigma_W = torch.zeros(dim_B, dim_B)
        self.class_means = []
        self.class_variance = []
        for c in range(self.num_classes):

            activations_per_class_tensor = activations_train[labels_train==c]
            if only_correct:
                labels_per_class_tensor = labels_train[labels_train==c]
                logits_per_class_tensor = activations_per_class_tensor @ self.w.T + self.b
                predictions_per_class_tensor = logits_per_class_tensor.max(dim=1).indices
                correct_idx = predictions_per_class_tensor==labels_per_class_tensor
                if correct_idx.sum()>0: # Make sure that any given class has correct predictions, 
                    activations_per_class_tensor = activations_per_class_tensor[correct_idx]
                else:
                    logger.info(f'No correct predictions for class {c} in the training set. The mean vector for this class uses all the activations that belong to class {c}.')
                class_mean = activations_per_class_tensor.mean(dim=0)
                self.class_means.append( class_mean )
            else:
                class_mean = activations_per_class_tensor.mean(dim=0)
                self.class_means.append( class_mean )
            # Compute metrics
            mu_cG = (class_mean - self.global_mean).reshape(-1,1)
            sigma_B = sigma_B + mu_cG @ mu_cG.T 
            # dim_W = class_mean.shape[0]
            # sigma_W = torch.zeros(dim_W, dim_W)
            H_k = torch.zeros(len(activations_per_class_tensor), dim_B)
            for j, h_ki in enumerate(activations_per_class_tensor):
                h_ki_c = (h_ki - class_mean).reshape(-1,1)
                H_k[j] = h_ki_c.T
                sigma_W = sigma_W + h_ki_c @ h_ki_c.T
            self.class_variance.append( torch.linalg.norm(H_k, dim=1, ord=2).pow(2).mean() )
        # Variability Collapse (Within-class variation collapse)
        K = self.num_classes
        N = activations_train.shape[0]
        sigma_B = (1/K) * sigma_B
        sigma_W = (1/(N*K)) * sigma_W
        var_collapse = (1/K)*torch.trace(sigma_W @ torch.linalg.pinv(sigma_B))
        # Equiangularity and Max-angle
        M = torch.vstack(self.class_means) - self.global_mean
        cos_uc = pairwise_cosine_similarity( M, zero_diagonal=False )
        cos_wc = pairwise_cosine_similarity( self.w, zero_diagonal=False )
        equiangular_uc, max_equiangular_uc = self.equiangular( cos_uc )
        equiangular_wc, max_equiangular_wc = self.equiangular( cos_wc )
        # Equinormality
        M_norm_class = torch.linalg.norm( M, dim=1, ord=2 )
        W_norm_class = torch.linalg.norm( self.w, dim=1, ord=2 )
        equinorm_uc = torch.std(M_norm_class)/torch.mean(M_norm_class)
        equinorm_wc = torch.std(W_norm_class)/torch.mean(W_norm_class)
        # Maximal-angle Equiangularity
        # max_equiangular_uc = self.max_angular( cos_uc )
        # max_equiangular_wc = self.max_angular( cos_wc )
        # Self-duality
        M_tilde = M/torch.linalg.norm(M, ord='fro')
        W_tilde = self.w/torch.linalg.norm(self.w, ord='fro')
        # ETF Simplex differences norm
        w_etf_diff = self.etf_simplex_difference(self.w/W_norm_class.unsqueeze(-1))
        M_etf_diff = self.etf_simplex_difference(M/M_norm_class.unsqueeze(-1))
        wM_etf_diff = self.etf_simplex_difference(self.w/W_norm_class.unsqueeze(-1), M/M_norm_class.unsqueeze(-1))
        # CDNV
        cdnv_score = self.cdnv(self.class_means, self.class_variance)
        # Bias collapse
        bias_collapse = torch.linalg.norm(self.b + self.w@self.global_mean, ord=2)
        self_duality = torch.linalg.norm(W_tilde-M_tilde, ord='fro').pow(2)
        
        logger.info(f"Variability collapse: {var_collapse}")
        logger.info(f"CDNV score: {cdnv_score}")
        logger.info(f"Bias collapse: {bias_collapse}")
        logger.info(f"Equiangularity means: {equiangular_uc}")
        logger.info(f"Equiangularity weights: {equiangular_wc}")
        logger.info(f"Max-Equiangularity means: {max_equiangular_uc}")
        logger.info(f"Max-Equiangularity weights: {max_equiangular_wc}")
        logger.info(f"Equinormality means: {equinorm_uc}")
        logger.info(f"Equinormality weights: {equinorm_wc}")
        logger.info(f"Self-duality: {self_duality}")
        logger.info(f"ETF difference (W): {w_etf_diff}")
        logger.info(f"ETF difference (M): {M_etf_diff}")
        logger.info(f"ETF difference (WMt): {wM_etf_diff}")

        self.nc_metrics = {'var_collapse':var_collapse,
                        'cdnv_score': cdnv_score,
                        'bias_collapse': bias_collapse,
                        'equiangular_uc':equiangular_uc,
                        'equiangular_wc':equiangular_wc,
                        'equinorm_uc':equinorm_uc,
                        'equinorm_wc':equinorm_wc,
                        'max_equiangular_uc':max_equiangular_uc,
                        'max_equiangular_wc':max_equiangular_wc,
                        'self_duality':self_duality,
                        'w_etf_diff':w_etf_diff,
                        'M_etf_diff':M_etf_diff,
                        'wM_etf_diff':wM_etf_diff,}

    def save_params(self, path:str|None=None, filename:str='NeuralCollapse_params'):
        # assert self.precision is not None, 'Precision matrix has not been computed...'
        assert self.class_means is not None, 'Class means have not been computed...'
        assert self.global_mean is not None, 'Global mean have not been computed...'
        assert self.class_variance is not None, 'Class variances have not been computed...'
        assert self.nc_metrics is not None, 'Neural Collapse metrics have not been computed...'
        # assert self.alpha is not None, 'Unique labels have not been computed...'
        params_dict = {
                        'global_mean': self.global_mean,
                        'class_means': self.class_means,
                        'class_variance': self.class_variance,
                        'nc_metrics': self.nc_metrics,
                        # 'unique_labels': self.unique_labels,
                        }
        if path is None:
            if os.path.exists(f'{self.cf.exp.dir}/params'):
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
            else:
                os.mkdir(f'{self.cf.exp.dir}/params')
                path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        logger.info(f'Neural Collapse Metrics: Saving parameters in {path}')
        torch.save(params_dict, path)

    def load_params(self, path:str|None=None, filename:str='NeuralCollapse_params'):
        if path is None:
            path = f'{self.cf.exp.dir}/params/{filename}.pt'
        else:
            path = f'{path}/{filename}.pt'
        assert os.path.exists(path), f'Specified path {path} does not exist..'
        logger.info(f'Neural Collapse Metrics: Loading parameters from {path}')
        params_dict = torch.load(path)
        self.global_mean = params_dict['global_mean']
        self.class_means = params_dict['class_means']
        self.class_variance = params_dict['class_variance']
        self.nc_metrics = params_dict['nc_metrics']

# SLURM_EXPORT_ENV=NONE salloc --nodes=1 --partition=ececis_research --gres=gpu:p100   --mem-per-cpu=20G --cpus-per-task=12  --time=12:00:00
# %%
