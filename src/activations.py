import torch
# import gpytorch
import torch.nn.functional as F
from torch import nn
import numpy as np
from scipy.stats import norm, skew, kurtosis
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Detect GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {DEVICE}...')

class ActivationsGradients:
    def __init__(self, model, layers, study_name, loss):
        self.model = model.to(DEVICE).eval()
        self.layers = layers
        self.study_name = study_name 
        # self.mode = mode
        self.activations = {}
        self.gradients = {}
        self.hooks_forward = []
        self._register_hooks_forward()
        self.hooks_backward = []
        self._register_hooks_backward()
        self.loss = loss
        # self.gp_model = None
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
        # self.num_inducing_points = num_inducing_points
        # self.variance_threshold = variance_threshold
        # self.lambda_variance = lambda_variance
        # self.alpha = alpha  # Rényi entropy parameter

    def _register_hooks_forward(self):
        def hook_fn(module, input, output, layer_name):
            self.activations[layer_name] = output.detach().cpu()
        
        for name, module in self.model.named_modules():
            if name in self.layers:
                hook_forward = module.register_forward_hook(lambda mod, inp, out, lname=name: hook_fn(mod, inp, out, lname))
                self.hooks_forward.append(hook_forward)
    
    def _register_hooks_backward(self):
        def hook_fn(module, grad_input, grad_output, layer_name):
            self.gradients[layer_name] = grad_output[0].detach().cpu()
        
        for name, module in self.model.named_modules():
            if name in self.layers:
                hook_backward = module.register_full_backward_hook(lambda mod, inp, out, lname=name: hook_fn(mod, inp, out, lname))
                self.hooks_backward.append(hook_backward)

    def _compute_activations_and_gradients(self, x, y):
        self.activations.clear()
        self.gradients.clear()
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.model.zero_grad()
        loss.backward()
        # if self.study_name == 'confidnet':
        batch_activations = []
        batch_gradients = []
        
        for layer in self.layers:
            act = self.activations[layer]
            act = act.numpy()
            batch_size = act.shape[0]
            act = act.reshape(batch_size, -1)
            batch_activations.append(act)

            grad = self.gradients[layer]
            grad = grad.numpy()
            batch_size = grad.shape[0]
            grad = grad.reshape(batch_size, -1)            
            batch_gradients.append(grad)
        
        return np.hstack(batch_activations), np.hstack(batch_gradients)
    
    def _compute_activations(self, x):
        self.activations.clear()
        # self.gradients.clear()
        _ = self.model(x)
        # loss = self.loss(logits, y)
        # self.model.zero_grad()
        # loss.backward()
        # if self.study_name == 'confidnet':
        batch_activations = []
        # batch_gradients = []
        
        for layer in self.layers:
            act = self.activations[layer]
            act = act.numpy()
            batch_size = act.shape[0]
            act = act.reshape(batch_size, -1)
            batch_activations.append(act)

            # grad = self.gradients[layer]
            # grad = grad.numpy()
            # batch_size = grad.shape[0]
            # grad = grad.reshape(batch_size, -1)            
            # batch_gradients.append(grad)
        
        return np.hstack(batch_activations)
    
    def _collect_activations_and_gradients(self, train_loader):
        train_activations, train_gradients = [], []
        # with torch.no_grad():
        for x, y in tqdm(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            activations,gradients = self._compute_activations_and_gradients(x,y)
            train_activations.append(activations)
            train_gradients.append(gradients)
        train_activations = np.vstack(train_activations)
        train_gradients = np.vstack(train_gradients)
        
        return train_activations, train_gradients
    
    def _collect_activations(self, train_loader):
        train_activations, train_labels = [], [],
        # with torch.no_grad():
        for x, y in tqdm(train_loader):
            x = x.to(DEVICE)
            y = y.detach().cpu().numpy()
            activations = self._compute_activations(x)
            train_activations.append(activations)
            train_labels.append(y)
        train_activations = np.vstack(train_activations)
        train_labels = np.concatenate(train_labels)
        
        return train_activations, train_labels
    
    def remove_hooks(self):
        if len(self.hooks_forward)>0:
            for hook in self.hooks_forward:
                hook.remove()
        if len(self.hooks_backward)>0:
            for hook in self.hooks_backward:
                hook.remove()

# [pip3] torch==1.13.1
# [pip3] torch_pca==1.0.0
# [pip3] torchmetrics==1.5.2
# [pip3] torchvision==0.14.1

# [conda] torchmetrics              1.5.2                    pypi_0    pypi
# [conda] torchvision               0.14.1                   pypi_0    pypi
# [conda] pytorch                   1.13.1          cpu_py310hd11e9c7_1    conda-forge
# [conda] pytorch-cuda              11.7                 h778d358_5    pytorch
# [

#     pip install torchmetrics==1.5.2
#     pip install gpytorch compatible with torch==1.13.1
#     python -m torch.utils.collect_env