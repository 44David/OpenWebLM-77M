import torch
import torch.nn as nn
import torch.nn.functional as F

class FastLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)

def LayerNorm(x, gamma, beta, eps=1e-5):
    return F.layer_norm(x, x.shape[-1:], gamma, beta, eps)
    