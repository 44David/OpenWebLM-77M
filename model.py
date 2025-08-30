import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenization import input_preprocessing, EmbeddingLayer
from layernorm import LayerNorm, FastLayerNorm
from attention import multi_head_attention, MultiHeadAttention
from feedforward_net import Net


class DecoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.d_model = d_model
        
        self.ln1 = FastLayerNorm(d_model)
        self.ln2 = FastLayerNorm(d_model)
        
        self.attention = MultiHeadAttention(d_model=d_model)
        self.ffn = Net(d_model)
        
    def forward(self, x):
        attn_out = self.attention(self.ln1(x))
        x = x + attn_out
        
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x    
     
    

class Transformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=128, n_layers=6):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model) for _ in range(n_layers)
        ])
        
        self.ln_final = FastLayerNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self._init_weights()
        
    def forward(self, x):
        x = self.embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, FastLayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
