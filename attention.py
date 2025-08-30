import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=192, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.register_buffer('mask', None)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.mask is None or self.mask.size(0) != seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
            
        attn_scores.masked_fill_(self.mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(attn_output)

def multi_head_attention(encoder_input, multi_head_w_q, multi_head_w_k, multi_head_w_v, w_o, seq_len):
    """Legacy function for backward compatibility"""
    if encoder_input.dim() == 2:
        encoder_input = encoder_input.unsqueeze(0)
    
    batch_size, actual_seq_len, d_model = encoder_input.shape
    n_heads = 8
    d_k = d_model // n_heads
    
    q = torch.matmul(encoder_input, multi_head_w_q).view(batch_size, actual_seq_len, n_heads, d_k).transpose(1, 2)
    k = torch.matmul(encoder_input, multi_head_w_k).view(batch_size, actual_seq_len, n_heads, d_k).transpose(1, 2)
    v = torch.matmul(encoder_input, multi_head_w_v).view(batch_size, actual_seq_len, n_heads, d_k).transpose(1, 2)
    
    scale = 1.0 / math.sqrt(d_k)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    mask = torch.triu(torch.ones(actual_seq_len, actual_seq_len, device=encoder_input.device, dtype=torch.bool), diagonal=1)
    attn_scores.masked_fill_(mask, float('-inf'))
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, actual_seq_len, d_model)
    
    output = torch.matmul(attn_output, w_o)
    
    if encoder_input.shape[0] == 1:
        output = output.squeeze(0)
    
    return output