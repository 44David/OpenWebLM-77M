import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=256, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer('pos_encoding', self._create_positional_encoding(max_seq_len, d_model))
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def forward(self, token_ids):
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            
        batch_size, seq_len = token_ids.shape
        
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        
        return token_emb + pos_emb

# Legacy global variables for backward compatibility  
d_model = 512
vocab_size = 50257
embed = None
embedding_layer = None

def input_preprocessing(token_ids, d_model, seq_len):
    global embedding_layer, embed
    
    if embedding_layer is None:
        embedding_layer = EmbeddingLayer(vocab_size=50257, d_model=d_model, max_seq_len=2048)
        if token_ids.is_cuda:
            embedding_layer = embedding_layer.cuda()
    
    encoded_embedding = embedding_layer(token_ids)
    
    # For backward compatibility, also return input_embedding
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    input_embedding = embedding_layer.token_embedding(token_ids)
    
    return encoded_embedding, input_embedding
