import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self, d_model=192):
        super(Net, self).__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model
        
        # Linear(dim_in, dim_out)
        self.fc1 = nn.Linear(self.d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, self.d_model)
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x
    
