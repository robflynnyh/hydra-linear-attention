import torch.nn as nn

class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None):
        '''x: (B, T, D)'''
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out = kvw.sum(dim=-2, keepdim=True) * q
        return self.out(out)