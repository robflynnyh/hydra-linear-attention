from einops import rearrange
import torch.nn as nn

class HydraAttention(nn.Module):
    def __init__(self, d_model):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)

    def forward(self, x):
        '''x: (B, T, D)'''
        qkv = self.qkv(x)
        q,k,v = rearrange(qkv, 'b n d -> b d n ()').chunk(3, dim=1)
        knorm = k / (k.norm(dim=-2, keepdim=True))
        qnorm = q / (q.norm(dim=-2, keepdim=True))
        kv = knorm.transpose(-2,-1).matmul(v)
        out = qnorm * kv
        return rearrange(out, 'b d n () -> b n d')