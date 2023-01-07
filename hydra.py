import torch.nn as nn, torch

class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='scale_and_bias'):
        '''
        output_layer: 'scale_and_bias' | 'linear' | 'none'
        '''
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        if output_layer == 'scale_and_bias':
            self.scale = nn.Parameter(torch.ones(1, 1, d_model))
            self.bias = nn.Parameter(torch.zeros(1, 1, d_model))
            self.out = lambda x: x * self.scale + self.bias
        elif output_layer == 'linear':
            self.out = nn.Linear(d_model, d_model)
        elif output_layer == 'none':
            self.out = nn.Identity()

    def forward(self, x):
        '''x: (B, T, D)'''
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        kv = (k * v).sum(dim=-2, keepdim=True)
        out = q * kv
        return self.out(out)