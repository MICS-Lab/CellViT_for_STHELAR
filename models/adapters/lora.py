import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super(LoRA, self).__init__()
        self.linear_layer = linear_layer
        self.in_dim = self.linear_layer.in_features
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.adapter_Q_downsample = nn.Parameter(torch.randn(self.in_dim, rank) * std)
        self.adapter_Q_upsample = nn.Parameter(torch.zeros(rank, self.in_dim))
        self.adapter_V_downsample = nn.Parameter(torch.randn(self.in_dim, rank) * std)
        self.adapter_V_upsample = nn.Parameter(torch.zeros(rank, self.in_dim))
        self.adapter_alpha = alpha
    
    def forward(self, x):
        x_q = self.adapter_alpha * (x @ self.adapter_Q_downsample @ self.adapter_Q_upsample)
        x_v = self.adapter_alpha * (x @ self.adapter_V_downsample @ self.adapter_V_upsample)
        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        return x