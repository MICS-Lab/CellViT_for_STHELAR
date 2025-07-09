import torch
import torch.nn as nn

class AdaptFormer(nn.Module):
    def __init__(self, layer_norm, linear, adapter_activation, reduction_factor):
        super(AdaptFormer, self).__init__()
        self.layer_norm = layer_norm
        self.linear = linear
        self.adapter_alpha = nn.Parameter(torch.ones(1))
        
        hidden_dim = int(self.linear.fc1.in_features/reduction_factor)
        self.adapter_downsample = nn.Linear(self.linear.fc1.in_features, hidden_dim)
        self.adapter_activation = getattr(nn, adapter_activation)()
        self.adapter_upsample = nn.Linear(hidden_dim, self.linear.fc1.in_features)
    
    def forward(self, x):
        main_x = self.linear(self.layer_norm(x))
        adapted_x = self.adapter_upsample(self.adapter_activation(self.adapter_downsample(x))) * self.adapter_alpha
        return main_x + adapted_x