import torch.nn as nn

class BottleNeck(nn.Module):
    def __init__(self, in_dim, adapter_activation, reduction_factor):
        super(BottleNeck, self).__init__()
        hidden_dim = int(in_dim/reduction_factor)
        self.adapter_activation = getattr(nn, adapter_activation)()
        self.adapter_downsample = nn.Linear(in_dim, hidden_dim)
        self.adapter_upsample = nn.Linear(hidden_dim, in_dim)
    
    def forward(self, x_in):
        x = self.adapter_downsample(x_in)
        x = self.adapter_activation(x)
        x = self.adapter_upsample(x)
        x += x_in
        return x