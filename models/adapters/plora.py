import torch
import torch.nn as nn
import torch.nn.functional as F

class PLoRA(nn.Module):
    def __init__(self, linear_layer, rank, alpha, inference_sampling=10):
        super(PLoRA, self).__init__()
        self.linear_layer = linear_layer
        self.in_dim = linear_layer.in_features
        self.rank = rank
        self.alpha = alpha
        self.inference_sampling = inference_sampling

        std = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.adapter_K_downsample = nn.Parameter(torch.randn(self.in_dim, self.rank*2) * std)
        self.adapter_K_upsample = nn.Parameter(torch.zeros(self.rank, self.in_dim))

    def reparametrize(self, mu, logvar):
        """
            Reparametrization trick for sampling from a Gaussian distribution. During training, sampling is
            performed once, during inference, sampling is performed multiple times.
            arguments:
                mu [torch.Tensor]: mean of the Gaussian distribution
                logvar [torch.Tensor]: log of the variance of the Gaussian distribution
            returns:
                z [torch.Tensor]: sample from the Gaussian distribution
        """
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + eps*std
        else:
            sigmas = [torch.exp(0.5*logvar)*torch.randn_like(torch.exp(0.5*logvar)) for _ in range(int(self.inference_sampling))]
            z = [mu + sigma for sigma in sigmas]
        return z

    def forward(self, x):
        """
            Forward pass of PLoRA, computing the forward pass of the linear layer and adding the PLoRA adaptation.
            arguments:
                x [torch.Tensor]: input to the multi-head attention layer
            returns:
                x [torch.Tensor]: adapted output
        """
        dx_q = torch.zeros_like(x)
        dx_v = torch.zeros_like(x)

        dx_k = self.reparametrize(*(x @ self.adapter_K_downsample).chunk(2, dim=-1))
        if isinstance(dx_k, list):
            dx_k = (F.sigmoid(self.alpha * (torch.stack(dx_k) @ self.adapter_K_upsample))).mean(axis=0)
        else:
            dx_k = F.sigmoid(self.alpha * (dx_k @ self.adapter_K_upsample))
        
        dx = torch.cat([dx_q, dx_v, dx_k], dim=-1)
        x = self.linear_layer(x) + dx
        return x