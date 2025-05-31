import torch
from torch import nn

class VectorToOffsetMatrix(nn.Module):
    def forward(self, x):
        x = x.squeeze()
        _, dim = x.size()
        result = torch.stack([torch.roll(x, shifts=i, dims=1) for i in range(dim)], dim=1)
        return result.unsqueeze(1)


def kl_bernoulli(p, q, eps=1e-6):
    p = torch.clamp(p, eps, 1 - eps)
    q = torch.clamp(q, eps, 1 - eps)
    res = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    return res.mean()
