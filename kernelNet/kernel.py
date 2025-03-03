# This module defines various kernel functions which can be used with the KernelNet model
# Kernel functions take an input in form of 2 tensors u and v, where u = [n_in, 1, n_dim] and v = [1, n_hid, n_dim] and return a connection matrix for inputs and outputs based on the distance between points along the last dimension of the tensor
import torch

# Gaussian kernel is used in the proposed models in the paper. 
def gaussian_kernel(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dist = torch.norm(u-v, dim=2, p=2)    
    out = (1-dist**2).clamp(min=0)
    return out
