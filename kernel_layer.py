# This module provides a simple implementation of a KernelNet layer in Pytorch. See kernel.py for available kernel implementations
import torch
import torch.nn as nn
from .kernel import gaussian_kernel

class KernelLayer(nn.Module):

    def __init__(self, 
                 n_in: int, # Number of input features
                 n_hid: int = 500, # Number of hidden units
                 n_dim: int = 5, # Number of dimensions to be embedded for kernelization
                 activation = torch.sigmoid, # Activation function for layer output
                 kernel = gaussian_kernel, # Kernel function
                 lambda_o: float = 0.013, # Sparsity regularization parameter 
                 lambda_2: int = 60) -> None: # L2 regularization parameter
        # Housekeeping

        super(KernelLayer, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.activation = activation
        self.kernel = kernel
        self.lambda_o = lambda_o
        self.lambda_2 = lambda_2

        # Define Pytorch learnable params

        # Connection matrix
        self.W = nn.Parameter(torch.Tensor(self.n_in, self.n_hid))
        # Learnable vectors, used in distance calculation in kernel function
        self.u = nn.Parameter(torch.Tensor(self.n_in, 1, self.n_dim))
        self.v = nn.Parameter(torch.Tensor(1, self.n_hid, self.n_dim))
        # Biases
        self.b = nn.Parameter(torch.Tensor(n_hid))

        # Initialize params. For some cases, it is beneficial not to do it randomly.
        nn.init.normal_(self.W, mean=0.0, std=1e-3)
        nn.init.normal_(self.u, mean=0.0, std=1e-3)
        nn.init.normal_(self.v, mean=0.0, std=1e-3)
        nn.init.zeros_(self.b)



