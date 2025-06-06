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
                 lambda_2: float = 60, # L2 regularization parameterk
                 ) -> None:
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_hat = self.kernel(self.u, self.v) # [n_in, n_hid]
        # Compute regularization terms
        sparse_reg_term = self.lambda_o * torch.sum(w_hat**2)
        l2_reg_term = self.lambda_2 * torch.sum(self.W**2)
        # Compute actual output
        w_eff = self.W * w_hat # [n_in, n_hid]
        y = torch.matmul(x, w_eff) + self.b
        y = self.activation(y)
        reg_term = sparse_reg_term + l2_reg_term 
        return y, reg_term

