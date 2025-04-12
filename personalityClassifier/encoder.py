# Definition of the encoder part of the autoencoder
from typing import override
from torch import nn
import torch.nn.functional as F
import torch

from personalityClassifier.kernel_layer import KernelLayer
from personalityClassifier.utils import get_device
from .kernel import gaussian_kernel

class Encoder(nn.Module):
    
    def __init__(self,
                 n_input, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 lambda_o: float = 0.013,
                 lambda_2: float = 60,
                 hidden_dims = 5,
                 kernel_function = gaussian_kernel,
                 activation = torch.sigmoid
                 ) -> None:
        super().__init__()
        self.n_input = n_input
        self.kernel_hidden = kernel_hidden
        self.lambda_o = lambda_o
        self.lambda_2 = lambda_2
        self.kernel_function = kernel_function.__name__
        self.activation = activation
        self.device = get_device()

        self.kernel1 = KernelLayer(n_in=n_input,
                            activation=nn.Identity(),
                            n_hid=kernel_hidden*3,
                            n_dim=20,
                            lambda_o=lambda_o,
                            lambda_2=lambda_2,
                            kernel=kernel_function
                            ).to(self.device)
        self.bn = nn.LayerNorm(kernel_hidden*3)
        self.kernel2 = KernelLayer(n_in=kernel_hidden*3,
                            activation=nn.Identity(),
                            n_hid=kernel_hidden,
                            n_dim=10,
                            lambda_o=lambda_o,
                            lambda_2=lambda_2,
                            kernel=kernel_function
                            ).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, first_reg = self.kernel1.forward(x)
        x = self.activation(self.bn(x))
        x, second_reg = self.kernel2.forward(x)
        return x, first_reg+second_reg

    @override
    def parameters(self, recurse: bool = True):
        return list(self.kernel1.parameters(recurse)) + list(self.bn.parameters(recurse)) + list(self.kernel2.parameters(recurse))

