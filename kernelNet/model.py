# This module contains sample models utilizing kernelNet layer
import torch
import torch.nn as nn

from .kernel import gaussian_kernel

from .kernel_layer import KernelLayer

class MultiLayerKernelNet(nn.Module):

    # General structure of the model, layers are fed to each other sequentially
    layers: nn.Module

    def __init__(self, 
                 n_input, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 kernel_layers = 2, # Number of sequential kernelNet layers
                 lambda_o: float = 0.013,
                 lambda_2: float = 60,
                 kernel_function = gaussian_kernel,
                 activation = torch.sigmoid
                 ) -> None:
        super(MultiLayerKernelNet, self).__init__()
        self.n_input = n_input
        self.kernel_hidden = kernel_hidden
        self.lambda_o = lambda_o
        self.lambda_2 = lambda_2
        self.kernel_function = kernel_function.__name__

        self.layers = nn.Sequential(
                KernelLayer(
                    n_input, 
                    n_hid=kernel_hidden,
                    lambda_o=lambda_o,
                    lambda_2=lambda_2,
                    kernel=kernel_function,
                    activation=activation
                ),
                KernelLayer(
                    kernel_hidden,
                    n_hid=kernel_hidden,
                    lambda_o=lambda_o,
                    lambda_2=lambda_2,
                    kernel=kernel_function,
                    activation=activation
                ),
                KernelLayer(kernel_hidden,
                    n_hid=n_input,
                    lambda_o=lambda_o,
                    lambda_2=lambda_2,
                    kernel=kernel_function,
                    activation=nn.Identity())
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_reg = None
        y = x
        for layer in self.layers.children():
            y, current_reg = layer.forward(y)
            total_reg = current_reg if total_reg is None else total_reg+current_reg
        return y, total_reg

    def __latent_forward(self, x: torch.Tensor):
        assert not torch.is_grad_enabled()
        children = list(self.layers.children())
        x, _ = children[0].forward(x)
        x, _ = children[1].forward(x)
        return x

    def mmd_forward(self, x: torch.Tensor):
        assert not torch.is_grad_enabled()
        return self.__latent_forward(x)

    def parameters(self, recurse: bool = True):
        return self.layers.parameters(recurse)
