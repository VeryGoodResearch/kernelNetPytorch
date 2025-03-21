# This module contains sample models utilizing kernelNet layer
import torch
import torch.nn as nn

from .kernel_layer import KernelLayer

class MultiLayerKernelNet(nn.Module):

    # General structure of the model, layers are fed to each other sequentially
    layers = []

    def __init__(self, 
                 n_input, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 kernel_layers = 2, # Number of sequential kernelNet layers
                 ) -> None:
        super(MultiLayerKernelNet, self).__init__()
        self.layers = [KernelLayer(n_input, n_hid=kernel_hidden) for _ in range(kernel_layers)]
        self.layers.append(KernelLayer(kernel_hidden, n_input, activation=nn.Identity))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_reg = None
        y = x
        for layer in self.layers:
            y, current_reg = layer.forward(x)
            total_reg = current_reg if total_reg is None else total_reg+current_reg
        return y, total_reg
