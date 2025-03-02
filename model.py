# This module contains sample models utilizing kernelNet layer
import torch
import torch.nn as nn

from .kernel_layer import KernelLayer

class MultiLayerKernelNet(nn.Module):
    def __init__(self, 
                 n_input, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 kernel_layers = 2, # Number of sequential kernelNet layers
                 ) -> None:
        super(MultiLayerKernelNet, self).__init__()

