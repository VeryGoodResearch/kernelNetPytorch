# Definition of the decoder part of the autoencoder
import torch
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.kernel_layer import KernelLayer
from personalityClassifier.utils import get_device
from torch import nn


class Decoder(nn.Module):
    
    def __init__(self,
                 n_output, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 lambda_o: float = 0.013,
                 lambda_2: float = 60,
                 hidden_dims = 5,
                 kernel_function = gaussian_kernel,
                 activation = torch.sigmoid
                 ) -> None:
        super().__init__()
        self.device = get_device()
        self.layer = KernelLayer(n_in=kernel_hidden,
                            activation=activation,
                            n_hid=n_output,
                            n_dim=hidden_dims,
                            lambda_o=lambda_o,
                            lambda_2=lambda_2,
                            kernel=kernel_function
                            ).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layer.forward(x)

    def parameters(self, recurse: bool = True):
        return self.layer.parameters(recurse)
