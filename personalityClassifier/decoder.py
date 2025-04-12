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
        self.layers = nn.Sequential(
                KernelLayer(n_in=kernel_hidden,
                            activation=nn.Identity(),
                            n_hid=n_output,
                            n_dim=50,
                            lambda_o=lambda_o,
                            lambda_2=lambda_2,
                            kernel=kernel_function
                            ).to(self.device)
                )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_reg = None
        y = x
        for layer in self.layers.children():
            y, current_reg = layer.forward(y)
            total_reg = current_reg if total_reg is None else total_reg+current_reg
        return y, total_reg

    def parameters(self, recurse: bool = True):
        return self.layers.parameters(recurse)
