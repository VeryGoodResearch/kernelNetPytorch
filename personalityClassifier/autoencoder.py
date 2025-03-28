## First stage of the classifier - an autoencoder model, separated into clear encoder and decoder parts is trained
from torch import nn
import torch

from personalityClassifier.decoder import Decoder
from personalityClassifier.encoder import Encoder
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.utils import get_device

class KernelNetAutoencoder(nn.Module):
    def __init__(self, 
                 n_input, # Input/output dims
                 kernel_hidden = 500, # Kernel layer hidden dimensions
                 lambda_o: float = 0.013,
                 lambda_2: float = 60,
                 hidden_dims = 5,
                 kernel_function = gaussian_kernel,
                 activation = torch.sigmoid
                 ) -> None:
        super(KernelNetAutoencoder, self).__init__()
        self.n_input = n_input
        self.kernel_hidden = kernel_hidden
        self.lambda_o = lambda_o
        self.lambda_2 = lambda_2
        self.kernel_function = kernel_function.__name__
        self.device = get_device()

        self.layers = nn.Sequential(
                Encoder(
                    n_input,
                    kernel_hidden,
                    lambda_o,
                    lambda_2,
                    hidden_dims,
                    kernel_function,
                    activation
                    ).to(self.device),
                Decoder(
                    n_input,
                    kernel_hidden,
                    lambda_o,
                    lambda_2,
                    hidden_dims,
                    kernel_function,
                    activation
                    ).to(self.device)
            ).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        total_reg = None
        y = x.to(self.device)
        for layer in self.layers.children():
            y, current_reg = layer.forward(y)
            total_reg = current_reg if total_reg is None else total_reg+current_reg
        return y, total_reg

    def parameters(self, recurse: bool = True):
        return self.layers.parameters(recurse)

