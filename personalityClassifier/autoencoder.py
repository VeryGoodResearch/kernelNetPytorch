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
                 activation = torch.sigmoid,
                 kl_activation = 0.02,
                 ) -> None:
        super(KernelNetAutoencoder, self).__init__()
        self.n_input = n_input
        self.kernel_hidden = kernel_hidden
        self.lambda_o = lambda_o
        self.lambda_2 = lambda_2
        self.kernel_function = kernel_function.__name__
        self.kl_activation = kl_activation
        self.device = get_device()
        self.enc = Encoder(
                    n_input,
                    kernel_hidden,
                    lambda_o,
                    lambda_2,
                    hidden_dims,
                    kernel_function,
                    activation
        ).to(self.device)
        self.dec = Decoder(
                    n_input,
                    kernel_hidden,
                    lambda_o,
                    lambda_2,
                    hidden_dims,
                    kernel_function,
                    activation
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, enc_reg = self.enc(x)
        kl_reg = self.__compute_KL_reg(x)
        x, dec_reg = self.dec(x)
        return x, enc_reg+dec_reg, kl_reg


    def __compute_KL_reg(self, activations: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        a = activations.clamp(min=eps, max=1 - eps)
        g = torch.full_like(a, self.kl_activation).clamp(min=eps, max=1 - eps)
        reg = a * torch.log(a / g) + (1 - a) * torch.log((1 - a) / (1 - g))
        if torch.isnan(reg).any():
            print("KL reg has NaNs:", a.min(), a.max())
        term = reg.sum() / a.shape[0]
        print(f"Mean activation: {activations.mean().item()}")
        return term

    def parameters(self, recurse: bool = True):
        return list(self.enc.parameters(recurse)) + list(self.dec.parameters(recurse))



