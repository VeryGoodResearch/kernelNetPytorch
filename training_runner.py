# This module provides some helper utilities used for training the model
import torch

from .model import MultiLayerKernelNet

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor):
    masked_diff = mask * (truth - predictions)
    loss = torch.sum(masked_diff**2) / 2 # tensorflow implementation of the opimizer divides by 2 for some reason, we do the same to replicate the results
    loss = loss + reg_term
    loss.backward()
    return loss


def _training_iter(model: MultiLayerKernelNet, 
                  t_data: torch.Tensor, 
                  v_data: torch.Tensor,
                  t_mask: torch.Tensor,
                  v_mask: torch.Tensor,
                  optimizer: torch.optim.LBFGS,
                  optimizer_steps: int = 1):
    def optimizer_run():
        optimizer.zero_grad()
        t_pred, t_reg = model.forward(t_data)
        loss = _loss(t_pred, t_data, t_reg, t_mask)
        return 0.0

    for _ in range(optimizer_steps):
        optimizer.step(optimizer_run)
        

