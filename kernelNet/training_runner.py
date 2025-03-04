# This module provides some helper utilities used for training the model
from numpy.strings import upper
import torch
import numpy as np
import time

from .kernel import gaussian_kernel
from .optimizer import LBFGSB
from .model import MultiLayerKernelNet

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor):
    masked_diff = mask * (truth - predictions)
    loss = torch.sum(masked_diff**2) / 2
    loss = loss + reg_term
    return loss


def _training_iter(model: MultiLayerKernelNet, 
                  epoch: int,
                  t_data: torch.Tensor, 
                  v_data: torch.Tensor,
                  t_mask: torch.Tensor,
                  v_mask: torch.Tensor,
                  optimizer: torch.optim.LBFGS):
    # Not the greatest idea to run it otherwise
    assert torch.is_grad_enabled()
    def optimizer_run():
        optimizer.zero_grad()
        t_pred, t_reg = model.forward(t_data)
        loss = _loss(t_pred, t_data, t_reg, t_mask)
        loss.backward()
        return loss

    optimizer.step(optimizer_run)

    # Validation
    with torch.no_grad():
        predictions, t_reg = model.forward(t_data)
        clipped = torch.clamp(predictions, 1.0, 5.0)
        v_predictions, v_reg = model.forward(v_data)
        v_clipped = torch.clamp(v_predictions, 1.0, 5.0)
        error_validation = (v_mask * (v_clipped - v_data) ** 2).sum() / v_mask.sum() #compute validation error
        error_train = (t_mask * (clipped - t_data) ** 2).sum() / t_mask.sum() #compute train error
        loss_train = _loss(predictions, t_data, t_reg, t_mask)
        loss_validation = _loss(v_predictions, v_data, v_reg, v_mask)

        print('.-^-._' * 12)
        print('epoch:', epoch) 
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train))
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train)
        print('.-^-._' * 12)

def train_model(
        epochs: int,
        training_data: torch.Tensor,
        validation_data: torch.Tensor,
        training_mask: torch.Tensor,
        validation_mask: torch.Tensor,
        lambda_o: float = 0.013,
        lambda_2: float = 60,
        activation = torch.sigmoid,
        kernel = gaussian_kernel,
        hidden_dims = 500,
        output_every: int = 5,
        # WARNING WARNING WARNING 
        # IF YOU HAVE BAD PC (LOW MEMORY) THEN TUNE THIS THING DOWN, OTHERWISE IT WILL PROBABLY EXPLODE
        history_size: int = 10,
        learning_rate: float = 1
        ):
    n_input = training_data.shape[1]
    model = MultiLayerKernelNet(
            n_input,
            lambda_o=lambda_o,
            lambda_2=lambda_2,
            kernel_hidden=hidden_dims,
            kernel_function=kernel,
            activation=activation,
            )
    # n_params = sum([np.prod(p.size()) for p in model.parameters()])
    # x_l=(torch.ones(n_params)*(-100.0))
    # x_u=(torch.ones(n_params)*(100.0))
    # optimizer = LBFGSB(
    #       model.parameters(),
    #       max_iter=output_every,
    #       history_size=history_size,
    #       upper_bound=x_u,
    #       lower_bound=x_l
    #       )
    # optimizer = torch.optim.LBFGS(
    #        model.parameters(), 
    #        max_iter=output_every, 
    #        history_size=history_size,
    #        lr=learning_rate,
    #        line_search_fn='strong_wolfe'
    #        )
    optimizer = torch.optim.Rprop(
            model.parameters(),
            lr=learning_rate
            )
    n_epochs = int(epochs/output_every)
    for epoch in range(n_epochs):
        start = time.time()
        _training_iter(
                model,
                epoch,
                training_data,
                validation_data,
                training_mask,
                validation_mask,
                optimizer
                )
        elapsed = time.time() - start
        print(f'Run took {elapsed} seconds')
    return model
