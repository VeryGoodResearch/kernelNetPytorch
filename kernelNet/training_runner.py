# This module provides some helper utilities used for training the model
import torch
import numpy as np

from .kernel import gaussian_kernel

from .model import MultiLayerKernelNet

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor):
    masked_diff = mask * (truth - predictions)
    # tensorflow implementation of the opimizer divides by 2 for some reason, we do the same to replicate the results
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
        predictions, _ = model.forward(t_data)
        clipped = torch.clamp(predictions, 1.0, 5.0)
        v_predictions, _ = model.forward(v_data)
        v_clipped = torch.clamp(v_predictions, 1.0, 5.0)
        error_validation = (v_mask * (clipped - v_data) ** 2).sum() / v_mask.sum() #compute validation error
        error_train = (t_mask * (v_clipped - t_data) ** 2).sum() / t_mask.sum() #compute train error

        print('.-^-._' * 12)
        print('epoch:', epoch, 'validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train))
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
        history_size: int = 10
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
    optimizer = torch.optim.LBFGS(
            model.parameters(), 
            max_iter=output_every, 
            history_size=history_size,
            line_search_fn="strong_wolfe"
            )
    
    n_epochs = int(epochs/output_every)
    for epoch in range(n_epochs):
        _training_iter(
                model,
                epoch,
                training_data,
                validation_data,
                training_mask,
                validation_mask,
                optimizer
                )
    return model
