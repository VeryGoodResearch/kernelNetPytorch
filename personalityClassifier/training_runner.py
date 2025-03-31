# This module provides some helper utilities used for training the model
from datetime import datetime
from os import path, makedirs
from numpy.strings import upper
import torch
import numpy as np
import time
from torchmin import ScipyMinimizer

from personalityClassifier.autoencoder import KernelNetAutoencoder
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.utils import get_device

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor,
          sparsity_factor: float):
    masked_diff = mask * (truth - predictions)
    masked_loss = torch.sum(masked_diff**2) / 2
    loss = torch.sum((truth-predictions)**2) / 2
    loss = loss/sparsity_factor + masked_loss + reg_term 
    return loss


def _training_iter(model: KernelNetAutoencoder, 
                  epoch: int,
                  t_data: torch.Tensor, 
                  v_data: torch.Tensor,
                  t_mask: torch.Tensor,
                  v_mask: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                  sparsity_factor: float,  
                  log_file,):
    # Not the greatest idea to run it otherwise
    assert torch.is_grad_enabled()
    def optimizer_run():
        optimizer.zero_grad()
        t_pred, t_reg = model.forward(t_data)
        loss = _loss(t_pred, t_data, t_reg, t_mask, sparsity_factor)
        loss.backward()
        """
        ## Debug
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient mean: {param.grad.mean()}")
        """        
        return loss

    optimizer.step(optimizer_run)

    # Validation
    with torch.no_grad():
        predictions, t_reg = model.forward(t_data)
        print(predictions)
        error_train = ((predictions - t_data) ** 2).sum() / predictions.numel()   # compute train error
        error_train_observed = (t_mask * (predictions - t_data)**2).sum() / t_mask.sum()
        loss_train = _loss(predictions, t_data, t_reg, t_mask, sparsity_factor)
        # Validation stuff
        predictions, t_reg = model.forward(v_data)
        error_validation = ((predictions - v_data) ** 2).sum() / predictions.numel()  # compute validation error
        error_validation_observed = (v_mask * (predictions - v_data)**2).sum() / v_mask.sum()
        loss_validation = _loss(predictions, v_data, t_reg, v_mask, sparsity_factor)

        print('.-^-._' * 12, file=log_file)
        print('epoch:', epoch, file=log_file) 
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train), file=log_file)
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train, file=log_file)
        print('validation observed rmse: ', np.sqrt(error_validation_observed), ', train observed rmse: ', np.sqrt(error_train_observed), file=log_file)
        print('.-^-._' * 12, file=log_file)

        print('.-^-._' * 12) 
        print('epoch:', epoch) 
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train))
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train)
        print('validation observed rmse: ', np.sqrt(error_validation_observed), ', train observed rmse: ', np.sqrt(error_train_observed))
        print('.-^-._' * 12)

def train_model(
        epochs: int,
        training_data: torch.Tensor,
        validation_data: torch.Tensor,
        training_mask: torch.Tensor,
        validation_mask: torch.Tensor,
        output_path: str, 
        lambda_o: float = 0.013,
        lambda_2: float = 60,
        activation = torch.sigmoid_,
        kernel = gaussian_kernel,
        hidden_dims = 500,
        output_every: int = 5,
        # WARNING WARNING WARNING 
        # IF YOU HAVE BAD PC (LOW MEMORY) THEN TUNE THIS THING DOWN, OTHERWISE IT WILL PROBABLY EXPLODE
        history_size: int = 10,
        learning_rate: float = 1,
        sparsity_factor: float = 0.035
        ):
    device = get_device()
    n_input = training_data.shape[1]
    model = KernelNetAutoencoder(
            n_input,
            lambda_o=lambda_o,
            lambda_2=lambda_2,
            kernel_hidden=hidden_dims,
            kernel_function=kernel,
            activation=activation,
            ).to(device)
    """
    optimizer = torch.optim.Rprop(
           model.parameters(),
           lr=learning_rate
           )
    optimizer = ScipyMinimizer(
            model.parameters(),
            method='L-BFGS-B',
            options={'maxiter': output_every, 'disp': True, 'maxcor': history_size}
            )
    """
    optimizer = torch.optim.LBFGS(
             model.parameters(), 
             max_iter=output_every, 
             history_size=history_size,
             lr=learning_rate,
             line_search_fn='strong_wolfe'
             )
    n_epochs = int(epochs/output_every)
    makedirs(output_path, exist_ok=True)
    log_path= path.join(output_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    with open(log_path, 'w+') as log_file:
        for epoch in range(n_epochs):
            start = time.time()
            _training_iter(
                    model,
                    epoch,
                    training_data,
                    validation_data,
                    training_mask,
                    validation_mask,
                    optimizer,
                    sparsity_factor,
                    log_file,
                    )
            elapsed = time.time() - start
            # save_model(model, output_path)
            print(f'Run took {elapsed} seconds')
    return model
