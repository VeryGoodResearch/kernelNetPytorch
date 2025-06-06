# This module provides some helper utilities used for training the model
from datetime import datetime
from os import path, makedirs
import torch
import numpy as np
import time
from ignite.metrics import MaximumMeanDiscrepancy
from torch.nn import KLDivLoss
from torchmin import ScipyMinimizer

from .kernel import gaussian_kernel
from .model import MultiLayerKernelNet
from .save_model import save_model

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor):
    masked_diff = mask * (truth - predictions)
    loss = torch.sum(masked_diff**2) / 2
    loss = loss + reg_term 
    return loss

def _kl_loss(predictions: torch.Tensor,
             truth: torch.Tensor,
             reg_term: torch.Tensor,
             mask: torch.Tensor):
    masked_truth = mask * truth
    masked_preds = mask * predictions
    kl_loss = KLDivLoss(reduction='batchmean')
    loss = kl_loss(masked_preds, masked_truth)
    loss = loss + reg_term 
    return loss

def _evaluate_mmd(model: MultiLayerKernelNet, X, yhat):
    original_latent = model.mmd_forward(X)
    reconstructed_latent = model.mmd_forward(yhat)
    metric = MaximumMeanDiscrepancy(var=0.1)
    metric.reset()
    metric.update((original_latent, reconstructed_latent))
    mmd = metric.compute()
    return mmd


def _training_iter(model: MultiLayerKernelNet, 
                  epoch: int,
                  t_data: torch.Tensor, 
                  v_data: torch.Tensor,
                  t_mask: torch.Tensor,
                  v_mask: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                  log_file,
                  _loss,
                  min_rating: float,
                  max_rating: float):
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
        clipped = torch.clamp(predictions, min_rating, max_rating)
        error_validation = (v_mask * (clipped - v_data) ** 2).sum() / v_mask.sum()  # compute validation error
        error_train = (t_mask * (clipped - t_data) ** 2).sum() / t_mask.sum()  # compute train error
        loss_train = _loss(predictions, t_data, t_reg, t_mask)
        loss_validation = _loss(predictions, v_data, t_reg, v_mask)
        mmd_train = _evaluate_mmd(model, t_data, predictions)
        mmd_validation = _evaluate_mmd(model, v_data, predictions)

        print('.-^-._' * 12, file=log_file)
        print('epoch:', epoch, file=log_file) 
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train), file=log_file)
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train, file=log_file)
        print('.-^-._' * 12, file=log_file)

        print('epoch:', epoch)
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train))
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train)
        print('validation mmd: ', mmd_validation, ', train mmd: ', mmd_train)
        print(f'reg term: {t_reg}')


def train_model(
        epochs: int,
        training_data: torch.Tensor,
        validation_data: torch.Tensor,
        training_mask: torch.Tensor,
        validation_mask: torch.Tensor,
        min_rating: float,
        max_rating: float,
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
        use_kl = False,
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
    """
    optimizer = torch.optim.LBFGS(
             model.parameters(), 
             max_iter=output_every, 
             history_size=history_size,
             lr=learning_rate,
             line_search_fn='strong_wolfe'
             )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = ScipyMinimizer(
            model.parameters(),
            method='L-BFGS-B',
            options={'maxiter': output_every, 'disp': True, 'maxcor': history_size}
            )
    """
    optimizer = torch.optim.Rprop(
           model.parameters(),
           lr=learning_rate
           )


    n_epochs = int(epochs/output_every)
    makedirs(output_path, exist_ok=True)
    log_path= path.join(output_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    loss = _kl_loss if use_kl else _loss 
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
                    log_file,
                    loss,
                    min_rating,
                    max_rating
                    )
            elapsed = time.time() - start
            save_model(model, output_path)
            print(f'Run took {elapsed} seconds')
    return model
