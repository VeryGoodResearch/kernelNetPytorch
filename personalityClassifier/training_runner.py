# This module provides some helper utilities used for training the model
from datetime import datetime
from os import path, makedirs
from numpy.strings import upper
import torch
import numpy as np
import time
from .save_encoder_decoder import save_encoder, save_decoder
from torchmin import ScipyMinimizer
from sklearn.metrics import ndcg_score

from personalityClassifier.autoencoder import KernelNetAutoencoder
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.utils import get_device

def _evaluate_reccomendation_list(X: np.ndarray, X_hat: np.ndarray, n=20):
    top_true = np.flip(np.argsort(X, axis=1)[:,-n:], axis=1)
    top_predicted = np.flip(np.argsort(X_hat, axis=1)[:,-n:], axis=1)
    return top_true, top_predicted

def _loss(predictions: torch.Tensor, 
          truth: torch.Tensor, 
          reg_term: torch.Tensor,
          mask: torch.Tensor,
          kl_reg: torch.Tensor,
          kl_reg_lambda: float):
    masked_diff = mask * (truth - predictions) / truth.shape[0]
    masked_loss = torch.sum(masked_diff**2) / 2
    loss = masked_loss + reg_term + (kl_reg_lambda*kl_reg)
    return loss


def _training_iter(model: KernelNetAutoencoder, 
                  epoch: int,
                  t_data: torch.Tensor, 
                  v_data: torch.Tensor,
                  t_mask: torch.Tensor,
                  v_mask: torch.Tensor,
                  optimizer: torch.optim.Optimizer,
                   kl_reg_lambda: float,  
                   log_file,
                   min_rating: float,
                   max_rating: float,
                   verbose = 2):
    # Not the greatest idea to run it otherwise
    assert torch.is_grad_enabled()
    def optimizer_run():
        optimizer.zero_grad()
        t_pred, t_reg, kl_reg = model.forward(t_data)
        loss = _loss(t_pred, t_data, t_reg, t_mask, kl_reg, kl_reg_lambda)
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
        predictions, t_reg, t_kl_reg = model.forward(t_data)
        if verbose > 1:
            print('Training data:')
            print(t_data[0])
            print('Model predicted')
            print(predictions[0])
            print('Some other random prediction')
            print(predictions[3])
        error_train = ((predictions - t_data) ** 2).sum() / predictions.numel()   # compute train error
        loss_train = _loss(predictions, t_data, t_reg, t_mask, t_kl_reg, kl_reg_lambda)
        predictions = predictions.clip(min_rating, max_rating) # We want to emulate the original masked mse as closely as possible
        error_train_observed = (t_mask * (predictions - t_data)**2).sum() / t_mask.sum()
        # Validation stuff
        v_predictions, t_reg, v_kl_reg = model.forward(v_data)
        error_validation = ((v_predictions - v_data) ** 2).sum() / v_predictions.numel()  # compute validation error
        loss_validation = _loss(v_predictions, v_data, t_reg, v_mask, v_kl_reg, kl_reg_lambda)
        v_predictions = v_predictions.clip(min_rating, max_rating) # We want to emulate the original masked mse as closely as possible
        error_validation_observed = (v_mask * (v_predictions - v_data)**2).sum() / v_mask.sum()

        print('.-^-._' * 12, file=log_file)
        print('epoch:', epoch, file=log_file) 
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train), file=log_file)
        print('validation observed rmse: ', np.sqrt(error_validation_observed), ', train observed rmse: ', np.sqrt(error_train_observed), file=log_file)
        print('-' * 50, file=log_file)
        print('validation mse: ', error_validation, ', train mse: ', error_train, file=log_file)
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train, file=log_file)
        print(f'Reg term: {t_reg}', file=log_file)
        print('.-^-._' * 12, file=log_file)

        print('.-^-._' * 12) 
        print('epoch:', epoch) 
        print(f'loss: {loss_train}')
        if verbose > 0:
            print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train))
            print('validation observed rmse: ', np.sqrt(error_validation_observed), ', train observed rmse: ', np.sqrt(error_train_observed))
            print('-' * 50, file=log_file)
            print('validation mse: ', error_validation, ', train mse: ', error_train) 
            print('validation loss: ', loss_validation, ', train_loss: ', loss_train)
            print(f'Reg term: {t_reg}, train kl reg: {t_kl_reg*kl_reg_lambda}, validation kl reg: {v_kl_reg*kl_reg_lambda}')
            train_lists = _evaluate_reccomendation_list(t_data.detach().numpy(), (predictions*t_mask).detach().numpy())
            validation_lists = _evaluate_reccomendation_list(v_data.detach().numpy(), (v_predictions*v_mask).detach().numpy())
            print(f'TOP 20 - Validation ndcg: {ndcg_score(*validation_lists)}, train ndcg: {ndcg_score(*train_lists)}')
            train_lists = _evaluate_reccomendation_list(t_data.detach().numpy(), (predictions*t_mask).detach().numpy(), n=5)
            validation_lists = _evaluate_reccomendation_list(v_data.detach().numpy(), (v_predictions*v_mask).detach().numpy(), n=5)
            print(f'TOP 5 - Validation ndcg: {ndcg_score(*validation_lists)}, train ndcg: {ndcg_score(*train_lists)}')
        print('.-^-._' * 12)

def train_model(
        epochs: int,
        training_data: torch.Tensor,
        validation_data: torch.Tensor,
        training_mask: torch.Tensor,
        validation_mask: torch.Tensor,
        output_path: str,
        min_rating: float,
        max_rating: float,
        lambda_o: float = 0.0013,
        lambda_2: float = 6,
        activation = torch.sigmoid_,
        kernel = gaussian_kernel,
        hidden_dims = 50,
        output_every: int = 5,
        # WARNING WARNING WARNING 
        # IF YOU HAVE BAD PC (LOW MEMORY) THEN TUNE THIS THING DOWN, OTHERWISE IT WILL PROBABLY EXPLODE
        history_size: int = 10,
        learning_rate: float = 1,
        kl_activation: float = 0.02,
        kl_lambda: float = 1e-6,
        verbose = 2):
    device = get_device()
    n_input = training_data.shape[1]
    model = KernelNetAutoencoder(
            n_input,
            lambda_o=lambda_o,
            lambda_2=lambda_2,
            kernel_hidden=hidden_dims,
            kernel_function=kernel,
            activation=activation,
            kl_activation=kl_activation
            ).to(device)
    """
    optimizer = torch.optim.Rprop(
           model.parameters(),
           lr=learning_rate
           )
    optimizer = torch.optim.LBFGS(
             model.parameters(), 
             max_iter=output_every, 
             history_size=history_size,
             lr=learning_rate,
             line_search_fn='strong_wolfe'
             )
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
            )
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
                    kl_lambda,
                    log_file,
                    min_rating,
                    max_rating,
                    verbose
                    )
            elapsed = time.time() - start
            print(f'Run took {elapsed} seconds')
    save_encoder(model.enc, output_path)
    save_decoder(model.dec, output_path)
    return model
