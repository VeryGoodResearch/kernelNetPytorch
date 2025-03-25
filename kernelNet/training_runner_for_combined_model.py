from datetime import datetime
from os import path, makedirs
import torch
import numpy as np
import time
from torchmin import ScipyMinimizer

from .kernel import gaussian_kernel
from .model import MultiLayerKernelNet
from .save_model import save_model
from .CombinedModel import CombinedResidualModel

def _loss(predictions: torch.Tensor,
          truth: torch.Tensor,
          reg_term: torch.Tensor,
          mask: torch.Tensor):
    masked_diff = mask * (truth - predictions)
    loss = torch.sum(masked_diff**2) / 2
    loss = loss + reg_term
    return loss

def _training_iter(model: CombinedResidualModel,
                   epoch: int,
                   t_data: torch.Tensor,
                   t_personality: torch.Tensor,
                   v_data: torch.Tensor,
                   v_personality: torch.Tensor,
                   t_mask: torch.Tensor,
                   v_mask: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   log_file,
                   min_rating: float,
                   max_rating: float):
    assert torch.is_grad_enabled()

    def optimizer_run():
        optimizer.zero_grad()
        # Używamy obu danych: ocen i cech osobowości
        t_pred, t_reg = model.forward(t_data, t_personality)
        loss = _loss(t_pred, t_data, t_reg, t_mask)
        loss.backward()
        return loss

    optimizer.step(optimizer_run)

    # Walidacja
    with torch.no_grad():
        predictions, t_reg = model.forward(t_data, t_personality)
        clipped = torch.clamp(predictions, 1.0, 5.0)
        error_validation = (v_mask * (clipped - v_data) ** 2).sum() / v_mask.sum()  # compute validation error
        error_train = (t_mask * (clipped - t_data) ** 2).sum() / t_mask.sum()  # compute train error
        loss_train = _loss(predictions, t_data, t_reg, t_mask)
        loss_validation = _loss(predictions, v_data, t_reg, v_mask)

        print('.-^-._' * 12, file=log_file)
        print('epoch:', epoch, file=log_file)
        print('validation rmse:', np.sqrt(error_validation), 'train rmse:', np.sqrt(error_train), file=log_file)
        print('validation loss: ', loss_validation, ', train_loss: ', loss_train, file=log_file)
        print('.-^-._' * 12, file=log_file)


def train_model(
        epochs: int,
        training_data: torch.Tensor,
        training_personality: torch.Tensor,  # dane osobowości dla treningu
        validation_data: torch.Tensor,
        validation_personality: torch.Tensor,  # dane osobowości dla walidacji
        training_mask: torch.Tensor,
        validation_mask: torch.Tensor,
        personality_feature_dim: int,  # wymiar cech osobowości
        n_users,
        min_rating: float,
        max_rating: float,
        output_path: str,
        lambda_o: float = 0.013,
        lambda_2: float = 60,
        activation=torch.sigmoid_,
        kernel=gaussian_kernel,
        hidden_dims=500,
        output_every: int = 5,
        history_size: int = 10,
        learning_rate: float = 1,
        user_features_weight: float = 1
):
    n_input = training_data.shape[1]
    rating_model = MultiLayerKernelNet(
        n_input,
        lambda_o=lambda_o,
        lambda_2=lambda_2,
        kernel_hidden=hidden_dims,
        kernel_function=kernel,
        activation=activation,
    )
    model = CombinedResidualModel(rating_model, personality_feature_dim, user_features_weight, residual_hidden_dim=n_users)

    optimizer = ScipyMinimizer(
        model.parameters(),
        method='L-BFGS-B',
        options={'maxiter': output_every, 'disp': True, 'maxcor': history_size}
    )
    n_epochs = int(epochs / output_every)
    makedirs(output_path, exist_ok=True)
    log_path = path.join(output_path, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')

    with open(log_path, 'w+') as log_file:
        for epoch in range(n_epochs):
            start = time.time()
            _training_iter(
                model,
                epoch,
                training_data,
                training_personality,
                validation_data,
                validation_personality,
                training_mask,
                validation_mask,
                optimizer,
                log_file,
                min_rating,
                max_rating
            )
            elapsed = time.time() - start
            #save_model(model, output_path)
            print(f'Run took {elapsed} seconds')
    return model
