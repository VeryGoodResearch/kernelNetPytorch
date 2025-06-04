import torch
import time
import numpy as np

from dataLoader.dataLoader import load_mid_movies_with_personality_traits
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.training_runner import train_model
from personalityClassifier.utils import get_device

def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data, _, _, _ = load_mid_movies_with_personality_traits(
        path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True, n=0)
    device = get_device()
    train_data = torch.from_numpy(train_data).to(device).squeeze()
    validation_data = torch.from_numpy(validation_data).to(device).squeeze()
    train_mask = torch.greater_equal(train_data, .1).float()
    validation_mask = torch.greater_equal(validation_data, .1).float()
    print(f'Training shape: {train_data.shape}, validation shape: {validation_data.shape}')
    print(f'Training mask: {train_mask.shape}, validation mask shape: {validation_mask.shape}')
    print(f'Device: {device}')
    output_every=50
    epochs = 500
    epochs = output_every * epochs
    model = train_model(
            epochs,
            train_data,
            validation_data,
            train_mask,
            validation_mask,
            kernel=gaussian_kernel,
            activation=torch.nn.Sigmoid(),
            lambda_o=7e-7,
            lambda_2=5e-5,
            history_size=5,
            output_every=50,
            hidden_dims=400,
            output_path='./output_mid_autoencoder/',
            learning_rate=0.0003,
            kl_activation=0.1,
            kl_lambda=2e-4,
            min_rating=0.5,
            max_rating=5.0,
            verbose=1)

    print(model)

if __name__ == '__main__':
    main()
