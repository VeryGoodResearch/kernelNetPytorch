# Some testing stuff for model
import torch
import time
import numpy as np

from dataLoader.dataLoader import load_ratings_with_personality_traits
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.training_runner import train_model
from personalityClassifier.utils import get_device


def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
        path='data/personality-isf2018/', valfrac=0.1, seed=seed, transpose=False, feature_classification=True)
    device = get_device()
    train_data = torch.from_numpy(train_data).to(device).squeeze()
    validation_data = torch.from_numpy(validation_data).to(device).squeeze()
    train_mask = torch.greater_equal(train_data, 1).float()
    validation_mask = torch.greater_equal(validation_data, 1).float()
    sparsity_factor = (train_mask.numel()/train_mask.sum())-1
    print(f'Training shape: {train_data.shape}, validation shape: {validation_data.shape}')
    print(f'Training mask: {train_mask.shape}, validation mask shape: {validation_mask.shape}')
    print(f'Sparsity factor: {sparsity_factor}')
    output_every=50
    epochs = output_every * 30
    model = train_model(
            epochs,
            train_data,
            validation_data,
            train_mask,
            validation_mask,
            kernel=gaussian_kernel,
            activation=torch.nn.Sigmoid(),
            lambda_o=0.026,
            lambda_2=120,
            history_size=5,
            output_every=50,
            hidden_dims=500,
            output_path='./output_autoencoder/',
            learning_rate=1.0)
    print(model)

if __name__ == '__main__':
    main()
    #train_without_personality_traits()
