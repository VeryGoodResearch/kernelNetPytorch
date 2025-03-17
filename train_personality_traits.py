# Some testing stuff for model
import torch
import time
from kernelNet.training_runner import train_model
from kernelNet.kernel import gaussian_kernel

from dataLoader.dataLoader import load_ratings_with_personality_traits


def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
        path='../personality-isf2018/', valfrac=0.1, seed=seed, transpose=False)

    train_data = torch.from_numpy(train_data)
    validation_data = torch.from_numpy(validation_data)
    train_user_features = torch.from_numpy(train_user_features)
    valid_user_features = torch.from_numpy(valid_user_features)
    train_mask = torch.greater_equal(train_data, 1).float()
    validation_mask = torch.greater_equal(validation_data, 1).float()
    print(f'Training shape: {train_data.shape}, validation shape: {validation_data.shape}')
    print(f'Training mask: {train_mask.shape}, validation mask shape: {validation_mask.shape}')

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
            lambda_o=0.013,
            lambda_2=60,
            history_size=10,
            output_every=50,
            hidden_dims=50,
            output_path='./output_personality/',
            min_rating=1.0,
            max_rating=5.0)
    print(model)

if __name__ == '__main__':
    main()
