# Some testing stuff for model
import torch
import time
from kernelNet.training_runner_for_combined_model import train_model
from kernelNet.training_runner import train_model as original_train

from kernelNet.kernel import gaussian_kernel

from dataLoader.dataLoader import load_ratings_with_personality_traits


def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
        path='../personality-isf2018/', valfrac=0.1, seed=seed, feature_classification = False, transpose=False)

    train_data = torch.from_numpy(train_data)
    validation_data = torch.from_numpy(validation_data)
    train_user_features = torch.from_numpy(train_user_features)
    n_users = train_user_features.shape[0]
    n_user_features = train_user_features.shape[1]
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
            train_user_features,
            validation_data,
            valid_user_features,
            train_mask,
            validation_mask,
            n_user_features,
            n_users,
            kernel=gaussian_kernel,
            activation=torch.nn.Sigmoid(),
            lambda_o=0.013,
            lambda_2=60,
            history_size=10,
            output_every=50,
            hidden_dims=50,
            output_path='./output_personality/',
            min_rating=0.5,
            max_rating=5.0,
            user_features_weight = 0.5)
    print(model)

def train_without_personality_traits():
    seed = int(time.time())
    torch.seed()
    # In case transpose=False each sample represents a user
    train_data, validation_data, _, _ = load_ratings_with_personality_traits(path='../personality-isf2018/', transpose=False, valfrac=0.1,
                                                  seed=seed)
    train_data = torch.from_numpy(train_data)
    validation_data = torch.from_numpy(validation_data)
    train_mask = torch.greater_equal(train_data, 1).float()
    validation_mask = torch.greater_equal(validation_data, 1).float()
    print(f'Training shape: {train_data.shape}, validation shape: {validation_data.shape}')
    print(f'Training mask: {train_mask.shape}, validation mask shape: {validation_mask.shape}')

    output_every = 50
    epochs = output_every * 30
    model = original_train(
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
        min_rating=0.5,
        max_rating=5.0)
    print(model)

if __name__ == '__main__':
    main()
    #train_without_personality_traits()
