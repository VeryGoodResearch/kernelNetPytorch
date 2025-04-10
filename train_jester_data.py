# Some testing stuff for model
import torch
import time
from kernelNet.training_runner import train_model
from kernelNet.kernel import gaussian_kernel
from dataLoader.dataLoader import load_movie_lens, load_jester_data_xls


def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data = load_jester_data_xls(r'..\jester-data-1\jester-data-1.xls', transpose=True, valfrac=0.1, seed=seed)
    train_data = torch.from_numpy(train_data)
    validation_data = torch.from_numpy(validation_data)
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
            lambda_o=0.03,
            lambda_2=60,
            history_size=10,
            hidden_dims = 50,
            output_every=50,
            output_path='./output_jester/',
            min_rating=1.0,
            max_rating=21.0)
    print(model)

if __name__ == '__main__':
    main()