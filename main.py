# Some testing stuff for model
import torch
import time
from kernelNet.training_runner import train_model
from kernelNet.kernel import gaussian_kernel
from dataLoader.dataLoader import load_data


def main():
    seed = int(time.time())
    torch.seed()
    train_data, validation_data = load_data('./data/ml-1m/ratings.dat', delimiter='::', transpose=True, valfrac=0.1, seed=seed)
    train_data = torch.from_numpy(train_data)
    validation_data = torch.from_numpy(validation_data)
    train_mask = torch.gt(train_data, 1e-12).float()
    validation_mask = torch.gt(validation_data, 1e-12).float()
    print(f'Training shape: {train_data.shape}, validation shape: {validation_data.shape}')
    print(f'Number of training samples: {train_mask.sum()}, number of validation samples: {validation_mask.sum()}')
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
            output_every=50
            )
    print(model)

if __name__ == '__main__':
    main()
