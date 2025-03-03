# Some testing stuff for model
import torch
import time
from kernelNet.training_runner import train_model
from dataLoader.dataLoader import load_data


def main():
    seed = int(time.time())
    torch.seed()
    train_data, test_data, validation_data = load_data('./data/ml-1m/ratings.dat', delimiter='::', transpose=True, valfrac=0.1, seed=seed)
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    validation_data = torch.from_numpy(validation_data)
    train_mask = torch.greater(train_data, 1e-12)
    validation_mask = torch.greater(validation_data, 1e-12)
    epochs = 500
    model = train_model(
            epochs,
            train_data,
            validation_data,
            train_mask,
            validation_mask,
            lambda_o=0.021,
            lambda_2=80,
            history_size=15
            )
    print(model)

if __name__ == '__main__':
    main()
