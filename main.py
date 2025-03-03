# Some testing stuff for model
import torch
from kernelNet.training_runner import train_model
from dataLoader.dataLoader import load_data


def main():
    train_data, test_data, validation_data = load_data('./data/ml-1m/ratings.dat', delimiter='::', transpose=True, valfrac=0.1)
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
            validation_mask
            )
    print(model)

if __name__ == '__main__':
    main()
