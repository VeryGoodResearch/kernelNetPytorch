# Some testing stuff for model
import torch
from kernelNet.helpers import generate_data_with_missing
from kernelNet.training_runner import train_model

def main():
    n = 1000
    m = 500
    train_data = generate_data_with_missing(n, m, missing_prob=0.5)
    val_data = generate_data_with_missing(n, m, missing_prob=0.5)
    train_mask = torch.greater(train_data, 1e-12)
    validation_mask = torch.greater(val_data, 1e-12)
    epochs = 500
    model = train_model(
            epochs,
            train_data,
            val_data,
            train_mask,
            validation_mask
            )
    print(model)

if __name__ == '__main__':
    main()
