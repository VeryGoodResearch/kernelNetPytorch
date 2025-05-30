import torch
import time
import numpy as np

from dataLoader.dataLoader import load_top_movies_with_personality_traits
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.load_encoder_decoder import load_decoder, load_encoder
from personalityClassifier.training_runner import train_model
from personalityClassifier.utils import get_device
from priors.top_prior import train_top_prior

def main():
    seed = int(time.time())
    torch.seed()
    train_data, test_data, X_train, X_test, _ = load_top_movies_with_personality_traits(
        path='data/personality-isf2018/', valfrac=0.1, seed=seed, transpose=False, feature_classification=True)
    device = get_device()
    train_data = torch.from_numpy(train_data).to(device).squeeze()
    test_data = torch.from_numpy(test_data).to(device).squeeze()
    train_mask = torch.greater_equal(train_data, 0.1).float()
    test_mask = torch.greater_equal(test_data, 0.1).float()
    encoder = load_encoder('output_top_autoencoder')
    y_train = encoder.forward(train_data)[0]
    y_test = encoder.forward(test_data)[0]
    X_train = torch.from_numpy(X_train).to(device)/7.0
    X_test = torch.from_numpy(X_test).to(device)/7.0
    prior = train_top_prior(X_train.squeeze(), X_test.squeeze(), y_train, y_test, epochs=1000, learning_rate = 0.01)
    decoder = load_decoder('output_top_autoencoder')
    preds = decoder.forward(prior.forward(X_test)[0])[0]*test_mask
    loss = torch.sqrt(((preds-test_data)**2).mean())
    print(f'Validation reconstructed ratings rmse: {loss.item()}')


if __name__ == '__main__':
    main()
