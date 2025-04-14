import json
from personalityClassifier.encoder import Encoder
from personalityClassifier.kernel import *
from torch import nn
import torch
import time

from dataLoader.dataLoader import load_ratings_with_personality_traits
from personalityClassifier.kernel import gaussian_kernel
from personalityClassifier.utils import get_device

def load_encoder(model_path):
    with open(model_path + "/encoder/encoder_params.json", 'r') as f:
        params = json.load(f)

    activation_map = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "sigmoid": torch.sigmoid,
        "sigmoid_": torch.sigmoid_,
    }

    kernel_map = {
        "gaussian_kernel": gaussian_kernel,
        "quadratic_gaussian_kernel": quadratic_gaussian_kernel,
        "sigma_kernel": sigma_kernel
    }

    model = Encoder(
        n_input=params["n_input"],
        kernel_hidden=params["hidden_dims"],
        lambda_o=params["lambda_o"],
        lambda_2=params["lambda_2"],
        kernel_function=kernel_map[params["kernel_function"]],
        activation=activation_map[params["activation"]]
    )
    model.load_state_dict(torch.load(model_path + '/encoder/encoder_weights.pth', map_location=torch.device('cpu')))

    return model

def test_loading():
    model = load_encoder("../output_autoencoder/")

    print(model.n_input)
    print(model.kernel_hidden)
    print(model.lambda_o)

    seed = int(time.time())
    torch.seed()
    train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
        path='../data/personality-isf2018/', valfrac=0.1, seed=seed, transpose=False, feature_classification=True)
    device = get_device()
    train_data = torch.from_numpy(train_data).to(device).squeeze()
    validation_data = torch.from_numpy(validation_data).to(device).squeeze()

    print("original: ", train_data[0])
    print("shape of original: ", train_data[0].shape)
    y, total_reg = model.forward(train_data[0])
    print("compressed: ", y)
    print(y.shape)