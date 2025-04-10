import json
import torch
from os import makedirs, path
from .kernel import *
from .encoder import Encoder

def save_encoder(model: Encoder, output_path):
    makedirs(output_path+r'/encoder/', exist_ok=True)
    torch.save(model.state_dict(), output_path+'/encoder/encoder_weights.pth')
    params = {
        "n_input": model.n_input,
        "hidden_dims": model.kernel_hidden,
        "lambda_o": model.lambda_o,
        "lambda_2": model.lambda_2,
        "kernel_function": model.kernel_function,
        "activation": type(model.activation).__name__
    }

    with open(output_path+"/encoder/encoder_params.json", "w") as f:
        json.dump(params, f)