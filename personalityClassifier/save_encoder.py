import json
import torch
from os import makedirs, path
from .kernel import *
from .encoder import Encoder
from .decoder import Decoder

def save_encoder(model: Encoder, output_path):
    makedirs(output_path+r'/encoder/', exist_ok=True)
    torch.save(model.state_dict(), output_path+'/encoder/encoder_weights.pth')

    params = {
        "n_input": model.n_input,
        "hidden_dims": model.kernel_hidden,
        "lambda_o": model.lambda_o,
        "lambda_2": model.lambda_2,
        "kernel_function": model.kernel_function,
        "activation": type(model.activation).__name__,
        "kernel_hidden": model.kernel_hidden,
    }
    with open(output_path+"/encoder/encoder_params.json", "w") as f:
        json.dump(params, f)

def save_decoder(model: Decoder, output_path):
    makedirs(output_path+r'/decoder/', exist_ok=True)
    torch.save(model.state_dict(), output_path+'/decoder/decoder_weights.pth')

    params = {
        "n_output": model.n_output,
        "kernel_hidden": model.kernel_hidden,
        "lambda_o": model.lambda_o,
        "lambda_2": model.lambda_2,
        "hidden_dims": model.hidden_dims,
        "kernel_function": model.kernel_function,
        "activation": type(model.activation).__name__,
    }
    with open(output_path+"/decoder/decoder_params.json", "w") as f:
        json.dump(params, f)