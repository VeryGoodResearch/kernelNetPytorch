import json
import torch
from os import makedirs, path
from .kernel import *
from .model import MultiLayerKernelNet
from .CombinedModel import CombinedResidualModel

def save_model(model: MultiLayerKernelNet, output_path):
    makedirs(output_path+r'/model/', exist_ok=True)
    torch.save(model.state_dict(), output_path+'/model/model_weights.pth')
    params = {
        "n_input": model.n_input,
        "hidden_dims": model.kernel_hidden,
        "lambda_o": model.lambda_o,
        "lambda_2": model.lambda_2,
        "kernel_function": model.kernel_function
    }

    with open(output_path+"/model/model_params.json", "w") as f:
        json.dump(params, f)
