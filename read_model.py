import json
import torch
from kernelNet.kernel import *
from kernelNet.model import MultiLayerKernelNet
from kernelNet.training_runner import _loss
from dataLoader.dataLoader import load_ratings_with_personality_traits

def read_model(rating_range: tuple[float, float], path):
    with open(path+"/model/model_params.json", "r") as f:
        params = json.load(f)

    model = MultiLayerKernelNet(
        params["n_input"],
        params["hidden_dims"],
        lambda_o=params["lambda_o"],
        lambda_2=params["lambda_2"],
        kernel_function=gaussian_kernel if params["kernel_function"] == "gaussian_kernel" else sigma_kernel
    )
    model.load_state_dict(torch.load(path+'/model/model_weights.pth'))
    model.eval()
    train_data, validation_data, _, _ = load_ratings_with_personality_traits(path='data/personality-isf2018/',
                                                                             transpose=False, valfrac=0.1,
                                                                             seed=1)
    train_data = torch.from_numpy(train_data)
    print("real, predicted")
    test_vector = train_data[0]
    with torch.no_grad():
        predictions, t_reg = model.forward(test_vector)
        for i in range(len(predictions)):
            print(str(test_vector[i])+ ", " + str(predictions[i]))

        loss = _loss(predictions, test_vector, t_reg, test_vector)
        print("loss " + str(loss))

        mask = torch.greater_equal(test_vector, 0.5).float()
        clipped = torch.clamp(predictions, rating_range[0], rating_range[1])
        error_validation = (mask * (clipped - test_vector) ** 2).sum() / mask.sum()
        print("error_validation " + str(error_validation))


read_model((0.5, 5.0), "output")
#read_model((0, 5), "output")