import torch
from torch import nn
import torch.nn.functional as F
import tqdm

from priors.utils import VectorToOffsetMatrix, kl_bernoulli

class MidPrior(nn.Module):
    def __init__(self, input_size, output_size, sparse_activation) -> None:
        super().__init__()
        self.kl_activation = sparse_activation
        self.layers = nn.Sequential(
                VectorToOffsetMatrix(),
                nn.Conv2d(1, 10, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(10*2*2, 128),
                nn.Sigmoid(),
                nn.Linear(128, 512),
                nn.Sigmoid(),
                nn.Dropout(0.2),
                nn.Linear(512, output_size),
                nn.Sigmoid()
                )

    def __compute_KL_reg(self, activations: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        a = activations.clamp(min=eps, max=1 - eps)
        g = torch.full_like(a, self.kl_activation).clamp(min=eps, max=1 - eps)
        reg = a * torch.log(a / g) + (1 - a) * torch.log((1 - a) / (1 - g))
        if torch.isnan(reg).any():
            print("KL reg has NaNs:", a.min(), a.max())
        term = reg.sum() / a.shape[0]
        return term

    def forward(self, X):
        res = self.layers.forward(X)
        reg = self.__compute_KL_reg(res)
        return res, reg
        

def train_mid_prior(X_train, X_test, y_train, y_test, epochs, learning_rate, output='./output_top_prior', sparse_lambda = 1e-6):
    print(y_train[0])
    model = MidPrior(X_train.shape[1], y_train.shape[1], 0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    pbar = tqdm.tqdm(range(epochs), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        optimizer.zero_grad()
        yhat, reg = model.forward(X_train)
        loss = kl_bernoulli(yhat, y_train) + (sparse_lambda*reg)
        loss.backward(retain_graph=True)
        optimizer.step()
        with torch.no_grad():
            yhat_train, regt = model(X_train)
            yhat_test, regv = model(X_test)
            rmse_train = torch.sqrt(F.mse_loss(yhat_train, y_train)).item()
            rmse_test = torch.sqrt(F.mse_loss(yhat_test, y_test)).item()
            loss_train = kl_bernoulli(y_train, yhat_train) + regt*sparse_lambda
            loss_val = kl_bernoulli(y_test, yhat_test) + regv*sparse_lambda
            ma_train = yhat_train.mean().item()
            ma_test = yhat_test.mean().item()

        pbar.set_postfix({
            "Train RMSE": f"{rmse_train:.4f}",
            "Val RMSE": f"{rmse_test:.4f}",
            "Train loss": f"{loss_train:.4f}",
            "Val loss": f"{loss_val:.4f}",
            "Train ma": f"{ma_train:.4f}",
            "Val ma": f"{ma_test:.4f}",
        })
    return model

