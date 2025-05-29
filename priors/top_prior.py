import torch
from torch import nn
import torch.nn.functional as F
import tqdm

from priors.utils import VectorToOffsetMatrix, kl_bernoulli

class TopPrior(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.Sigmoid(),
                nn.Linear(16, 64),
                nn.Sigmoid(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size),
                nn.Sigmoid()
                )

    def forward(self, X):
        return self.layers.forward(X)
        

def train_top_prior(X_train, X_test, y_train, y_test, epochs, learning_rate, output='./output_top_prior'):
    model = TopPrior(X_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    pbar = tqdm.tqdm(range(epochs), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        optimizer.zero_grad()
        yhat = model.forward(X_train)
        kl_bernoulli(yhat, y_train)
        optimizer.step()
        with torch.no_grad():
            yhat_train = model(X_train)
            yhat_test = model(X_test)
            rmse_train = torch.sqrt(F.mse_loss(yhat_train, y_train)).item()
            rmse_test = torch.sqrt(F.mse_loss(yhat_test, y_test)).item()
            loss_train = kl_bernoulli(y_train, yhat_train).mean()
            loss_val = kl_bernoulli(y_test, yhat_test).mean()
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

