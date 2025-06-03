import torch
from torch import nn
import os
import tqdm

from priors.utils import VectorToOffsetMatrix

class PersonalityModeller(nn.Module):

    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
                VectorToOffsetMatrix(),
                nn.Conv2d(1, 20, kernel_size=3, stride=2),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(20*2*2, 64),
                nn.Dropout(0.2),
                nn.Sigmoid(),
                nn.Linear(64, output_size),
                nn.Sigmoid()
                )

    def forward(self, X: torch.Tensor):
        return self.layers.forward(X)


def train_personality_modeller(X_train, X_test, y_train, y_test, epochs=500, lr=0.001, output='output_personality_modeller'):
    model = PersonalityModeller(X_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    pbar = tqdm.tqdm(range(epochs), desc="Training", dynamic_ncols=True)
    criterion = nn.MSELoss()

    for epoch in pbar:
        optimizer.zero_grad()
        yhat= model.forward(X_train)
        loss = criterion(yhat, y_train)
        loss.backward(retain_graph=True)
        optimizer.step()
        with torch.no_grad():
            yhat_train= model(X_train)
            yhat_test = model(X_test)
            loss_train = criterion(y_train, yhat_train)
            loss_val = criterion(y_test, yhat_test)

        pbar.set_postfix({
            "Train loss": f"{loss_train:.4f}",
            "Val loss": f"{loss_val:.4f}",
        })
    os.makedirs(output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output, 'model.torch'))
    return model
