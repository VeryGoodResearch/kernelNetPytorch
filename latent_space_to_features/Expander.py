import torch
import torch.nn as nn
class Block(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return x + self.block(x)

class Expander(nn.Module):
    def __init__(self, input_dim, target_dim):
        super().__init__()
        self.start = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU()
        )
        self.res1 = Block(512)
        self.res2 = Block(512)
        self.out = nn.Linear(512, target_dim)

    def forward(self, x):
        x = self.start(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.out(x)
