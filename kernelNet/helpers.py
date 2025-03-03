# Some helper stuff
import torch


def generate_data_with_missing(n, m, missing_prob=0.8):
    # Generate ratings 1 to 5 for all entries.
    data = torch.randint(low=1, high=6, size=(n, m))
    # Create a mask for missing entries.
    missing_mask = torch.rand(n, m) < missing_prob
    data[missing_mask] = 0
    return data

