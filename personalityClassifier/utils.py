import torch
import numpy as np
from sklearn.metrics import ndcg_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _compute_ndcg_from_indices(true_items, predicted_items, k=5, num_items=None):
    if num_items is None:
        num_items = max(max(true_items, default=0), max(predicted_items, default=0)) + 1

    y_true = np.zeros(num_items)
    for item in true_items:
        y_true[item] = 1

    y_score = np.zeros(num_items)
    for rank, item in enumerate(predicted_items[:k]):
        y_score[item] = k - rank

    return ndcg_score([y_true], [y_score], k=k)

def compute_ndcg(X_true, X_pred, k, num_items):
    scores = np.zeros((len(X_true)))
    for idx, user in enumerate(X_true):
        scores[idx] = _compute_ndcg_from_indices(user, X_pred[idx], k=k, num_items=num_items)
    return scores.mean()
