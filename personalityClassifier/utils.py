import torch
import numpy as np
from sklearn.metrics import ndcg_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _compute_ndcg_from_indices(true_items, predicted_items, true_ratings, k=5, num_items=None):
    if num_items is None:
        num_items = max(max(true_items, default=0), max(predicted_items, default=0)) + 1

    y_true = np.zeros(num_items)
    for idx, item in enumerate(true_items):
        y_true[item] = true_ratings[idx]

    y_score = np.zeros(num_items)
    for rank, item in enumerate(predicted_items[:k]):
        y_score[item] = 1/(rank+1)

    return ndcg_score([y_true], [y_score], k=k)

def evaluate_reccomendation_list(X: np.ndarray, X_hat: np.ndarray, n=20, threshold=2.0):
    top_true = [np.where(user_ratings > threshold)[0] for user_ratings in X]
    true_ratings = [X[idx][r] for idx, r in enumerate(top_true)]
    top_predicted = np.argsort(X_hat, axis=1, stable=True)[:,::-1][:, :n]
    return top_true, top_predicted, true_ratings

def compute_ndcg(X_true, X_pred, true_ratings, k, num_items):
    scores = np.zeros((len(X_true)))
    for idx, user in enumerate(X_true):
        scores[idx] = _compute_ndcg_from_indices(user, X_pred[idx], true_ratings[idx], k=k, num_items=num_items)
    return scores.mean()

def compute_itemwise_ndcg(X_true, X_pred, true_ratings, k, num_items):
    scores = np.zeros((len(X_true)))
    for idx, user in enumerate(X_true):
        scores[idx] = _compute_ndcg_from_indices(user, X_pred[idx], true_ratings[idx], k=k, num_items=num_items)
    return scores

