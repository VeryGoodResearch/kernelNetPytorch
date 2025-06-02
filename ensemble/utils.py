import numpy as np

def get_relevant_items(X):
    top_true = [np.where(user_ratings > 0.1)[0] for user_ratings in X]
    true_ratings = [X[idx][r] for idx, r in enumerate(top_true)]
    return top_true, true_ratings
