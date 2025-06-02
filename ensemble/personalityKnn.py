import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_personality(personality: np.ndarray, ratings: np.ndarray, personalities: np.ndarray, n = 5):
    sims = cosine_similarity(personality, personalities)
    top = np.argsort(-sims, axis=1)[:, :n]
    pred_ratings = ratings[top].mean(axis=1)
    return pred_ratings

