import torch
import numpy as np

from dataLoader.dataLoader import load_mid_movies_with_personality_traits, load_ratings_with_personality_traits, load_top_movies_with_personality_traits
from ensemble.ensemble_model import EnsembleModel
from ensemble.personality_modeller import PersonalityModeller
from ensemble.utils import get_relevant_items
from personalityClassifier.load_encoder_decoder import load_decoder
from personalityClassifier.utils import compute_ndcg, calculate_hitrate
from priors.mid_prior import MidPrior
from priors.top_prior import TopPrior


def main():
    model, _, X_test, _, p_test, _, test_mask = load_ensemble_model()
    with torch.no_grad():
        v_hat = model.forward(p_test)
        relevant, rates = get_relevant_items(X_test.squeeze())
        v_andy = compute_ndcg(relevant, v_hat.cpu().long(), rates, num_items=X_test.shape[2], k=20)
        print(f'Test ndcg@20: {v_andy}')
        v_andy = compute_ndcg(relevant, v_hat.cpu().long(), rates, num_items=X_test.shape[2], k=10)
        print(f'Test ndcg@10: {v_andy}')
        v_andy = compute_ndcg(relevant, v_hat.cpu().long(), rates, num_items=X_test.shape[2], k=5)
        print(f'Test ndcg@5: {v_andy}')
        hr = calculate_hitrate(relevant, v_hat.cpu().long(), 20)
        print(f'Test hr@20: {hr}')
        hr = calculate_hitrate(relevant, v_hat.cpu().long(), 10)
        print(f'Test hr@10: {hr}')
        hr = calculate_hitrate(relevant, v_hat.cpu().long(), 5)
        print(f'Test hr@5: {hr}')


def load_ensemble_model():
    # Data loading
    X_train, X_test, p_train, p_test = load_ratings_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True)
    _, _, _, _, top_indices = load_top_movies_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True)
    _, _, _, _, mid_indices = load_mid_movies_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True, n=0)
    train_mask = torch.greater_equal(torch.from_numpy(X_train), 0.1).float().squeeze()
    test_mask = torch.greater_equal(torch.from_numpy(X_test), 0.1).float().squeeze()
    # Model loading
    top_dec = load_decoder('output_top_autoencoder')
    top_prior = TopPrior(5, 100, 0.1)
    top_prior.load_state_dict(torch.load('output_top_prior/model.torch', weights_only=True))
    mid_dec = load_decoder('output_mid_autoencoder')
    mid_prior = MidPrior(5, 400, 0.1)
    mid_prior.load_state_dict(torch.load('output_mid_prior/model.torch', weights_only=True))
    personality_modeller = PersonalityModeller(5, 3)
    personality_modeller.load_state_dict(torch.load('output_personality_modeller/model.torch'))
    # Pytorch cooking
    p_train = torch.from_numpy(p_train)/7.0
    p_test = torch.from_numpy(p_test)/7.0
    p_train = p_train.squeeze()
    p_test = p_test.squeeze()
    model = EnsembleModel(top_dec, top_prior, mid_dec, mid_prior, X_train, p_train, top_indices, mid_indices, 20, personality_modeller)
    return model, X_train, X_test, p_train, p_test, train_mask, test_mask

if __name__ == '__main__':
    main()
