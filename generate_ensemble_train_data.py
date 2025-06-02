import torch
import numpy as np

from dataLoader.dataLoader import load_mid_movies_with_personality_traits, load_ratings_with_personality_traits, load_top_movies_with_personality_traits
from ensemble.ensemble_model import EnsembleModel
from ensemble.utils import get_relevant_items
from personalityClassifier.load_encoder_decoder import load_decoder
from personalityClassifier.utils import compute_ndcg
from priors.mid_prior import MidPrior
from priors.top_prior import TopPrior

def main():
    model, X_train, _, p_train, _, train_mask, _ = load_ensemble_model()
    split_index = 638
    train_data_first = model.generate_training_data(p_train[:split_index], X_train[:split_index], train_mask[:split_index])
    print(train_data_first.shape)
    train_data_second = model.generate_training_data(p_train[split_index:], X_train[split_index:], train_mask[split_index:])
    print(train_data_second.shape)
    del model
    train_data = np.concatenate((train_data_first, train_data_second))
    del train_data_first
    del train_data_second
    print(train_data.shape)
    print(train_data[0])

def load_ensemble_model():
    # Data loading
    X_train, X_test, p_train, p_test = load_ratings_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True)
    _, _, _, _, top_indices = load_top_movies_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True)
    _, _, _, _, mid_indices = load_mid_movies_with_personality_traits(path='data/personality-isf2018/', valfrac=0.1, transpose=False, feature_classification=True, n=250)
    train_mask = torch.greater_equal(torch.from_numpy(X_train), 0.1).float().squeeze()
    test_mask = torch.greater_equal(torch.from_numpy(X_test), 0.1).float().squeeze()
    # Model loading
    top_dec = load_decoder('output_top_autoencoder')
    top_prior = TopPrior(5, 100, 0.1)
    top_prior.load_state_dict(torch.load('output_top_prior/model.torch', weights_only=True))
    mid_dec = load_decoder('output_mid_autoencoder')
    mid_prior = MidPrior(5, 400, 0.1)
    mid_prior.load_state_dict(torch.load('output_mid_prior/model.torch', weights_only=True))
    # Pytorch cooking
    p_train = torch.from_numpy(p_train)/7.0
    p_test = torch.from_numpy(p_test)/7.0
    p_train = p_train.squeeze()
    p_test = p_test.squeeze()
    model = EnsembleModel(top_dec, top_prior, mid_dec, mid_prior, X_train, p_train, top_indices, mid_indices)
    return model, X_train, X_test, p_train, p_test, train_mask, test_mask


if __name__ == '__main__':
    main()
