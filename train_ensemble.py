import torch
import numpy as np
from torch.nn.functional import softmax

from dataLoader.dataLoader import load_mid_movies_with_personality_traits, load_ratings_with_personality_traits, load_top_movies_with_personality_traits
from ensemble.ensemble_model import EnsembleModel
from ensemble.personality_modeller import train_personality_modeller
from ensemble.utils import get_relevant_items
from personalityClassifier.load_encoder_decoder import load_decoder
from personalityClassifier.utils import compute_ndcg
from priors.mid_prior import MidPrior
from priors.top_prior import TopPrior

def generate_training_data():
    model, X_train, X_test, p_train, p_test, train_mask, test_mask = load_ensemble_model()
    with torch.no_grad():
        train_out =  model.generate_training_data(p_train, X_train).float()
        print(f'Mean ndcgs for train: {torch.mean(train_out, dim=0)}')
        row_sums = train_out.sum(dim=1, keepdim=True).float()
        train_data = train_out / row_sums
        train_data[(row_sums==0).squeeze()==1] = torch.zeros(3)
        test_out = model.generate_training_data(p_test, X_test, exclusive=False, detailed=True).float()
        print(f'Mean ndcgs for test: {torch.mean(test_out, dim=0)}')
        row_sums = test_out.sum(dim=1, keepdim=True).float()
        test_data = test_out / row_sums
        test_data[(row_sums==0).squeeze()==1] = torch.zeros(3)
        return p_train.float(), p_test.float(), train_data, test_data

def main():
    X_train, X_test, y_train, y_test = generate_training_data() 
    print(X_train.shape)
    print(y_train.shape)
    print(y_train[0])
    print(torch.mean(y_train, dim=0))
    print(torch.mean(y_test, dim=0))
    model = train_personality_modeller(X_train, X_test, y_train, y_test)
    pred = model(X_test[:10])
    print(f'True probs: {y_test[:10]}')
    print(f'Predicted: {pred}')


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
    # Pytorch cooking
    p_train = torch.from_numpy(p_train)/7.0
    p_test = torch.from_numpy(p_test)/7.0
    p_train = p_train.squeeze()
    p_test = p_test.squeeze()
    model = EnsembleModel(top_dec, top_prior, mid_dec, mid_prior, X_train, p_train, top_indices, mid_indices)
    return model, X_train, X_test, p_train, p_test, train_mask, test_mask

def softmax_rows(X):
    e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


if __name__ == '__main__':
    main()
