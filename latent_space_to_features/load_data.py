import time
from dataLoader.dataLoader import load_ratings_with_personality_traits
from kernelNet.kernel import *
from personalityClassifier.load_encoder_decoder import load_encoder, load_decoder
from personalityClassifier.utils import get_device


def load_data():
    device = get_device()
    seed = int(time.time())
    torch.seed()
    train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
        path='../../personality-isf2018/', valfrac=0.1, seed=seed, feature_classification=False, transpose=False)

    """
    nonzero_rows = validation_data.sum(axis=1) > 0
    validation_data = validation_data[nonzero_rows]
    valid_user_features = valid_user_features[nonzero_rows]
    """

    train_ratings = torch.from_numpy(train_data).to(device).squeeze()
    validation_ratings = torch.from_numpy(validation_data).to(device).squeeze()

    train_user_features = torch.tensor(train_user_features, dtype=torch.float32).squeeze(1)
    valid_user_features = torch.tensor(valid_user_features, dtype=torch.float32).squeeze(1)

    train_mask = torch.greater_equal(train_ratings, 0.5).float()
    validation_mask = torch.greater_equal(validation_ratings, 0.5).float()

    encoder = load_encoder("./output_autoencoder/")
    with torch.no_grad():
        train_latent_space, _ = encoder.forward(train_ratings)
        test_latent_space, _ = encoder.forward(validation_ratings)

    return train_user_features, valid_user_features, train_latent_space, test_latent_space, train_ratings, validation_ratings, train_mask, validation_mask
