import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from Expander import Expander
from load_data import load_data
from personalityClassifier.load_encoder_decoder import load_decoder
from personalityClassifier.utils import get_device


def masked_kl_loss(log_preds, targets, mask):
    """
    Oblicza KL divergence tylko dla pozycji, gdzie maska == 1
    """
    mask = mask.bool()
    log_preds = log_preds[mask]
    targets = targets[mask]
    return F.kl_div(log_preds, targets, reduction='batchmean')

def masked_rmse_loss(preds, targets, mask):
    """
    Oblicza RMSE tylko dla pozycji, gdzie maska == 1
    """
    mask = mask.bool()
    preds = preds[mask]
    targets = targets[mask]
    mse = F.mse_loss(preds, targets, reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse


def main():
    X_train, X_test, train_latent_space, test_latent_space, train_ratings, test_ratings, train_ratings_mask, test_rating_mask = load_data()
    device = get_device()

    print("train user characteristics: ", X_train.shape)
    print("train latent space: ", train_latent_space.shape)
    print("train ratings: ", train_ratings.shape)


    # normalization (0-7 -> 0-1)
    max_feature_value = 7
    X_train = X_train/max_feature_value
    X_test = X_test/max_feature_value
    print("test user characteristics: ", X_test)

    decoder = load_decoder("./output_autoencoder/")
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    print("original latent space")
    print(train_latent_space)
    print("decoded original latent space")
    decoded, _ = decoder(train_latent_space)
    print(decoded)

    input_dim = X_train.shape[1]
    target_dim = decoder.kernel_hidden
    expander = Expander(input_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(expander.parameters(), lr=1e-3)

    num_epochs = 100
    dataset = TensorDataset(X_train, train_ratings, train_ratings_mask)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        expander.train()
        total_loss = 0
        for x_small, target_out, mask in dataloader:
            z_expanded = expander(x_small)
            decoded, _ = decoder(z_expanded)
            log_preds = F.log_softmax(decoded, dim=1)
            targets = F.softmax(target_out, dim=1)

            loss = masked_rmse_loss(decoded, targets, mask)
            kl = masked_kl_loss(log_preds, targets, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, KL: {kl.item():.4f}")

        expander.eval()
        with torch.no_grad():
            z_val = expander(X_test)
            decoded_val, _ = decoder(z_val)
            log_preds_val = F.log_softmax(decoded_val, dim=1)
            targets_val = F.softmax(test_ratings, dim=1)

            val_loss = masked_rmse_loss(decoded_val, targets_val, test_rating_mask)
            kl = masked_kl_loss(log_preds_val, targets_val, test_rating_mask)
            print(f"Validation RMSE: {val_loss.item():.4f}")
            print(f"Validation KL: {kl.item():.4f}")

            print("real: ", test_ratings[0])
            print("predicted: ", decoded_val[0])
            print("real: ", test_ratings[1])
            print("predicted: ", decoded_val[1])




main()
