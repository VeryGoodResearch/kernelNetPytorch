from dataLoader.dataLoader import load_ratings_matrix_from_personality_traits_dataset
import numpy as np
import pandas as pd
from personalityClassifier.load_encoder_decoder import load_decoder, load_encoder
import matplotlib.pyplot as plt
from personalityClassifier.utils import get_device
import torch


def normalize_ratings(mat):
    norm_mat = np.clip(mat, 0.5, 5)
    norm_mat = (norm_mat - 0.5) / (5 - 0.5)  # skala 0-1
    norm_mat[mat == 0] = 0  # 0 = brak oceny
    return norm_mat

def show_matrix(matrix, title):
    subset_rows = slice(0, 100)  # lub np. slice(None) dla całości
    subset_cols = slice(0, 500)

    plt.figure(figsize=(10, 4))
    plt.imshow(matrix[subset_rows, subset_cols], cmap='gray', aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Movies')
    plt.ylabel('Users')
    plt.show()


original_ratings = load_ratings_matrix_from_personality_traits_dataset(path='../../personality-isf2018/')
device = get_device()

original_ratings = torch.from_numpy(original_ratings).to(device).squeeze()

decoder = load_decoder("./output_autoencoder/")
decoder.eval()
for p in decoder.parameters():
    p.requires_grad = False

encoder = load_encoder("../output_autoencoder/")
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

latent_space, _ = encoder(original_ratings)
reconstructed_ratings, _ = decoder(latent_space)

orig_norm = normalize_ratings(original_ratings)
recon_norm = normalize_ratings(reconstructed_ratings)
diff = np.abs(orig_norm - recon_norm)


show_matrix(orig_norm, "Original matrix")
show_matrix(recon_norm, "Reconstructed matrix")
show_matrix(diff, "Difference (|oryg - reconstructed|)")