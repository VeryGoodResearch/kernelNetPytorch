import torch
from torch import nn
import numpy as np

from ensemble.personalityKnn import recommend_personality

class EnsembleModel(nn.Module):
    def __init__(self, 
                 small_dec, # trained instance of the top 500 movies decoder
                 small_prior, #trained instance of the top 500 movies prior model
                 mid_dec, # trained instance of the top 20k movies decoder
                 mid_prior, # trained instance of the top 20k movies prior model
                 user_ratings, # user ratings for user personality knn
                 user_personalities, # user personalities for user personality knn
                 top_indices, # movie index map for top 500 movies recommendation
                 mid_indices, # movie index map for top 20k movies recommendation
                 k = 20 # number of reccomended items
                 ) -> None:
        super().__init__()
        self.small_dec = small_dec
        self.small_prior = small_prior
        self.mid_dec = mid_dec
        self.mid_prior = mid_prior
        self.user_ratings = user_ratings
        self.user_personalities = user_personalities
        self.top_map = top_indices
        self.mid_map = mid_indices
        self.k = k

    def __map_subset_preds(self, output, map):
        out = torch.empty((output.shape[0], self.user_ratings.shape[2]))
        for i in range(out.shape[0]):
            tmp = torch.zeros((self.user_ratings.shape[2]))
            for idx, rating in enumerate(output[i]):
                tmp[map[idx]] = rating
            out[i] = tmp
        return out

    def __get_rec_list(self, ratings):
        top_indices = torch.argsort(ratings, descending=True, dim=1)[:,:self.k]
        return top_indices

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        top_preds = self.small_dec(self.small_prior(X)[0])[0]
        top_preds = self.__map_subset_preds(top_preds, self.top_map)
        top_preds = top_preds * mask if mask is not None else top_preds
        top_preds = self.__get_rec_list(top_preds).unsqueeze(1)
        mid_preds = self.mid_dec(self.mid_prior(X)[0])[0]
        mid_preds = self.__map_subset_preds(mid_preds, self.mid_map)
        mid_preds = mid_preds*mask if mask is not None else mid_preds
        mid_preds = self.__get_rec_list(mid_preds).unsqueeze(1)
        k_preds = recommend_personality(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze())
        k_preds = self.__get_rec_list(torch.from_numpy(k_preds)).unsqueeze(1)
        return torch.concat((top_preds, mid_preds, k_preds), dim=1)
