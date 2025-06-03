import torch
from torch import nn
import numpy as np

from ensemble.personalityKnn import recommend_personality, recommend_personality_exclusive
from ensemble.utils import get_relevant_items
from personalityClassifier.utils import compute_itemwise_ndcg

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
                 k = 20, # number of reccomended items
                 probability_mapper = None,
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
        self.mapper = probability_mapper

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

    def __fuse_rec_lists(self, top: torch.Tensor, mid: torch.Tensor, sim: torch.Tensor, probs: torch.Tensor):
        out = torch.empty(self.k, dtype=top.dtype)
        filled = 0
        all_common = top[torch.isin(top, mid) & torch.isin(top, sim) & torch.isin(mid, sim)]
        for e in all_common:
            out[filled] = e
            filled += 1
            if filled >= self.k:
                return out

        tm = top[torch.isin(top, mid) & ~torch.isin(top, all_common)]
        for e in tm:
            out[filled] = e
            filled += 1
            if filled >= self.k:
                return out

        ts = top[torch.isin(top, sim) & ~torch.isin(top, all_common)]
        for e in ts:
            out[filled] = e
            filled += 1
            if filled >= self.k:
                return out

        ms = mid[torch.isin(mid, sim) & ~torch.isin(mid, all_common)]
        for e in ms:
            out[filled] = e
            filled += 1
            if filled >= self.k:
                return out

        top_mask = ~torch.isin(top, all_common) & ~torch.isin(top, tm) & ~torch.isin(top, ts)
        mid_mask = ~torch.isin(mid, all_common) & ~torch.isin(mid, tm) & ~torch.isin(mid, ms)
        sim_mask = ~torch.isin(sim, all_common) & ~torch.isin(sim, ts) & ~torch.isin(sim, ms)

        top_pool = top[top_mask]
        mid_pool = mid[mid_mask]
        sim_pool = sim[sim_mask]
        pools = [top_pool, mid_pool, sim_pool]

        for _ in range(self.k - filled):
            pool_idx = torch.multinomial(probs, 1).item()
            e = pools[pool_idx][0]
            pools[pool_idx] = pools[pool_idx][1:pools[pool_idx].shape[0]]
            out[filled] = e
            filled += 1

        return out

    def generate_training_data(self, X: torch.Tensor, true_ratings: torch.Tensor, exclusive = True):
        assert not torch.is_grad_enabled()
        top_picks, true_rates = get_relevant_items(true_ratings.squeeze())
        top_preds = self.small_dec(self.small_prior(X)[0])[0]
        top_preds = self.__map_subset_preds(top_preds, self.top_map)
        top_preds = self.__get_rec_list(top_preds).unsqueeze(1)
        top_preds = compute_itemwise_ndcg(top_picks, top_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        mid_preds = self.mid_dec(self.mid_prior(X)[0])[0]
        mid_preds = self.__map_subset_preds(mid_preds, self.mid_map)
        mid_preds = self.__get_rec_list(mid_preds).unsqueeze(1)
        mid_preds = compute_itemwise_ndcg(top_picks, mid_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        k_preds = recommend_personality_exclusive(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze()) if exclusive else recommend_personality(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze())
        k_preds = self.__get_rec_list(torch.from_numpy(k_preds)).unsqueeze(1)
        k_preds = compute_itemwise_ndcg(top_picks, k_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        return torch.vstack((torch.from_numpy(top_preds), torch.from_numpy(mid_preds), torch.from_numpy(k_preds))).permute((1, 0))

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        assert not torch.is_grad_enabled()
        top_preds = self.small_dec(self.small_prior(X)[0])[0]
        top_preds = self.__map_subset_preds(top_preds, self.top_map)
        top_preds = top_preds * mask if mask is not None else top_preds
        top_preds = self.__get_rec_list(top_preds)
        mid_preds = self.mid_dec(self.mid_prior(X)[0])[0]
        mid_preds = self.__map_subset_preds(mid_preds, self.mid_map)
        mid_preds = mid_preds * mask if mask is not None else mid_preds
        mid_preds = self.__get_rec_list(mid_preds)
        k_preds = recommend_personality(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze())
        k_preds = self.__get_rec_list(torch.from_numpy(k_preds))
        list
        probs = self.mapper(X)
        data = torch.stack((top_preds, mid_preds, k_preds), dim=1)
        out = torch.empty((X.shape[0], self.k))
        for i, e in enumerate(data):
            out[i] = self.__fuse_rec_lists(*e, probs[i])
        return out
