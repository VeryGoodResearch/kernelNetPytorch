import torch
from torch import nn
import numpy as np

from ensemble.personalityKnn import recommend_personality, recommend_personality_exclusive
from ensemble.utils import get_relevant_items
from personalityClassifier.utils import calculate_hitrate, compute_itemwise_ndcg, compute_ndcg

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

    def __get_rec_list(self, ratings, k=20):
        top_indices = torch.argsort(ratings, descending=True, dim=1)[:,:k]
        return top_indices

    def __fuse_rec_lists(self, top: torch.Tensor, mid: torch.Tensor, sim: torch.Tensor, top_rates: torch.Tensor, mid_rates: torch.Tensor, sim_rates: torch.Tensor, probs: torch.Tensor):
        out = torch.zeros(self.user_ratings.shape[2], dtype=torch.float32)
        top = top.long()
        mid = mid.long()
        sim = sim.long()
        for i, r in enumerate(top):
            out[r] += top_rates[i]*probs[0]
        for i, r in enumerate(mid):
            out[r] += mid_rates[i]*probs[1]
        for i, r in enumerate(sim):
            out[r] += sim_rates[i]*probs[2]
        recs = self.__get_rec_list(out.reshape((1, -1)))
        return recs

    def generate_training_data(self, X: torch.Tensor, true_ratings: torch.Tensor, exclusive = True, detailed=False):
        assert not torch.is_grad_enabled()
        top_picks, true_rates = get_relevant_items(true_ratings.squeeze())
        top_preds = self.small_dec(self.small_prior(X)[0])[0]
        top_preds = self.__map_subset_preds(top_preds, self.top_map)
        top_preds = self.__get_rec_list(top_preds, self.k).unsqueeze(1)
        top_scores = compute_itemwise_ndcg(top_picks, top_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        mid_preds = self.mid_dec(self.mid_prior(X)[0])[0]
        mid_preds = self.__map_subset_preds(mid_preds, self.mid_map)
        mid_preds = self.__get_rec_list(mid_preds, self.k).unsqueeze(1)
        mid_scores = compute_itemwise_ndcg(top_picks, mid_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        k_preds = recommend_personality_exclusive(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze()) if exclusive else recommend_personality(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze())
        k_preds = self.__get_rec_list(torch.from_numpy(k_preds), self.k).unsqueeze(1)
        k_scores = compute_itemwise_ndcg(top_picks, k_preds, true_rates, k=20, num_items=true_ratings.shape[2])
        if detailed:
            top_preds = top_preds.squeeze().long()
            mid_preds = mid_preds.squeeze().long()
            k_preds = k_preds.squeeze().long()
            print('Metrics for top 500 model')
            n20 = compute_ndcg(top_picks, top_preds, true_rates, k=20, num_items=true_ratings.shape[2])
            n10 = compute_ndcg(top_picks, top_preds, true_rates, k=10, num_items=true_ratings.shape[2])
            n5 = compute_ndcg(top_picks, top_preds, true_rates, k=5, num_items=true_ratings.shape[2])
            h20 = calculate_hitrate(top_picks, top_preds, 20)
            h10 = calculate_hitrate(top_picks, top_preds, 10)
            h5 = calculate_hitrate(top_picks, top_preds, 5)
            print(f'NDCG@20: {n20}')
            print(f'NDCG@10: {n10}')
            print(f'NDCG@5: {n5}')
            print(f'Hitrate@20: {h20}')
            print(f'Hitrate@10: {h10}')
            print(f'Hitrate@5: {h5}')
            print('Metrics for top 2000 model')
            n20 = compute_ndcg(top_picks, mid_preds, true_rates, k=20, num_items=true_ratings.shape[2])
            n10 = compute_ndcg(top_picks, mid_preds, true_rates, k=10, num_items=true_ratings.shape[2])
            n5 = compute_ndcg(top_picks, mid_preds, true_rates, k=5, num_items=true_ratings.shape[2])
            h20 = calculate_hitrate(top_picks, mid_preds, 20)
            h10 = calculate_hitrate(top_picks, mid_preds, 10)
            h5 = calculate_hitrate(top_picks, mid_preds, 5)
            print(f'NDCG@20: {n20}')
            print(f'NDCG@10: {n10}')
            print(f'NDCG@5: {n5}')
            print(f'Hitrate@20: {h20}')
            print(f'Hitrate@10: {h10}')
            print(f'Hitrate@5: {h5}')
            print('Metrics for personality knn model')
            n20 = compute_ndcg(top_picks, k_preds, true_rates, k=20, num_items=true_ratings.shape[2])
            n10 = compute_ndcg(top_picks, k_preds, true_rates, k=10, num_items=true_ratings.shape[2])
            n5 = compute_ndcg(top_picks, k_preds, true_rates, k=5, num_items=true_ratings.shape[2])
            h20 = calculate_hitrate(top_picks, k_preds, 20)
            h10 = calculate_hitrate(top_picks, k_preds, 10)
            h5 = calculate_hitrate(top_picks, k_preds, 5)
            print(f'NDCG@20: {n20}')
            print(f'NDCG@10: {n10}')
            print(f'NDCG@5: {n5}')
            print(f'Hitrate@20: {h20}')
            print(f'Hitrate@10: {h10}')
            print(f'Hitrate@5: {h5}')

        return torch.vstack((torch.from_numpy(top_scores), torch.from_numpy(mid_scores), torch.from_numpy(k_scores))).permute((1, 0))

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None):
        assert not torch.is_grad_enabled()
        top_preds = self.small_dec(self.small_prior(X)[0])[0]
        top_preds = self.__map_subset_preds(top_preds, self.top_map)
        top_preds = top_preds * mask if mask is not None else top_preds
        top_list = self.__get_rec_list(top_preds, self.k*2)
        top_rates = top_preds.gather(1, top_list)
        del top_preds
        mid_preds = self.mid_dec(self.mid_prior(X)[0])[0]
        mid_preds = self.__map_subset_preds(mid_preds, self.mid_map)
        mid_preds = mid_preds * mask if mask is not None else mid_preds
        mid_list = self.__get_rec_list(mid_preds, self.k*2)
        mid_rates = mid_preds.gather(1, mid_list)
        del mid_preds
        k_preds = recommend_personality(X.cpu().numpy(), self.user_ratings.squeeze(), self.user_personalities.squeeze())
        k_preds = torch.from_numpy(k_preds)
        k_list = self.__get_rec_list(k_preds, self.k*2)
        k_rates = k_preds.gather(1, k_list)
        probs = self.mapper(X)
        data = torch.stack((top_list, mid_list, k_list, top_rates, mid_rates, k_rates), dim=1)
        out = torch.empty((X.shape[0], self.k))
        for i, e in enumerate(data):
            out[i] = self.__fuse_rec_lists(*e, probs[i])
        return out
