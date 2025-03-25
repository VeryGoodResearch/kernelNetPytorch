import torch
import torch.nn as nn


class CombinedResidualModel(nn.Module):
    def __init__(self, rating_model: nn.Module, personality_feature_dim: int, user_features_weight: float, residual_hidden_dim: int):
        super(CombinedResidualModel, self).__init__()
        self.personality_feature_dim = personality_feature_dim
        self.residual_hidden_dim = residual_hidden_dim
        self.rating_model = rating_model
        self.residual_module = ResidualCorrectionModule(personality_feature_dim, residual_hidden_dim)
        self.user_features_weight = user_features_weight

    def forward(self, rating_data: torch.Tensor, personality_features: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor]:
        rating_pred, reg = self.rating_model(rating_data)
        correction = self.residual_module(personality_features)
        final_pred = rating_pred + self.user_features_weight * correction
        return final_pred, reg

class ResidualCorrectionModule(nn.Module):
    def __init__(self, personality_feature_dim, hidden_dim):
        super(ResidualCorrectionModule, self).__init__()
        self.fc1 = nn.Linear(personality_feature_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, personality_features: torch.Tensor) -> torch.Tensor:
        x = self.fc1(personality_features)
        x = self.relu(x)
        correction = self.fc2(x)
        return correction
