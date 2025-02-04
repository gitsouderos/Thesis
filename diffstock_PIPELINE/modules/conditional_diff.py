import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from att_dcnn import AttDCNN
from MRT import MaskedRelationalTransformer
from time_embedding import get_timestep_embedding

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation_rates=[1,2,4], num_heads=4):
        """
        Args:
            input_dim: Number of features per day in historical data.
            hidden_dim: Feature dimension used in the condition network.
        """
        super(ConditionalDiffusionModel, self).__init__()
        # Condition network: extracts features from historical data and applies relational modeling.
        self.att_dcnn = AttDCNN(input_dim, hidden_dim, kernel_size, dilation_rates)
        self.mrt = MaskedRelationalTransformer(hidden_dim, num_heads)
        # MLP that takes as input the concatenation of:
        #   - x_k (noisy target, 1 channel),
        #   - condition embedding (hidden_dim), and
        #   - time embedding (hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(1 + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # predict noise (same shape as x_k)
        )
    
    def forward(self, x_k, historical_data, relation_mask, t):
        """
        Args:
            x_k: Noisy target data, shape (batch_size, N, 1)
            historical_data: Historical data, shape (batch_size, N, L, input_dim)
            relation_mask: Binary relation mask, shape (N, N)
            t: Diffusion time step, tensor of shape (batch_size,)
        Returns:
            Predicted noise, shape (batch_size, N, 1)
        """
        # Extract condition features from historical data
        cond = self.att_dcnn(historical_data)   # (batch_size, N, hidden_dim)
        cond = self.mrt(cond, relation_mask)      # (batch_size, N, hidden_dim)
        
        # Compute time embedding
        time_embed = get_timestep_embedding(t, cond.size(-1))  # (batch_size, hidden_dim)
        time_embed = time_embed.unsqueeze(1).expand_as(cond)     # (batch_size, N, hidden_dim)
        
        # Concatenate x_k, condition embedding, and time embedding along the last dimension
        combined = torch.cat([x_k, cond, time_embed], dim=-1)    # (batch_size, N, 1+2*hidden_dim)
        
        # Predict the noise
        noise_pred = self.fc(combined)  # (batch_size, N, 1)
        return noise_pred