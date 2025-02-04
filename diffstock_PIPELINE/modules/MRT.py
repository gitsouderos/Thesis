##############################################
# 3. Masked Relational Transformer (MRT)
##############################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MaskedRelationalTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        """
        Args:
            hidden_dim: Dimension of the input feature vector.
            num_heads: Number of attention heads.
        """
        super(MaskedRelationalTransformer, self).__init__()
        # Using batch_first=True so that input is (batch_size, N, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, relation_mask):
        """
        Args:
            x: Tensor of shape (batch_size, N, hidden_dim) from Att-DCNN.
            relation_mask: Tensor of shape (N, N), binary mask where 1 indicates allowed attention.
        Returns:
            Tensor of shape (batch_size, N, hidden_dim) after relational processing.
        """
        # Create additive mask: positions with 0 in relation_mask get a large negative value.
        # MultiheadAttention expects attn_mask of shape (N, N)
        mask = (1 - relation_mask.float()) * -1e9  # (N, N)
        
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        return x