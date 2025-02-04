import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math 

##############################################
# 1. Timestep Embedding Function
##############################################

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: Tensor of shape (batch_size,) containing diffusion steps.
        embedding_dim: Dimension of the embedding vector.
    Returns:
        Tensor of shape (batch_size, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad if needed
        emb = F.pad(emb, (0, 1))
    return emb