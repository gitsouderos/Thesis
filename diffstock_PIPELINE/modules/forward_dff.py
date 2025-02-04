##############################################
# 5. Forward Diffusion Process
##############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def forward_diffusion_sample(x0, t, betas):
    """
    Given a clean target x0 and a diffusion step t, compute the noisy version x_t.
    Uses the closed-form for q(x_t | x0):
    
        x_t = sqrt(alpha_bar_t)*x0 + sqrt(1 - alpha_bar_t)*noise
    
    Args:
        x0: Clean data tensor of shape (batch_size, N, 1)
        t: Tensor of shape (batch_size,) containing diffusion steps (integers).
        betas: Tensor of shape (T,) containing the beta schedule.
    Returns:
        x_t: Noisy version of x0 at step t.
        noise: The noise that was added.
    """
    # Compute alpha and cumulative product alpha_bar for each timestep.
    alphas = 1 - betas  # shape: (T,)
    # Compute cumulative product alpha_bar_t for each t.
    alpha_bars = torch.cumprod(alphas, dim=0)  # shape: (T,)
    
    # Gather alpha_bar_t for each sample in the batch.
    # Ensure t is of type long for indexing.
    t = t.long()
    alpha_bar_t = alpha_bars[t].unsqueeze(-1).unsqueeze(-1)  # shape: (batch_size, 1, 1)
    
    noise = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise