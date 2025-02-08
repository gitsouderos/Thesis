import numpy as np
import torch

def forward_diffusion_sample(x_0, timestep, betas):
    """
    Returns a noisy sample from the forward diffusion model.

    Args: 
    - x_0: Initial value of the sample (tensor of shape [batch_size, dim])
    - timestep: Timestep at which to sample
    - betas: Array of beta values for each timestep

    Returns:
    - x_t: Noisy sample at the given timestep
    - noise: Noise added to the sample
    """

    # Compute alpha values from beta
    alphas = 1 - betas # Shape: [T]

    # Compute cumulative product of alphas
    alpha_bars = torch.cumprod(alphas,dim=0) # Shape: [T]
    
    # Select appropriate alpha bar for the timestep
    alpha_bar_t = alpha_bars[timestep].unsqueeze(1) # shape: [batch_size, 1]

    # Generate Gaussian noise
    noise = torch.randn_like(x_0)

    # Compute noisy sample
    x_t = x_0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt((1 - alpha_bar_t))

    return x_t, noise