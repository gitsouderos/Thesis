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

def get_time_embedding(t,embedding_dim):
    """
    Returns a time embedding for the given timestep.

    Args:
    - t: tesor of shape [batch_size] containing the timesteps
    - embedding_dim: desired length of the time embedding

    Returns:
    - time_embedding: Time embedding tensor of shape [batch_size, embedding_dim]
    """

    # Reshape t to [batch_size, 1]
    t = t.unsqueeze(1)

    # Create a vector for indices : j=0,1,2,...,embedding_dim/2 -1
    j = torch.arange(0,embedding_dim//2).unsqueeze(0)

    # Compute the scaling factor: scale = 10000^(2j/embedding_dim)
    scale = 10000**(2*j/embedding_dim)

    # Devide t by the scaling factor for the sin/cos functions
    args = t / scale # Shape: [batch_size, embedding_dim//2]

    # Compute sin and cos on the arguments
    sin = torch.sin(args) # Shape: [batch_size, embedding_dim//2]
    cos = torch.cos(args) # Shape: [batch_size, embedding_dim//2]

    # Concatenate sin and cos to get the time embedding along the last dimension
    time_embedding = torch.cat([sin,cos],dim=-1) # Shape: [batch_size, embedding_dim]

    return time_embedding

    

def reverse_diffusion_sample(x_T, timestep):
    
    """
    
    """

