import numpy as np
import torch
import torch.nn as nn

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


class model_architecture(nn.Module):
    """
    Returns the output of the model architecture given the input tensor x_combined.

    Args:
    - x_combined: Input tensor of shape [batch_size, dim + embedding_dim]

    Returns:
    - output: Output tensor of the model architecture
    """

    def __init__(self, dim, embedding_dim, hidden_size =256):
        super(model_architecture, self).__init__()
        self.fc1 = nn.Linear(dim + embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim)
        self.relu = nn.ReLU()

    def forward(self, x_combined):
        # x_combined has shape [batch_size, dim + embedding_dim]
        hidden1 = self.relu(self.fc1(x_combined))
        hidden2 = self.relu(self.fc2(hidden1))
        predicted_noise = self.fc3(hidden2)
        return predicted_noise


def reverse_diffusion_sample(x_T, betas, timestep, embedding_dim, denoise_net):
    
    """
    Performs one reverse diffusion step.
    
    Args:
      - x_T: Noisy sample at current timestep (tensor of shape [batch_size, dim])
      - timestep: A tensor of shape [batch_size] containing the current diffusion timesteps (as integers)
      - betas: 1D tensor of shape [T] containing the beta schedule for each timestep.
      - denoise_net: An instance of your denoising network (e.g., model_architecture)
    
    Returns:
      - x_t_minus_1: Updated (less noisy) sample, tensor of shape [batch_size, dim]
    """

    # Get dimensions of the input tensor
    # print(f"x_T shape: {x_T.shape}")
    _, dim = x_T.shape

    # Get time embeddings
    # print(f"timestep shape: {timestep.shape}")
    time_embedding = get_time_embedding(timestep, embedding_dim)
    # print(f"time_embedding shape: {time_embedding.shape}")

    # get embedding dimension
    _,embedding_dim = time_embedding.shape

    # Concatenate x_T and time_embedding
    x = torch.cat([x_T, time_embedding], dim=-1) # Shape: [batch_size, dim + embedding_dim]
    # print(f"x shape: {x.shape}")

    # pass x through the model architecture
    predicted_noise = denoise_net(x) # Shape: [batch_size, dim]
    
    # Retrieve beta_t for each sample from the beta schedule.
    beta_t = betas[timestep].unsqueeze(1)  # Shape: [batch_size, 1]

    # Compute alpha_t = 1 - beta_t
    alpha_t = 1 - beta_t  # Shape: [batch_size, 1]

    # Compute the cumulative product of alphas over the entire schedule
    alphas = 1 - betas  # Shape: [T]
    alpha_bars = torch.cumprod(alphas, dim=0)  # Shape: [T]

    # Retrieve alpha_bar_t for the current timestep and unsqueeze for broadcasting.
    alpha_bar_t = alpha_bars[timestep].unsqueeze(1)  # Shape: [batch_size, 1]

    # Compute the necessary square roots
    sqrt_alpha_t = torch.sqrt(alpha_t)                   # Shape: [batch_size, 1]
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)  # Shape: [batch_size, 1]

    # Apply the reverse diffusion update:
    # x_{t-1} = ( x_T - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise ) / sqrt(alpha_t)
    x_t_minus_1 = (x_T - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise) / sqrt_alpha_t

    return x_t_minus_1
