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

    # Unsqueeze x0
    x_0 = x_0.unsqueeze(1) # Shape: [batch_size, 1]

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


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a beta schedule based on a cosine function.
    
    Args:
      - timesteps: Total number of diffusion steps (T).
      - s: A small constant to adjust the schedule (commonly 0.008).
    
    Returns:
      - betas: A torch tensor of shape [T] with the beta values.
    """
    # Create an array from 0 to T (inclusive)
    x = np.linspace(0, timesteps, timesteps + 1)
    
    # Compute cumulative product (alpha_bar) using the cosine function
    alphas_cumprod = np.cos(((x / timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize so that alpha_bar_0 = 1
    
    # Compute betas from the cumulative product of alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Clip betas to prevent extreme values (0, 1)
    betas = np.clip(betas, 0, 0.999)
    
    return torch.tensor(betas, dtype=torch.float32)


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


class Context_Encoder(nn.Module):
    """
    Returns the context embedding given the context input of x0
    Args:
    - context : Context input of shape [batch_size,context_len, num_features]

    Returns:
    - context_embedding: Context embedding tensor of shape [batch_size, context_embedding_size]
    """

    def __init__(self,num_features, context_len, context_embedding_size):
        super(Context_Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size = num_features, hidden_size = context_embedding_size, batch_first = True)

    def forward(self, context):
        # context has shape [batch_size, context_len, num_features]
        # pass context through the LSTM
        _, (h_n,_) = self.lstm(context)
        # print(f"h_n shape: {h_n.shape}")
        # h_n has shape [1, batch_size, context_embedding_size]
        context_embedding = h_n.squeeze(0) # shape: [batch_size, context_embedding_size]
        return context_embedding

class model_architecture(nn.Module):
    """
    Returns the output of the model architecture given the input tensor x_combined.

    Args:
    - x_combined: Input tensor of shape [batch_size, dim + embedding_dim]

    Returns:
    - output: Output tensor of the model architecture
    """

    def __init__(self, dim, embedding_dim, context_embedding_size, hidden_size =256):
        super(model_architecture, self).__init__()
        #Total input dimension is dim + embedding_dim + context_embedding_size
        total_input_dim = dim + embedding_dim + context_embedding_size
        self.fc1 = nn.Linear(total_input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim)
        self.relu = nn.ReLU()

    def forward(self, x_combined):
        # x_combined has shape [batch_size, dim + embedding_dim+ context_embedding_size]
        hidden1 = self.relu(self.fc1(x_combined))
        hidden2 = self.relu(self.fc2(hidden1))
        predicted_noise = self.fc3(hidden2)
        return predicted_noise

class ResidualMLP(nn.Module):
    def __init__(self, dim, embedding_dim,context_embedding_size, hidden_size=256):
        super(ResidualMLP, self).__init__()
        #Total input dimension is dim + embedding_dim + context_embedding_size
        total_input_dim = dim + embedding_dim + context_embedding_size
        self.fc1 = nn.Linear(total_input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, dim)
        self.relu = nn.ReLU()
    
    def forward(self, x_combined):
        # Pass through first layer
        hidden1 = self.relu(self.fc1(x_combined))  # shape: [batch_size, hidden_size]
        
        # Pass through second layer
        hidden2 = self.relu(self.fc2(hidden1))      # shape: [batch_size, hidden_size]
        
        # Add a residual connection: add hidden1 to hidden2
        combined = hidden2 + hidden1                # shape: [batch_size, hidden_size]
        
        # Final layer: map to output dimension (dim)
        predicted_noise = self.fc3(combined)        # shape: [batch_size, dim]
        return predicted_noise


def reverse_diffusion_sample(x_T, betas, timestep, embedding_dim, context, context_net, denoise_net):
    
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

    # Get context embedding
    context_embedding = context_net(context) # Shape: [batch_size, context_embedding_size]

    # Concatentate x and context_embedding
    x_combined = torch.cat([x, context_embedding], dim=-1) # Shape: [batch_size, dim + embedding_dim + context_embedding_size]

    # pass x through the model architecture
    predicted_noise = denoise_net(x_combined) # Shape: [batch_size, dim]
    # print(f"noisy sample : {x_T}")
    # print(f"predicted noise : {predicted_noise}")
    
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
    # print(f"square root of alpha_t : {sqrt_alpha_t}")
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)  # Shape: [batch_size, 1]
    # print(f"square root of 1 - alpha_bar_t : {sqrt_one_minus_alpha_bar_t}")

    # Apply the reverse diffusion update:
    # x_{t-1} = ( x_T - (beta_t / sqrt(1 - alpha_bar_t)) * predicted_noise ) / sqrt(alpha_t)
    x_t_minus_1 = (x_T - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise) / sqrt_alpha_t
    # print(f"less noisy sample : {x_t_minus_1}")

    return x_t_minus_1

def run_reverse_diffusion(denoise_net, context_net, context, betas, num_steps, batch_size, dim, embedding_dim):
    """
    Runs the full reverse diffusion chain to generate samples.
    
    Args:
      - denoise_net: The trained denoising network.
      - betas: 1D tensor of beta values (shape: [num_steps]).
      - num_steps: Total number of diffusion steps.
      - batch_size: Number of samples to generate.
      - dim: Dimensionality of each sample (for your case, 1).
      
    Returns:
      - x_0_pred: The generated sample(s) after running reverse diffusion.
    """
    # Initialize x_T as pure Gaussian noise
    x_t = torch.randn(batch_size, dim)
    # Loop from t = num_steps - 1 down to 0
    for t_val in reversed(range(num_steps)):
        print(f"timestep: {t_val}")
        # Create a timestep tensor for the current step, shape: [batch_size]
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long)
        # Update x_t by performing one reverse diffusion step
        x_t = reverse_diffusion_sample(x_t, betas, t_tensor,embedding_dim, context, context_net, denoise_net,context_net,context)
    return x_t

