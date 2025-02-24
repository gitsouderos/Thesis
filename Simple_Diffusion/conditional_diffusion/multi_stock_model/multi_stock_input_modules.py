import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



##---------------------------- PRE DIFFUSION STEPS -----------------------------------##


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
    j = torch.arange(0,embedding_dim//2,device=t.device).unsqueeze(0)

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


# Create embedding based on the ticker (stock) of the batch sample
class TickerEmbedding(nn.Module):
    def __init__(self,ticker_list, ticker_embedding_dim):
        super(TickerEmbedding,self).__init__()
        
        '''
        Create a dictionary based on the index of the individual tickers. Then, we create an instance of nn.Embedding
        which creates a matrix of size [num_tickers, embedding_dim] where each row is a learnable vector (a ticker)

        Args :
          - ticker_list : list of all unique tickers
          - ticker_embedding_dim: the dimension we want the embeddings to be, in this case 32

        Returns:
          -ticker embeddings 
        '''
        self.embedding_dict = {ticker:index for index,ticker in enumerate(ticker_list)}
        self.embedding = nn.Embedding(num_embeddings=len(ticker_list), embedding_dim = ticker_embedding_dim)
    
    def forward(self,tickers):
        ticker_dict = torch.tensor([self.embedding_dict[t] for t in tickers], dtype=torch.long, device=self.embedding.weight.device) # Shape: [batch_size]
        ticker_emb = self.embedding(ticker_dict) # Shape : [batch_size,embedding_dim]

        return ticker_emb
        

    

    


# Define the Att-DCNN module for context encoding.
class Context_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation_rates=[1, 2, 4], num_heads=4):
        """
        Args:
          - input_dim: Number of features in the context (e.g., 9).
          - hidden_dim: The output dimension for the convolution layers and the attention block.
          - kernel_size: Size of the convolution kernels.
          - dilation_rates: List of dilation rates.
          - num_heads: Number of attention heads.
        """
        super(Context_Encoder, self).__init__()
        # Create a set of dilated convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                      kernel_size=kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2)
            for d in dilation_rates
        ])
        # Multi-head self-attention over the sequence
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        # Fully connected layer to refine the aggregated context embedding
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
          - x: Tensor of shape [batch_size, sequence_length, input_dim]
        Returns:
          - context_embedding: Tensor of shape [batch_size, hidden_dim]
        """
        # Permute for convolution: [batch_size, input_dim, sequence_length]
        x_conv = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x_conv))  # shape: [batch_size, hidden_dim, sequence_length]
            conv_outs.append(conv_out)
        # Average the outputs from different dilation rates
        x_conv = torch.stack(conv_outs, dim=0).mean(dim=0)  # shape: [batch_size, hidden_dim, sequence_length]
        # Permute back for attention: [batch_size, sequence_length, hidden_dim]
        x_conv = x_conv.permute(0, 2, 1)
        # Apply multi-head self-attention; query, key, and value are all x_conv
        attn_output, _ = self.attention(x_conv, x_conv, x_conv)  # shape: [batch_size, sequence_length, hidden_dim]
        # Aggregate over the sequence dimension (mean pooling)
        context_embedding = attn_output.mean(dim=1)  # shape: [batch_size, hidden_dim]
        # Refine with a fully connected layer and non-linearity
        context_embedding = self.relu(self.fc(context_embedding))
        return context_embedding
    



## ------------------------------------------DIFFUSION STEPS-----------------------------------------#



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
    # x_0 = x_0.unsqueeze(1) # Shape: [batch_size, 1]

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
    
class ResidualMLPWithExtraBlock(nn.Module):
    def __init__(self, dim, embedding_dim, context_embedding_size, ticker_embedding_size, hidden_size=512, num_chunks=8, attn_heads=4, dropout_prob=0.1):
        """
        A Residual MLP with attention that includes an extra residual attention block,
        layer normalization, and dropout. Modified to handle multi-stock input.
        
        Args:
          - dim: Dimension of the noisy sample (e.g., 1 for a scalar target).
          - embedding_dim: Dimension of the time embedding.
          - context_embedding_size: Dimension of the context embedding.
          - ticker_embedding_size: Dimension of the ticker embedding.
          - hidden_size: Hidden layer size for the MLP.
          - num_chunks: Number of chunks to split the hidden vector into for attention.
          - attn_heads: Number of attention heads.
          - dropout_prob: Dropout probability.
          
        The total input dimension is: dim + embedding_dim + context_embedding_size.
        """
        super(ResidualMLPWithExtraBlock, self).__init__()
        total_input_dim = dim + embedding_dim + context_embedding_size + ticker_embedding_size
        self.fc1 = nn.Linear(total_input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        
        # Parameters for splitting the hidden vector into chunks
        self.num_chunks = num_chunks
        assert hidden_size % num_chunks == 0, "hidden_size must be divisible by num_chunks"
        self.chunk_size = hidden_size // num_chunks
        
        # First multi-head attention block over chunks
        self.attention1 = nn.MultiheadAttention(embed_dim=self.chunk_size, num_heads=attn_heads, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        
        # Extra residual attention block
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        
        # Final output projection
        self.fc4 = nn.Linear(hidden_size, dim)
    
    def forward(self, x_combined):
        """
        Args:
          - x_combined: Input tensor of shape [batch_size, num_stocks, total_input_dim] 
                        where total_input_dim = dim + embedding_dim + context_embedding_size.
        Returns:
          - predicted_noise: Tensor of shape [batch_size, num_stocks, dim]
        """
        # batch_size, num_stocks, _ = x_combined.shape
        # # Flatten batch and stocks: [batch_size*num_stocks, total_input_dim]
        # x_flat = x_combined.reshape(batch_size * num_stocks, -1)

        
        hidden1 = self.relu(self.fc1(x_combined))  # shape: [batch_size*num_stocks, hidden_size]
        hidden1 = self.dropout(hidden1)
        hidden1 = self.layernorm1(hidden1)

        batch_size = x_combined.shape[0]
        
        # Split into chunks: reshape to [batch_size*num_stocks, num_chunks, chunk_size]
        hidden_chunks = hidden1.view(batch_size, self.num_chunks, self.chunk_size)
        # First attention block over chunks
        attn_output1, _ = self.attention1(hidden_chunks, hidden_chunks, hidden_chunks)
        attn_flat1 = attn_output1.reshape(batch_size, -1)  # [batch_size*num_stocks, hidden_size]
        
        combined1 = attn_flat1 + hidden1  # residual connection
        combined1 = self.relu(self.fc2(combined1))
        combined1 = self.dropout(combined1)
        combined1 = self.layernorm2(combined1)
        
        # Extra residual attention block
        # We treat the vector as a sequence of length 1
        seq = combined1.unsqueeze(1)  # [batch_size*num_stocks, 1, hidden_size]
        attn_output2, _ = self.attention2(seq, seq, seq)  # [batch_size*num_stocks, 1, hidden_size]
        attn_flat2 = attn_output2.squeeze(1)  # [batch_size*num_stocks, hidden_size]
        
        combined2 = attn_flat2 + combined1  # residual connection
        combined2 = self.relu(self.fc3(combined2))
        combined2 = self.dropout(combined2)
        combined2 = self.layernorm3(combined2)
        
        predicted_noise = self.fc4(combined2)  # [batch_size, dim]

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

