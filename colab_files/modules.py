import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##---------------------------- PRE DIFFUSION STEPS -----------------------------------##

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Generates a beta schedule based on a cosine function.
    """
    x = np.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = np.cos(((x / timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize so that alpha_bar_0 = 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    return torch.tensor(betas, dtype=torch.float32)

def get_time_embedding(t, embedding_dim):
    """
    Returns a time embedding for the given timestep.
    """
    t = t.unsqueeze(1)  # [batch_size, 1]
    # Create index tensor on same device as t
    j = torch.arange(0, embedding_dim // 2, device=t.device).unsqueeze(0)
    scale = 10000 ** (2 * j / embedding_dim)
    args = t / scale  # [batch_size, embedding_dim//2]
    sin = torch.sin(args)
    cos = torch.cos(args)
    time_embedding = torch.cat([sin, cos], dim=-1)  # [batch_size, embedding_dim]
    return time_embedding

# Create embedding based on the ticker (stock) of the batch sample
class TickerEmbedding(nn.Module):
    def __init__(self, ticker_list, ticker_embedding_dim):
        super(TickerEmbedding, self).__init__()
        self.embedding_dict = {ticker: index for index, ticker in enumerate(ticker_list)}
        self.embedding = nn.Embedding(num_embeddings=len(ticker_list), embedding_dim=ticker_embedding_dim)
    
    def forward(self, tickers):
        ticker_dict = torch.tensor([self.embedding_dict[t] for t in tickers], dtype=torch.long, device=self.embedding.weight.device)
        ticker_emb = self.embedding(ticker_dict)
        return ticker_emb

# Define the Att-DCNN module for context encoding.
class Context_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation_rates=[1, 2, 4], num_heads=4):
        super(Context_Encoder, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                      kernel_size=kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2)
            for d in dilation_rates
        ])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [batch_size, sequence_length, input_dim] -> [batch_size, input_dim, sequence_length]
        x_conv = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x_conv))
            conv_outs.append(conv_out)
        x_conv = torch.stack(conv_outs, dim=0).mean(dim=0)
        x_conv = x_conv.permute(0, 2, 1)  # Back to [batch_size, sequence_length, hidden_dim]
        attn_output, _ = self.attention(x_conv, x_conv, x_conv)
        context_embedding = attn_output.mean(dim=1)
        context_embedding = self.relu(self.fc(context_embedding))
        return context_embedding

## ------------------------------------------DIFFUSION STEPS-----------------------------------------##

def forward_diffusion_sample(x_0, timestep, betas):
    """
    Returns a noisy sample from the forward diffusion model.
    """
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alpha_bars[timestep].unsqueeze(1)  # [batch_size, 1]
    noise = torch.randn_like(x_0)
    x_t = x_0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt(1 - alpha_bar_t)
    return x_t, noise

class ResidualMLPWithExtraBlock(nn.Module):
    def __init__(self, dim, embedding_dim, context_embedding_size, ticker_embedding_dim, hidden_size=512, num_chunks=8, attn_heads=4, dropout_prob=0.1):
        """
        A Residual MLP with extra attention blocks.
        """
        super(ResidualMLPWithExtraBlock, self).__init__()
        total_input_dim = dim + embedding_dim + context_embedding_size + ticker_embedding_dim
        self.fc1 = nn.Linear(total_input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        
        self.num_chunks = num_chunks
        assert hidden_size % num_chunks == 0, "hidden_size must be divisible by num_chunks"
        self.chunk_size = hidden_size // num_chunks
        
        self.attention1 = nn.MultiheadAttention(embed_dim=self.chunk_size, num_heads=attn_heads, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        
        self.attention2 = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=attn_heads, batch_first=True)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        
        self.fc4 = nn.Linear(hidden_size, dim)
    
    def forward(self, x_combined):
        hidden1 = self.relu(self.fc1(x_combined))
        hidden1 = self.dropout(hidden1)
        hidden1 = self.layernorm1(hidden1)

        batch_size = x_combined.shape[0]
        hidden_chunks = hidden1.view(batch_size, self.num_chunks, self.chunk_size)
        attn_output1, _ = self.attention1(hidden_chunks, hidden_chunks, hidden_chunks)
        attn_flat1 = attn_output1.reshape(batch_size, -1)
        
        combined1 = attn_flat1 + hidden1
        combined1 = self.relu(self.fc2(combined1))
        combined1 = self.dropout(combined1)
        combined1 = self.layernorm2(combined1)
        
        seq = combined1.unsqueeze(1)
        attn_output2, _ = self.attention2(seq, seq, seq)
        attn_flat2 = attn_output2.squeeze(1)
        
        combined2 = attn_flat2 + combined1
        combined2 = self.relu(self.fc3(combined2))
        combined2 = self.dropout(combined2)
        combined2 = self.layernorm3(combined2)
        
        predicted_noise = self.fc4(combined2)
        return predicted_noise

def reverse_diffusion_sample(x_T, betas, timestep, embedding_dim, context, context_net, denoise_net):
    """
    Performs one reverse diffusion step.
    """
    _, dim = x_T.shape
    time_embedding = get_time_embedding(timestep, embedding_dim)
    x = torch.cat([x_T, time_embedding], dim=-1)
    context_embedding = context_net(context)
    x_combined = torch.cat([x, context_embedding], dim=-1)
    predicted_noise = denoise_net(x_combined)
    beta_t = betas[timestep].unsqueeze(1)
    alpha_t = 1 - beta_t
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alpha_bars[timestep].unsqueeze(1)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x_t_minus_1 = (x_T - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise) / sqrt_alpha_t
    return x_t_minus_1

def run_reverse_diffusion(denoise_net, context_net, context, betas, num_steps, batch_size, dim, embedding_dim):
    """
    Runs the full reverse diffusion chain to generate samples.
    """
    # Create x_T on the same device as betas
    x_t = torch.randn(batch_size, dim, device=betas.device)
    for t_val in reversed(range(num_steps)):
        print(f"timestep: {t_val}")
        t_tensor = torch.full((batch_size,), t_val, dtype=torch.long, device=betas.device)
        x_t = reverse_diffusion_sample(x_t, betas, t_tensor, embedding_dim, context, context_net, denoise_net)
    return x_t
