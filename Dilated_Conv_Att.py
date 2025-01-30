import torch
import torch.nn as nn
import torch.nn.functional as F

class AttDiCEm(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dilation_rates=[1, 2, 4]):
        super(AttDiCEm, self).__init__()

        # Fix: Add correct padding to maintain sequence length
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                      kernel_size=kernel_size, padding=(kernel_size - 1) * d // 2, 
                      dilation=d) 
            for d in dilation_rates
        ])

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=2, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Input Shape: (batch_size, seq_len, input_size)
        Output Shape: (batch_size, 1) - Predicting next value
        """

        # Swap dimensions for Conv1D (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # Apply dilated convolutions
        conv_outputs = [F.relu(conv(x)) for conv in self.conv_layers]

        # Fix: Ensure that all outputs have the same sequence length
        min_seq_len = min([out.shape[2] for out in conv_outputs])
        conv_outputs = [out[:, :, :min_seq_len] for out in conv_outputs]

        # Stack and average across convolution outputs
        x = torch.stack(conv_outputs, dim=-1).mean(dim=-1)  # (batch_size, hidden_size, seq_len)

        # Swap back dimensions (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)

        # Apply attention
        att_output, _ = self.attention(x, x, x)  # Self-attention

        # Reduce dimensions with average pooling
        x = torch.mean(att_output, dim=1)  # (batch_size, hidden_size)

        # Fully connected output layer
        output = self.fc(x)

        return output
