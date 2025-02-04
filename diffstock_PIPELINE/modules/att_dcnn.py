##############################################
# 2. Att-DCNN: Attention-based Dilated CNN for Temporal Feature Extraction
##############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttDCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation_rates=[1, 2, 4]):
        """
        Args:
            input_dim: Number of features per day in the historical data.
            hidden_dim: Number of output channels (feature dimension) per stock.
            kernel_size: Size of the convolution kernel.
            dilation_rates: List of dilation rates to capture different temporal scales.
        """
        super(AttDCNN, self).__init__()
        self.convs = nn.ModuleList()
        for d in dilation_rates:
            padding = (kernel_size - 1) * d // 2  # to preserve the sequence length
            self.convs.append(
                nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                          kernel_size=kernel_size, dilation=d, padding=padding)
            )
        # Fully-connected layer to further process the aggregated convolution outputs.
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, N, L, input_dim)
               - batch_size: number of examples
               - N: number of stocks
               - L: number of historical days
               - input_dim: number of features per day
        Returns:
            Tensor of shape (batch_size, N, hidden_dim)
        """
        batch_size, N, L, input_dim = x.size()
        # Reshape: merge batch and stocks, and prepare for 1D convolution over time
        x = x.view(batch_size * N, L, input_dim).permute(0, 2, 1)  # (batch_size*N, input_dim, L)
        
        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))  # (batch_size*N, hidden_dim, L)
            conv_outputs.append(out)
        
        # Average across different dilation rates
        out = torch.stack(conv_outputs, dim=0).mean(dim=0)  # (batch_size*N, hidden_dim, L)
        # Global average pooling over the time dimension
        out = out.mean(dim=2)  # (batch_size*N, hidden_dim)
        out = self.fc(out)     # (batch_size*N, hidden_dim)
        # Reshape back to (batch_size, N, hidden_dim)
        out = out.view(batch_size, N, -1)
        return out