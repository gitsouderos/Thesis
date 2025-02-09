import torch
from torch.utils.data import Dataset
import numpy as np
from diffusion import forward_diffusion_sample

class StockDiffusionDataset(Dataset):
    def __init__(self, data, target_column='Close'):
        # Convert the target column to a tensor.
        self.targets = torch.tensor(data[target_column].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        # For each sample, return the clean target.
        # Optionally, you can add dimension reshaping if needed.
        x0 = self.targets[idx]
        # print(x0.shape)
        # x0 = torch.tensor([x0], dtype=torch.float32)
        # If x0 is a scalar and you want a tensor of shape [1],
        # you might do: x0 = x0.unsqueeze(0) or x0 = torch.tensor([x0], dtype=torch.float32)
        return x0