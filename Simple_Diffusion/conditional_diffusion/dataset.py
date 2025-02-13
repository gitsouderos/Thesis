import torch
from torch.utils.data import Dataset
import numpy as np

class ConditionalStockDataset(Dataset):
    def __init__(self, data, context_len, feature_columns, target_column='Close'):

        # Convert relevant columns to torch tensors
        self.features = torch.tensor(data[feature_columns].values, dtype=torch.float32)
        self.targets = torch.tensor(data[target_column].valeus, dtype = torch.float32)
        self.context_len = context_len

    def __len__(self):
        return len(self.features)- self.context_len
    
    def __getitem__(self, idx):
        context = self.features[idx:idx+self.context_len]
        x0 = self.targets[idx+self.context_len]

        return context, x0