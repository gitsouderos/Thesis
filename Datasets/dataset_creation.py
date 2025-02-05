# Simple Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class StockDataset(Dataset):
    def __init__(self, data, seq_len, feature_columns):
        # Inputs
        self.features = torch.tensor(data[feature_columns].values, dtype=torch.float32) # (Num_samples, Num_features)
        # Targets  
        self.targets = torch.tensor(data['Close'].values, dtype=torch.float32) # (Num_samples,)
        # Sequence to look back 
        self.seq_len = seq_len

    def __len__(self):
        # Number of valid sequences
        return len(self.features) - self.seq_len  

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len] 
        # Prediction 
        y = self.targets[idx + self.seq_len]  
        return x, y