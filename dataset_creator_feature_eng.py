# Define a custom Dataset that uses only engineered features as input,
# and uses the raw (normalized) Close price as the target.

# Simple Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class StockDataset(Dataset):
    def __init__(self, data, seq_len, feature_columns, target_column='Close'):
        self.features = torch.tensor(data[feature_columns].values, dtype=torch.float32)
        self.targets = torch.tensor(data[target_column].values, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len -1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len +1]
        return x, y