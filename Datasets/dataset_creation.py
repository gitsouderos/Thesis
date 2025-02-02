# Simple Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class StockDataset(Dataset):
    def __init__(self, data, seq_len, feature_columns):
        self.features = torch.tensor(data[feature_columns].values, dtype=torch.float32)  # Features
        self.targets = torch.tensor(data['Close'].values, dtype=torch.float32)  # Target is Close price
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len  # Total available sequences

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]  # Input sequence
        y = self.targets[idx + self.seq_len]  # Target next day's Close price
        return x, y