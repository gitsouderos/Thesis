import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def compute_financial_indicators(df):
   
    df = df.copy()
    # Make sure the data is sorted by Date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Daily Return (percentage change)
    df['Return'] = (df['Close'] - df['Open']) / df['Open']
    
    # Price Difference
    df['Diff'] = df['Close'] - df['Open']
    
    # High-Low Difference (a simple proxy for volatility)
    df['HL_Diff'] = df['High'] - df['Low']
    
    # 5-day Moving Average of the Close (shifted by one day to avoid leakage)
    df['MA5'] = df['Close'].rolling(window=5).mean().shift(1)
    
    # 5-day Moving Average of the Return (shifted)
    df['Return_MA5'] = df['Return'].rolling(window=5).mean().shift(1)
    
    # Drop rows with NaN values from the rolling calculations
    df = df.dropna().reset_index(drop=True)
    return df

def load_all_stock_data(data_folder):
    """
    A dictionary mapping ticker symbols to their processed DataFrames.
    """
    stock_data = {}
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            ticker = file.split('.')[0]  
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            df = compute_financial_indicators(df)
            stock_data[ticker] = df
    print(f"Loaded and processed data for {len(stock_data)} stocks")
    return stock_data




class ConditionalStockDataset(Dataset):
    def __init__(self, data, context_len, feature_columns, target_column='Close'):

        # Convert relevant columns to torch tensors
        for ticker in data:

            self.features = torch.tensor(data[ticker][feature_columns].values, dtype=torch.float32)
            self.targets = torch.tensor(data[ticker][target_column].values, dtype = torch.float32)
            self.context_len = context_len

    def __len__(self):
        return len(self.features)- self.context_len
    
    def __getitem__(self, idx):
        context = self.features[idx:idx+self.context_len]
        x0 = self.targets[idx+self.context_len]

        return context, x0

