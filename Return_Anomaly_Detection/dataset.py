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
    
    # Intraday Return (percentage change)
    df['Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100

    # Overnight return (percentage change)
    df['Overnight_Return'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)) * 100

    # True range max(High - Low, |High - Close_prev|, |Low - Close_prev|) / Close_prev * 100%
    df['TR1'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
    df['TR2'] = (np.abs(df['High'] - df['Close'].shift(1))) / df['Close'].shift(1) * 100
    df['TR3'] = (np.abs(df['Low'] - df['Close'].shift(1))) / df['Close'].shift(1) * 100
    df['TR'] = np.max(df[['TR1', 'TR2', 'TR3']], axis=1)

    # Rolling volatility (standard deviation of returns)
    df['Vol_5d'] = df['Return'].rolling(5).std()
    df['Vol_10d'] = df['Return'].rolling(10).std()

    # Z-score of daily returns (shows how unusual today's return is)
    df['Return_Z'] = (df['Return'] - df['Return'].rolling(20).mean()) / df['Return'].rolling(20).std()

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
    def __init__(self, data, context_len, feature_columns, target_column='Close', split='train', device='cpu', normalization= 'min_max'):
        self.device = device
        self.samples = []
        # Lets return train and test immediately.
        train =[]
        test = []
        self.context_scalers = {}
        self.target_scalers = {}
        self.normalization = normalization

        # Convert relevant columns to torch tensors
        for ticker in data:
            # Retrieve dataframe
            df = data[ticker]
            # Sort values on date, ascending, inplace
            df.sort_values(by=['Date'], inplace=True)
            # All the samples of the specific ticker
            ticker_samples = []

            for i in range(0, len(df) - context_len):  # Eg: (0, 10-2) : i = 0,1,..,7
                # Extract the context window based on the feature columns
                context = df[i:i+context_len][feature_columns]
                # Get x0
                x0 = df[target_column].iloc[i+context_len-1]
                # Convert to tensors
                context_tensor = torch.tensor(context.values, dtype=torch.float32)
                # If x0 is a scalar inside a Series, get its value
                x0_tensor = torch.tensor(x0 if np.isscalar(x0) else x0.values, dtype=torch.float32)
                grouped = (ticker, context_tensor, x0_tensor)
                ticker_samples.append(grouped)
            
            # Perform min-max normalization on the context and x0
            contexts = [sample[1] for sample in ticker_samples]
            contexts_cat = torch.cat(contexts, dim=0)
            context_min, _ = torch.min(contexts_cat, dim=0)
            context_max, _ = torch.max(contexts_cat, dim=0)
            context_std = torch.std(contexts_cat, dim=0)
            context_mean = torch.mean(contexts_cat, dim=0)

            # For the targets
            targets = torch.stack([sample[2] for sample in ticker_samples], dim=0)
            target_min = torch.min(targets)
            target_max = torch.max(targets)
            target_std = torch.std(targets)
            target_mean = torch.mean(targets)

            # Store scalers for the ticker
            self.context_scalers[ticker] = (context_min, context_max, context_std, context_mean)
            self.target_scalers[ticker] = (target_min, target_max, target_std, target_mean)
            
            # Split into train (80%) and test (20%)
            train_size = int(len(ticker_samples) * 0.8)
            train.extend(ticker_samples[:train_size])
            test.extend(ticker_samples[train_size:])
            if split == "train":
                self.samples = train
            else:
                self.samples = test
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, context, x0 = self.samples[idx]
        # Retrieve scalers for normalization
        context_min = self.context_scalers[ticker][0]
        context_max = self.context_scalers[ticker][1]
        context_std = self.context_scalers[ticker][2]
        context_mean = self.context_scalers[ticker][3]
        target_min = self.target_scalers[ticker][0]
        target_max = self.target_scalers[ticker][1]
        target_std = self.target_scalers[ticker][2]
        target_mean = self.target_scalers[ticker][3]

        if self.normalization == 'min_max':
            context_norm = (context - context_min) / (context_max - context_min + 1e-8)
            x0_norm = (x0 - target_min) / (target_max - target_min + 1e-8)
        elif self.normalization == 'standard':
            context_norm = (context - context_mean) / (context_std + 1e-8)
            x0_norm = (x0 - target_mean) / (target_std + 1e-8)
        # Move tensors to the proper device
        return (ticker, context_norm.to(self.device), x0_norm.to(self.device))
