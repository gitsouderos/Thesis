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

        self.samples = []

        # Convert relevant columns to torch tensors
        for ticker in data:

            # Retrive dataframe
            df = data[ticker]

            # Sort values on date, ascending
            df.sort_values(by=['Date'])

            for i in range(0,len(df)-context_len): # Eg: (0, 10-2) : i = 0,1,..,7
                # For each dataframe extract the context window based on the feature columns
                context = df[feature_columns][i:i+context_len-1] # Eg i = 2, we take from i = 2 until 2+2-1 = 3

                # Get x0
                x0 = df[target_column].iloc[i+context_len] # We get item at position 2+2 = 4
            
                # Save a tuple containing ticker, the context and the x0
                context_tensor = torch.tensor(context.values,dtype=torch.float32)
                x0_tensor = torch.tensor(x0.values, dtype=torch.float32)
                grouped = (ticker,context_tensor,x0_tensor)
                self.samples.append(grouped)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker = self.samples[idx][0]
        context = self.samples[idx][1]
        x0 = self.samples[idx][2]
        return (ticker,context,x0)

