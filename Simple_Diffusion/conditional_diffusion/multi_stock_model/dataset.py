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
    def __init__(self, data, context_len, feature_columns, target_column='Close',split='train'):

        self.samples = []
        # Lets return train and test immidiately.
        train =[]
        test = []

        # Convert relevant columns to torch tensors
        for ticker in data:

            # Retrive dataframe
            df = data[ticker]

            # Sort values on date, ascending
            df.sort_values(by=['Date'],inplace=True)

            # All the samples of the specific ticker
            ticker_samples = []

            for i in range(0,len(df)-context_len): # Eg: (0, 10-2) : i = 0,1,..,7
                # For each dataframe extract the context window based on the feature columns
                context = df[i:i+context_len][feature_columns] # Eg i = 2, we take from i = 2 until 2+2-1 = 3

                # Get x0
                x0 = df[target_column].iloc[i+context_len] # We get item at position 2+2 = 4
            
                # Save a tuple containing ticker, the context and the x0
                context_tensor = torch.tensor(context.values,dtype=torch.float32)
                x0_tensor = torch.tensor(x0.values, dtype=torch.float32)
                grouped = (ticker,context_tensor,x0_tensor)
                # self.samples.append(grouped)
                ticker_samples.append(grouped)

            train_size = int(len(ticker_samples)*0.8)
            # print(ticker,train_size)
            train.extend(ticker_samples[:train_size])
            test.extend(ticker_samples[train_size:])
            if split =="train":
                self.samples = train
            else:
                self.samples = test

        # Now that we have all the dta, its time for normalization (min max)
        # To do that, we will compute the min and max on the training data for features and targets
        # Then we will use that same info to apply normalization to test data too
        
        # Calculate training max and min
        contexts = [sample[1] for sample in train]
        contexts_cat = torch.cat(contexts,dim=0)
        self.context_min,_ = torch.min(contexts_cat,dim=0)
        self.context_max,_ = torch.max(contexts_cat,dim=0)
        # For the targets
        targets = torch.stack([sample[2] for sample in train],dim=0)
        self.target_min = torch.min(targets)
        self.target_max = torch.max(targets)
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker,context,x0 = self.samples[idx]
        # Normalized context:
        context_norm = (context- self.context_min)/(self.context_max - self.context_min + 1e-8)
        x0_norm = (x0- self.target_min)/(self.target_max - self.target_min + 1e-8)
        
        return (ticker,context_norm,x0_norm)

