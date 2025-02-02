import os
import pandas as pd
import numpy as np

def compute_financial_indicators(df):
    """
    Given a DataFrame with daily stock data, compute additional financial indicators.
    
    Expected columns in df: Date, Open, High, Low, Close, Volume.
    
    Returns:
        A DataFrame with additional columns:
          - Return: (Close - Open) / Open
          - Diff: Close - Open
          - HL_Diff: High - Low
          - MA5: 5-day moving average of Close (shifted to avoid leakage)
          - Return_MA5: 5-day moving average of Return (shifted)
    """
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
    Load CSV files for all stocks from the specified folder.
    
    Assumes each CSV filename is like "AAPL.csv", "MSFT.csv", etc.
    
    Returns:
        A dictionary mapping ticker symbols to their processed DataFrames.
    """
    stock_data = {}
    for file in os.listdir(data_folder):
        if file.endswith('.csv'):
            ticker = file.split('.')[0]  # e.g., "AAPL" from "AAPL.csv"
            file_path = os.path.join(data_folder, file)
            df = pd.read_csv(file_path)
            df = compute_financial_indicators(df)
            stock_data[ticker] = df
            print(f"Loaded and processed data for {ticker}: {len(df)} records.")
    return stock_data

