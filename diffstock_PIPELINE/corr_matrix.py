import numpy as np
import pandas as pd

def compute_stock_relation_matrix(stock_data, relation_type='return_correlation', threshold=0.5):
    """
    Compute a relation matrix among stocks based on the specified relation type.
    
    Currently implemented relation:
      - 'return_correlation': Uses the correlation of daily returns.
    
    Args:
        stock_data: dict, mapping ticker -> DataFrame.
        relation_type: str, type of relation to compute.
        threshold: float, the absolute correlation threshold for a relation.
        
    Returns:
        tickers: list of tickers in order.
        relation_matrix: numpy array of shape (N, N, 1) where N is the number of stocks.
    """
    tickers = list(stock_data.keys())
    N = len(tickers)
    # We start with G=1 relation type.
    relation_matrix = np.zeros((N, N, 1))
    
    if relation_type == 'return_correlation':
        for i in range(N):
            for j in range(N):
                # Self-relation is always 1
                if i == j:
                    relation_matrix[i, j, 0] = 1
                else:
                    df_i = stock_data[tickers[i]]
                    df_j = stock_data[tickers[j]]
                    # Merge on Date to align the two series (assumes each df has a 'Date' column)
                    merged = pd.merge(df_i[['Date', 'Return']], df_j[['Date', 'Return']], on='Date', suffixes=('_i', '_j'))
                    if len(merged) > 0:
                        corr = merged['Return_i'].corr(merged['Return_j'])
                        # If the absolute correlation meets/exceeds the threshold, mark as related.
                        relation_matrix[i, j, 0] = 1 if abs(corr) >= threshold else 0
                    else:
                        relation_matrix[i, j, 0] = 0
    else:
        raise ValueError(f"Unknown relation type: {relation_type}")
        
    return tickers, relation_matrix
