import numpy as np
import pandas as pd

def compute_stock_relation_matrix(stock_data, relation_types=['return_correlation'], return_threshold=0.5, MA5_threshold=0.5):
    """
    Compute a relation matrix among stocks based on one or more specified relation types.
    
    For each relation type, the function computes a binary relation between stocks
    (1 if related, 0 otherwise) based on a threshold.
    
    Args:
        stock_data: dict, mapping ticker -> DataFrame.
        relation_types: list of strings, each specifying a type of relation.
                        For example, ['return_correlation', 'MA5_correlation'].
        threshold: float, the absolute correlation threshold for a relation.
        
    Returns:
        tickers: list of tickers in order.
        relation_matrix: numpy array of shape (N, N, G) where N is the number of stocks and
                         G is the number of relation types.
    """
    tickers = list(stock_data.keys())
    N = len(tickers)
    G = len(relation_types)
    relation_matrix = np.zeros((N, N, G))
    
    for g, rtype in enumerate(relation_types):
        if rtype == 'return_correlation':
            for i in range(N):
                for j in range(N):
                    if i == j:
                        relation_matrix[i, j, g] = 1
                    else:
                        df_i = stock_data[tickers[i]]
                        df_j = stock_data[tickers[j]]
                        # Merge on Date to align the two series (assumes each df has a 'Date' column)
                        merged = pd.merge(
                            df_i[['Date', 'Return']],
                            df_j[['Date', 'Return']],
                            on='Date',
                            suffixes=('_i', '_j')
                        )
                        if len(merged) > 0:
                            corr = merged['Return_i'].corr(merged['Return_j'])
                            relation_matrix[i, j, g] = 1 if abs(corr) >= return_threshold else 0
                        else:
                            relation_matrix[i, j, g] = 0
                            
        elif rtype == 'MA5_correlation':
            for i in range(N):
                for j in range(N):
                    if i == j:
                        relation_matrix[i, j, g] = 1
                    else:
                        df_i = stock_data[tickers[i]]
                        df_j = stock_data[tickers[j]]
                        # Merge on Date to align the two series (assumes each df has a 'Date' column)
                        merged = pd.merge(
                            df_i[['Date', 'MA5']],
                            df_j[['Date', 'MA5']],
                            on='Date',
                            suffixes=('_i', '_j')
                        )
                        if len(merged) > 0:
                            corr = merged['MA5_i'].corr(merged['MA5_j'])
                            relation_matrix[i, j, g] = 1 if abs(corr) >= MA5_threshold else 0
                        else:
                            relation_matrix[i, j, g] = 0
        else:
            raise ValueError(f"Unknown relation type: {rtype}")
    
    return tickers, relation_matrix

