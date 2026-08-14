import torch
from torch.utils.data import Dataset
import numpy as np

feature_cols = ['return_1d', 'ma5', 'ma20', 'vol5', 'vol20', 'vol_zscore', 'rsi14']
target_col = 'target'

class StockDataset(Dataset):
    def __init__(self, df, window_size=20):
        self.window_size = window_size
        all_X = []
        all_y = []
        
        for ticker in df['ticker'].unique():
            ticker_df = df[df['ticker']==ticker].sort_index()
            X_ticker, y_ticker = self._build_windows(ticker_df)
            all_X.append(X_ticker)
            all_y.append(y_ticker)
        
        self.X = torch.tensor(np.concatenate(all_X), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(all_y), dtype=torch.long)
        
    def _build_windows(self, ticker_df):
        features = ticker_df[feature_cols].values
        targets = ticker_df[target_col].values
        X_ticker = []
        y_ticker = []
        
        for i in range(len(features) - self.window_size):
            X_ticker.append(features[i: i + self.window_size])
            y_ticker.append(targets[i + self.window_size])
        
        X_ticker = np.stack(X_ticker)
        y_ticker = np.array(y_ticker)
        
        return X_ticker, y_ticker
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    