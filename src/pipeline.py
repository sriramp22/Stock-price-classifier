import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import StockDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'combined.csv')

def main():
    df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    print(f"loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
    
    train_df = df[df.index < '2022-01-01']
    val_df = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
    test_df = df[df.index >= '2023-01-01']
    
    train_ds = StockDataset(train_df)
    val_ds = StockDataset(val_df)
    test_ds = StockDataset(test_df)
    
    print(f'The total number of windows in training set is {len(train_ds)}')
    print(f'The total number of windows in validation set is {len(val_ds)}')
    print(f'The total number of windows in test set is {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    X_batch, y_batch = next(iter(train_loader))
    print(f'X_batch shape: {X_batch.shape}')
    print(f'y_batch shape: {y_batch.shape}')
    
if __name__ == '__main__':
    main()