import yfinance as yf
import pandas as pd
import numpy as np
import os

TICKERS = ['AAPL', 'MSFT', 'LLY', 'UNH', 'SBUX', 'NKE', 'NFLX']
start_date = '2015-01-01'
end_date = '2024-12-31'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
processed_dir = os.path.join(BASE_DIR, 'data', 'processed')

def download_data(tickers=TICKERS, start=start_date, end=end_date):
    os.makedirs(raw_dir, exist_ok=True)
    for ticker in tickers:
        print(f'Downloading {ticker}')
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(f'{raw_dir}/{ticker}.csv')
        print(f'Saved {len(df)} rows to data/raw/{ticker}.csv')
        
def engineer_features(ticker):
    df = pd.read_csv(f'{raw_dir}/{ticker}.csv', index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['return_1d'] = df['Close'].pct_change()
    
    df['ma5'] = df['Close'].rolling(5).mean() / df['Close']
    df['ma20'] = df['Close'].rolling(20).mean() / df['Close']
    
    df['vol5'] = df['return_1d'].rolling(5).std()
    df['vol20'] = df['return_1d'].rolling(20).std()
    
    df['vol_zscore'] = ((df['Volume'] - df['Volume'].rolling(20).mean()) /
                        df['Volume'].rolling(20).std())
    
    df['rsi14'] = compute_rsi(df['Close'], period=14)
    
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    df.dropna(inplace=True)
    return df
    
def compute_rsi(series, period = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100/(1 + rs))
    return rsi

def time_split(df):
    train = df[df.index < '2022-01-01']
    val = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
    test = df[df.index >= '2023-01-01']
    return train, val, test

def process_all(tickers=TICKERS):
    os.makedirs(processed_dir, exist_ok=True)
    all_dfs = []
    
    for ticker in tickers:
        df = engineer_features(ticker)
        df['ticker'] = ticker
        df.to_csv(f'{processed_dir}/{ticker}_features.csv')
        print(f'Processed {ticker}: {len(df)} rows, {df['target'].mean():.2%} up days')
        all_dfs.append(df)
        
    combined = pd.concat(all_dfs)
    combined.to_csv(f'{processed_dir}/combined.csv')
    print(f'\nCombined dataset: {len(combined)} rows across {len(tickers)} tickers')
    return combined

if __name__ == '__main__':
    download_data()
    process_all()