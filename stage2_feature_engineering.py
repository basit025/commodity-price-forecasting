import pandas as pd
import numpy as np
import os
import ta

data_dir = './data'
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def engineer_features(df):
    # Ensure data is sorted by index just in case
    df = df.sort_index()
    
    # ---------------------------
    # Price-based Features
    # ---------------------------
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    df['Lag_5'] = df['Close'].shift(5)
    df['Lag_10'] = df['Close'].shift(10)
    
    df['Daily_Range'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    # ---------------------------
    # Rolling / Moving Averages
    # ---------------------------
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    df['Rolling_Std_10'] = df['Return'].rolling(window=10).std()
    df['Rolling_Std_20'] = df['Return'].rolling(window=20).std()
    
    # ---------------------------
    # Momentum / Technical Indicators
    # ---------------------------
    # RSI 14
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(close=df['Close'])
    # By default ta uses standard parameters: window_fast=12, window_slow=26, window_sign=9
    df['MACD'] = macd_indicator.macd_diff() # Standard MACD histogram/difference
    
    # Volume MA 10
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    
    # ---------------------------
    # Target Columns (What we predict)
    # ---------------------------
    df['Target_Close_Next'] = df['Close'].shift(-1)
    df['Target_Return_Next'] = df['Return'].shift(-1)
    # Direction: 1 if next day's return > 0 else 0
    # Note: Using > 0 strictly means exactly 0 return is class 0.
    df['Target_Direction'] = (df['Target_Return_Next'] > 0).astype(int)
    # Correct the last row where Target_Return_Next is NaN, so Target_Direction should also be NaN
    df.loc[df['Target_Return_Next'].isna(), 'Target_Direction'] = np.nan
    
    return df

def main():
    for name in commodities:
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {name}: Data file not found.")
            continue
            
        print(f"Processing {name}...")
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # We assume the OHLCV columns are already there from Stage 0
        df = engineer_features(df)
        
        # Check rows before drop
        initial_rows = len(df)
        
        # Drop rows with NaN (due to rolling windows, lags, and shift(-1) target)
        df.dropna(inplace=True)
        
        final_rows = len(df)
        print(f"Dropped {initial_rows - final_rows} rows due to NaN. Final shape: {df.shape}")
        
        # Save back to CSV, overwriting with new features
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        df.to_csv(file_path)
        print(f"Saved {name}.csv with technical features.")
        print("-" * 50)

if __name__ == "__main__":
    main()
