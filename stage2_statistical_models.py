import pandas as pd
import numpy as np
import os
import json
import warnings
from prophet import Prophet
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings from Prophet and ARIMA
warnings.filterwarnings('ignore')

data_dir = 'p:/ffcproj/data'
results_dir = 'p:/ffcproj/results'
os.makedirs(results_dir, exist_ok=True)
commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

def calculate_metrics(y_true, y_pred, y_true_prev):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate directional accuracy
    actual_dir = np.sign(y_true - y_true_prev)
    pred_dir = np.sign(y_pred - y_true_prev)
    
    correct_dir = (actual_dir == pred_dir)
    dir_acc = np.mean(correct_dir) * 100
    
    return float(mae), float(rmse), float(dir_acc)

def main():
    results = {}
    
    for name in commodities:
        print(f"--- Processing {name.upper()} ---")
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"Data for {name} not found.")
            continue
            
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # Chronological train/test split (80/20) on the clean dataset
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()
        
        y_train = train_df['Close'].values
        y_test = test_df['Close'].values
        
        # y_true_prev is needed to calculate directional accuracy.
        # For the test set, the "previous" value of the first element is the last element of the train set.
        y_true_prev = np.concatenate([[y_train[-1]], y_test[:-1]])
        
        # --- 1. Prophet Model ---
        print("  Training Prophet...")
        prophet_df = train_df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        # Remove timezone to prevent Prophet errors
        if prophet_df['ds'].dt.tz is not None:
             prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
             
        # Initialize and fit
        m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        m.fit(prophet_df)
        
        # Predict on test dates
        future = test_df.reset_index()[['Date']].rename(columns={'Date': 'ds'})
        if future['ds'].dt.tz is not None:
             future['ds'] = future['ds'].dt.tz_localize(None)
             
        forecast = m.predict(future)
        prophet_preds = forecast['yhat'].values
        p_mae, p_rmse, p_dir = calculate_metrics(y_test, prophet_preds, y_true_prev)
        
        # --- 2. ARIMA Model (Auto-ARIMA) ---
        print("  Training Auto-ARIMA...")
        # Limiting max_p and max_q to speed up auto_arima (financial data usually doesn't need huge lags)
        # We don't use seasonal=True as it makes ARIMA extremely slow for daily data
        arima_model = pm.auto_arima(
            y_train, 
            seasonal=False, 
            max_p=3, 
            max_q=3, 
            d=None, # let it automatically determine differencing
            trace=False, 
            suppress_warnings=True, 
            error_action='ignore'
        )
        arima_preds = arima_model.predict(n_periods=len(y_test))
        a_mae, a_rmse, a_dir = calculate_metrics(y_test, arima_preds, y_true_prev)
        
        # --- 3. Naive Baseline (Re-calculated on this exact subset) ---
        # The naive prediction is exactly the previous day's close
        naive_preds = y_true_prev
        n_mae, n_rmse, n_dir = calculate_metrics(y_test, naive_preds, y_true_prev)
        
        # Save results
        results[name] = {
            'Test_Size': len(y_test),
            'Naive': {'MAE': round(n_mae,4), 'RMSE': round(n_rmse,4), 'Dir_Acc': round(n_dir,2)},
            'Prophet': {'MAE': round(p_mae,4), 'RMSE': round(p_rmse,4), 'Dir_Acc': round(p_dir,2)},
            'ARIMA': {'MAE': round(a_mae,4), 'RMSE': round(a_rmse,4), 'Dir_Acc': round(a_dir,2)}
        }
        
        print(f"  [Results] Naive Baseline : MAE {n_mae:.4f} | RMSE {n_rmse:.4f} | Dir Acc {n_dir:.2f}%")
        print(f"  [Results] Prophet        : MAE {p_mae:.4f} | RMSE {p_rmse:.4f} | Dir Acc {p_dir:.2f}%")
        print(f"  [Results] ARIMA          : MAE {a_mae:.4f} | RMSE {a_rmse:.4f} | Dir Acc {a_dir:.2f}%")
        print()
        
    out_path = os.path.join(results_dir, 'stage2_stat_models_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"All statistical models trained and evaluated. Results saved to {out_path}.")

if __name__ == "__main__":
    main()
