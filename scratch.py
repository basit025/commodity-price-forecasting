import json
import os

results_dir = './results'

# Load the JSON files
with open(os.path.join(results_dir, 'stage2_stat_models_metrics.json'), 'r') as f:
    stat_data = json.load(f)
    
with open(os.path.join(results_dir, 'stage2_ml_models_metrics.json'), 'r') as f:
    ml_data = json.load(f)
    
with open(os.path.join(results_dir, 'stage2_dl_models_metrics.json'), 'r') as f:
    dl_data = json.load(f)

commodities = ['gold', 'silver', 'copper', 'natural_gas', 'crude_oil', 'wheat']

md_content = "# Comprehensive Forecasting Results\n\n"
md_content += "This document compiles the performance metrics (MAE, RMSE, and Directional Accuracy) of all models evaluated across the different stages of the project.\n\n"

for comm in commodities:
    md_content += f"## {comm.replace('_', ' ').title()}\n\n"
    
    md_content += "| Model Category | Specific Model | MAE | RMSE | Directional Accuracy (%) |\n"
    md_content += "|---|---|---|---|---|\n"
    
    # Statistical / Baseline
    if comm in stat_data:
        naive = stat_data[comm]['Naive']
        arima = stat_data[comm]['ARIMA']
        md_content += f"| **Baseline** | Naive (Random Walk) | {naive['MAE']:.4f} | {naive['RMSE']:.4f} | {naive['Dir_Acc']:.2f}% |\n"
        md_content += f"| **Statistical** | ARIMA(1,1,1) | {arima['MAE']:.4f} | {arima['RMSE']:.4f} | {arima['Dir_Acc']:.2f}% |\n"
        
    # Machine Learning
    if comm in ml_data:
        xgb_r = ml_data[comm]['XGB_Return_Target']
        lgb_r = ml_data[comm]['LGB_Return_Target']
        md_content += f"| **Machine Learning** | XGBoost (Returns) | {xgb_r['MAE']:.4f} | {xgb_r['RMSE']:.4f} | {xgb_r['Dir_Acc']:.2f}% |\n"
        md_content += f"| **Machine Learning** | LightGBM (Returns) | {lgb_r['MAE']:.4f} | {lgb_r['RMSE']:.4f} | {lgb_r['Dir_Acc']:.2f}% |\n"
        
    # Deep Learning
    if comm in dl_data:
        lstm = dl_data[comm]['LSTM']
        transformer = dl_data[comm]['Transformer']
        md_content += f"| **Deep Learning** | LSTM | {lstm['MAE']:.4f} | {lstm['RMSE']:.4f} | {lstm['Dir_Acc']:.2f}% |\n"
        md_content += f"| **Deep Learning** | Transformer | {transformer['MAE']:.4f} | {transformer['RMSE']:.4f} | {transformer['Dir_Acc']:.2f}% |\n"
        
    md_content += "\n"

out_path = os.path.join(results_dir, 'results.md')
with open(out_path, 'w') as f:
    f.write(md_content)

print(f"Results successfully compiled into {out_path}")
