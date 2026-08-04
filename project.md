# Project Context: Commodity Price Forecasting Platform

## Project Overview
 
I am building an AI-powered commodity price forecasting platform

6 commodities I want to forecast:
1.Gold
2.Silver
3.Crude oil
4.Natural gas
5.Copper
6.Cotton

I am a beginner in ML and want to build this project **incrementally, stage by stage** — starting with simple numerical price data (OHLCV data from yahoo finance), validating with baseline/statistical models (ARIMA, Prophet) then ML models like (XGBoost, LightGBM), DL models(RNN,LSTM, Transformers), then progressively adding richer data sources (macro, fundamentals, news, sentiment) , trying the models as earlier , then trying the more sophisticated models, testing at every stage.
 
**Important working principle:** Do not jump ahead to advanced stages. Help me complete and validate each stage before moving to the next. At every stage, compare new results against the previous stage's best model — improvements must be measurable, not assumed.

## Commodities in Scope
 
Six commodities, **each with its own separate model** (not one combined model), because each has distinct price drivers, volatility profiles, and relevant data:
 
| Commodity | Ticker (Yahoo Finance) | Category |
|---|---|---|
| Gold | GC=F | Precious metal |
| Silver | SI=F | Precious metal |
| Copper | HG=F | Industrial metal |
| Natural Gas | NG=F | Energy |
| Crude Oil | CL=F | Energy |
| Wheat | ZW=F | Agricultural crop |
 

## Project Scope — Staged Roadmap:
 
### Stage 0 — Data Pipeline Setup
- Build ingestion pipeline pulling daily OHLCV (Open, High, Low, Close, Adj Close, Volume) for all 6 tickers via `yfinance` and save it in csv for each commodity e.g gold.csv, silver.csv, etc. I want to maintain the same columns and order as `yfinance`. Make sure you do the same for all the 6 commodities in one go. Each commodity has a different name in yahoo finance , so use the appropriate ticker and name.

- Target ~10 years of daily history (2014–present) as the default window, adjustable per commodity later

- Clean and align data: handle missing trading days, verify no gaps, handle known extreme outliers (e.g., crude oil's negative price event in April 2020) explicitly rather than silently

### Stage 1 — Baseline Model
- Naive forecast baseline: tomorrow's price = today's price (per commodity)

- This is the benchmark every later model must be measurably compared against

- Establish evaluation metrics: MAE, RMSE, and directional accuracy (up/down correctness), tracked per commodity


### Stage 2 — Train Statistical Time Series Models(ARIMA,Prophet) on OHLCV only while using ML Models(XGBoost,LightGBM)+ DL Models(LSTM, Transformers) on OHLCV+Technical Features only

-For each commodity :

- Derive following technical features purely from OHLCV: 
## Price-based:

Return — daily % change in Close (Close.pct_change())

Log_Return — log of price ratio (more statistically well-behaved than raw % return)

Lag_1, Lag_2, Lag_3, Lag_5, Lag_10 — Close price N days ago

Daily_Range — High - Low (a simple volatility proxy)

Daily_Range_Pct — (High - Low) / Close

## Rolling/moving averages:

MA_5, MA_10, MA_20, MA_50 — simple moving averages of Close

EMA_10, EMA_20 — exponential moving averages (react faster to recent moves)

Rolling_Std_10, Rolling_Std_20 — rolling volatility (standard deviation of returns)

## Momentum/technical indicators (still just derived from OHLCV):

RSI_14 — Relative Strength Index, classic overbought/oversold indicator

MACD — Moving Average Convergence Divergence

Volume_MA_10 — rolling average of volume (spikes can signal news events)

## Target column (what I am predicting):

Target_Close_Next — next day's close (for 1-day-ahead forecast)

Target_Return_Next — if I'm framing it as return prediction (often better — more stationary than raw price)

Target_Direction — (1 if price goes up, 0 if down) for evaluating directional accuracy separately.



That gives us ~20 columns per commodity, all derivable from just OHLCV


## Note that after deriving all these features, add them into the corresponding commodity csv file with the same order and same column names as mentioned above.Also note that the first few rows will have NaN values for some features (e.g. RSI, MACD, etc.) due to the rolling window calculations. We should drop these rows before training the models. 

## Note : Store all the csv files in 'data' sub-folder(inside p:/ffcproj/) as 'commodity_name.csv' for each commodity


After this : 

- Train ARIMA and/or Prophet models per commodity using Close price history 

- Chronological train/test split (never random shuffle — this is time series data)

- Compare against Stage 1 baseline

- Train XGBoost and LightGBM models per commodity using Close price history and technical features 

- Train RNN,LSTM and Transformers models per commodity using Close price history and technical features 

- Compare against Stage 1 baseline and ARIMA and Prophet models




