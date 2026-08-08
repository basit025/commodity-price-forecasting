# System Explanation: Commodity Forecasting Pipeline

This document explains what the commodity forecasting system actually is, how it works end-to-end, and what it outputs. Use this as grounding context before generating code — the goal is a working pipeline that matches this shape, not a redesign.

## The big picture

This is one pipeline, run separately for each of the 6 commodities (gold, silver, copper, natural gas, crude oil, wheat). Each commodity gets its own trained models and its own output — there is no single combined model across commodities. The pipeline has five stages:

```
Data collection → Feature engineering → Multiple models (trained in parallel) → Ensemble → Forecast output
```

## Stage 1 — Data collection

- Daily OHLCV data per commodity (Open, High, Low, Close, Adj Close, Volume), ~10 years of history
- Macro/fundamental data merged in on top (USD Index, yields, VIX, CPI, plus commodity-specific sources like EIA inventory, USDA reports, weather data)
- Output of this stage: one clean daily-frequency CSV/table per commodity, no missing values, macro data aligned by actual release date (not the period it describes) to avoid look-ahead bias

## Stage 2 — Feature engineering

- Derived from OHLCV: lag values, moving averages, RSI, MACD, rolling volatility, daily range
- Derived from macro data: forward-filled daily values of each macro series, aligned to release date
- Output: one feature table per commodity — this exact same table is fed into all three models in Stage 3
- Feature selection (deciding what to keep/drop) happens after an initial model run, using SHAP values or permutation importance — not guessed in advance

## Stage 3 — Multiple models, trained in parallel

Models are trained on the same feature table per commodity. The roster includes the current foundation models, alongside future expansions designed to test specific architectural advantages:

### Tree-Based Models (No scaling required)
1. **XGBoost** — Good at finding non-linear threshold rules across features. The primary baseline.
2. **LightGBM** — Similar strengths to XGBoost, leaf-wise growth makes it faster on larger feature sets.
3. **CatBoost (Future Addition)** — Handles noisy/smaller datasets well out of the box. Has built-in handling for categorical features (useful when we add things like "season" or "region" as macro features) and often needs less hyperparameter tuning than XGBoost.
4. **Random Forest (Future Addition)** — Simpler and more robust to overfitting than boosted trees. A great sanity-check baseline against XGBoost/LightGBM since it is less prone to chasing noise.

### Deep Learning Models (Requires scaled input & sequences)
5. **LSTM** — Sequence model that captures temporal patterns across time (e.g. last 10 days → predict day 11).
6. **GRU / Gated Recurrent Unit (Future Addition)** — Simpler cousin of LSTM with fewer parameters. Trains faster and sometimes performs comparably on smaller datasets like ours (~2,500 rows/commodity) — worth trying as a lighter alternative.
7. **Temporal Fusion Transformer / TFT (Future Addition)** — Purpose-built for exactly this kind of problem: multi-horizon forecasting with mixed data types (price, macro, sentiment). Features built-in interpretability (attention weights double as feature importance). A perfect mid-project stretch goal once macro/sentiment data is in play.
8. **N-BEATS (Future Addition)** — A newer, pure time-series deep learning architecture designed specifically for forecasting (not adapted from NLP like LSTM/Transformer). Known to perform well without heavy feature engineering — interesting to compare against our heavily engineered tree models.

Each model is evaluated independently against a naive baseline (tomorrow's price = today's price) using MAE, RMSE, and directional accuracy. It's expected and normal for different models to win on different commodities — e.g. XGBoost may perform best for gold while LSTM performs best for natural gas.

## Stage 4 — Ensemble / model combination

- Combine the three models' predictions into one final prediction per commodity, typically a weighted average
- Weighting is based on each model's recent historical accuracy for that specific commodity — not a fixed global weight applied to every commodity the same way
- This produces both a point estimate and, ideally, a range/confidence interval rather than a single exact number

## Stage 5 — Forecast output

The final output per commodity is a structured result containing:

- **Predicted price range** (e.g. $2,395–$2,460), not a single point value
- **Direction** (bullish/bearish/neutral)
- **Confidence score** (based on model agreement / historical accuracy)
- **Which model or ensemble** produced this specific forecast
- **Top drivers** — a short, plain-language explanation of what's pushing the forecast, derived from SHAP feature importance values on the winning model

This structured result (essentially a JSON object per commodity) is what feeds the dashboard/UI layer — the actual "backend engine" of the project is the code that loads the latest data, runs it through the saved trained models, combines the outputs, runs SHAP for the explanation, and returns this structured result.

## What the output looks like to the end user

One forecast card per commodity, all following the same visual/data structure so it's reusable for the other asset classes (stocks, crypto, mutual funds) later in the broader advisor project. Each card shows:

- Commodity name/ticker and forecast horizon (e.g. "7-day forecast")
- Current price
- Predicted range, shown as a range bar
- Confidence percentage
- Which model/ensemble was used
- 2-3 top drivers with a +/- indicator showing whether each is pushing price up or down

A full dashboard is six of these cards, one per commodity.

## Key architectural principles to preserve while building

- **Separate models per commodity** — shared pipeline code is fine and encouraged, but never one combined model trained across all six commodities
- **Always output a range + confidence + direction, never a single bare point-price prediction**
- **Every new feature or data source must be benchmarked against the previous best model before being kept** — maintain a running comparison table (model, commodity, MAE, directional accuracy, vs. baseline)
- **Strict chronological train/test splits — never randomly shuffle time series data**
- **No look-ahead bias** — macro data must be aligned to its actual public release date, not the period it describes
- **Tree models (XGBoost/LightGBM) and LSTM need different preprocessing** — trees use the raw/unscaled feature table directly; LSTM needs scaled features reshaped into sequences
- **Explainability is part of the output, not an afterthought** — SHAP-based driver explanations are a required part of the forecast output, not optional polish

## Current status

OHLCV + technical indicator models (XGBoost, LightGBM, LSTM) have been trained. Macro data has been partially collected for the six commodities. Next steps: merge macro data into the existing feature tables (respecting release-date alignment), retrain, compare against the OHLCV-only baseline, then build the ensemble and output layer described above.