# Linear_Regression_trade_analysis
C:\Users\JEKUGA BRANDON\Desktop\trading_back_test\back_test.ipynb
Analyzing trade and following trade direction for its ups/downs in order to enter the market
Compact README (one-page summary for repo/post)
Project: Linear-Regression Market Direction Predictor

What this is

A lightweight strategy that uses simple linear regression on lagged returns to predict the next-period market direction.
Focus is on interpretability, reproducibility, and a robust backtest rather than model complexity.
Files

linear_regression_stock.ipynb — Notebook with data loading, feature (lags) creation, model training, and backtest steps.
LRVectorBacktester.py — Backtesting utility used to simulate trades and compute P&L.
backtest.h5 — Saved backtest data (signals, trades, results).
How it works (contract)

Input: historical price series (OHLC or close).
Features: a set of lagged returns (e.g., t-1, t-2, ...).
Model: ordinary least squares linear regression predicting next-period return or direction.
Output: binary directional signal used by the backtester to simulate long/short trades.
Validation: historical backtest with timezone-aware handling and fixed random seeds for reproducibility.
Usage

Open linear_regression_stock.ipynb.
Run cells top-to-bottom (ensure dependencies installed: numpy, pandas, matplotlib, seaborn, statsmodels, yfinance).
Inspect the backtest output and saved backtest.h5 for trade-level results.
