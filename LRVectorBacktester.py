import numpy as np
import pandas as pd
from scipy.optimize import minimize

class LRVectorBacktester(object):
    def __init__(self, symbol, start, end, amount, tc):
        """
        Parameters
        ----------
        symbol: str
            ticker symbol
        start: str
            start date for data retrieval
        end: str
            end date for data retrieval
        amount: float
            initial amount to be invested per trade
        tc: float
            proportional transaction costs (e.g., 0.0 for none, 0.001 for 0.1%)
        """
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data()

    def get_data(self):
        """Retrieves and prepares the data."""
        raw = pd.read_hdf('backtest.h5', 'forex')
        raw = raw[self.symbol].to_frame('price')
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    def prepare_features(self, start=None, end=None, lags=None):
        """Prepares the feature vectors for prediction."""
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        if lags is None:
            lags = 5
        
        df = pd.DataFrame(self.data.loc[start:end])
        self.feature_columns = []
        for lag in range(1, lags + 1):
            col = f'lag_{lag}'
            df[col] = df['returns'].shift(lag)
            self.feature_columns.append(col)
        self.feature_data = df.dropna()

    def fit_model(self, start=None, end=None):
        """Implements the regression step."""
        self.coeffs = np.zeros(len(self.feature_columns))
        if len(self.feature_data) > 0:
            reg = np.linalg.lstsq(
                self.feature_data[self.feature_columns],
                self.feature_data['returns'],
                rcond=None)[0]
            self.coeffs = reg

    def run_strategy(self, start=None, end=None, lags=None):
        """Backtests the trading strategy."""
        self.prepare_features(start=start, end=end, lags=lags)
        self.fit_model(start=start, end=end)
        
        # make predictions and calculate positions
        self.feature_data['prediction'] = np.dot(
            self.feature_data[self.feature_columns], self.coeffs)
        self.feature_data['position'] = np.sign(
            self.feature_data['prediction'])
        
        # calculate strategy returns
        self.feature_data['strategy'] = (
            self.feature_data['position'].shift(1) *
            self.feature_data['returns'])
        
        # determine when a trading position changes
        trades = self.feature_data['position'].diff().fillna(0) != 0
        
        # subtract transaction costs from return when position changes
        self.feature_data['strategy'][trades] -= self.tc
        
        self.results = self.feature_data
        
        # calculate cumulative returns for strategy
        self.results['creturns'] = self.amount * \
            np.exp(self.results['returns'].cumsum())
        
        # calculate cumulative returns for strategy
        self.results['cstrategy'] = self.amount * \
            np.exp(self.results['strategy'].cumsum())
        
        return round(self.results['cstrategy'].iloc[-1], 2)

    def plot_results(self):
        """Plots the cumulative performance of the trading strategy compared to the buy and hold strategy."""
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
            return
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))