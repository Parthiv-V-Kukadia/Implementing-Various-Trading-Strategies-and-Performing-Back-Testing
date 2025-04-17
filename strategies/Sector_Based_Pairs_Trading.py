import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from statsmodels.tsa.stattools import coint  # Cointegration test

class SectorPairsTradingStrategy(Strategy):
    stop_loss_pct = 0.02     # 2% stop-loss
    take_profit_pct = 0.05   # 5% take-profit

    def init(self):
        self.asset2 = self.data.df['Asset2']
        self.spread = self.data.Close - self.asset2

        # Z-score for spread
        self.mean = self.I(lambda x: x.rolling(30).mean(), self.spread)
        self.std = self.I(lambda x: x.rolling(30).std(), self.spread)
        self.zscore = self.I(lambda: (self.spread - self.mean) / self.std)

        # MACD and Signal - Calculating outside the `self.I` method
        self.macd = self.I(self.calculate_macd, self.data.Close)
        self.signal = self.I(self.calculate_signal, self.data.Close)

        # Bollinger Bands (based on spread)
        self.upper_bb = self.I(lambda x: x.rolling(20).mean() + 2 * x.rolling(20).std(), self.spread)
        self.lower_bb = self.I(lambda x: x.rolling(20).mean() - 2 * x.rolling(20).std(), self.spread)

    def calculate_macd(self, price):
        # 12-period and 26-period EMAs
        ema_12 = self.ema(price, 12)
        ema_26 = self.ema(price, 26)
        return ema_12 - ema_26

    def calculate_signal(self, price):
        # Signal is 9-period EMA of the MACD
        macd = self.macd
        return self.ema(macd, 9)

    def ema(self, series, span):
        # Exponentially weighted moving average (manual calculation)
        alpha = 2 / (span + 1)
        result = np.zeros_like(series)
        result[0] = series[0]
        for i in range(1, len(series)):
            result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
        return result

    def next(self):
        z = self.zscore[-1]
        macd_value = self.macd[-1]
        signal_value = self.signal[-1]

        current_price = self.data.Close[-1]
        position = self.position
        print(f"Z-score: {z:.2f}, MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, PnL: {position.pl:.2f}" if position else f"Z-score: {z:.2f}, MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, No open position")

        # Entry Logic
        if z > 0.7 and macd_value > signal_value and not position:
            print("Z > 0.7 and MACD > Signal: Selling asset1, buying asset2")
            self.sell()
            self.buy()

        elif z < -0.7 and macd_value < signal_value and not position:
            print("Z < -0.7 and MACD < Signal: Buying asset1, selling asset2")
            risk = 0.3
            capital = self.equity
            expected_return = 0.1
            size = self.calculate_position_size(capital, risk, expected_return)
            self.buy(size=size)
            self.sell(size=size)

        # Exit Logic
        elif abs(z) < 0.6 and position:
            print("Mean-reversion: Closing both positions")
            position.close()

        # Stop-loss and Take-profit conditions
        elif position:
            pnl_pct = position.pl / self.equity
            if pnl_pct <= -self.stop_loss_pct:
                print("Stop-loss hit: Closing position")
                position.close()
            elif pnl_pct >= self.take_profit_pct:
                print("Take-profit hit: Closing position")
                position.close()

    def calculate_position_size(self, capital, risk_percent, expected_return):
        kelly_fraction = expected_return / (capital * risk_percent)
        return max(0.01, kelly_fraction)


def download_pair_data(symbol1, symbol2, start_date, end_date):
    try:
        df1 = yf.download(symbol1, start=start_date, end=end_date, group_by='ticker')
        df2 = yf.download(symbol2, start=start_date, end=end_date, group_by='ticker')

        if df1.empty or df2.empty:
            print(f"Missing data for {symbol1} or {symbol2}")
            return None

        # Flatten MultiIndex if present
        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.get_level_values(1)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(1)

        # Only retain standard OHLCV data from stock 1
        df = df1[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Asset2'] = df2['Close']
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Download error: {e}")
        return None

def run_backtest(data):
    bt = Backtest(data, SectorPairsTradingStrategy,
                  cash=initial_cash, commission=0.00,
                  exclusive_orders=True)
    results = bt.run()
    print(results)
    bt.plot(resample=False)

def test_cointegration(asset1, asset2):
    """
    Test the cointegration of two assets using the Engle-Granger method.
    Returns the p-value of the cointegration test.
    """
    score, p_value, _ = coint(asset1, asset2)
    return p_value

if __name__ == "__main__":
    print("Sector-Based Pairs Trading Backtest")
    initial_cash = float(input("Enter initial capital (e.g., 10000): "))
    symbol1 = input("Enter the first stock symbol (e.g., AAPL): ")
    symbol2 = input("Enter the second stock symbol (e.g., MSFT): ")
    start_date = input("Enter the start date (e.g., YYYY-MM-DD): ")
    end_date = input("Enter the end date (e.g., YYYY-MM-DD): ")

    # Fix Yahoo-style symbols
    def fix_symbol(s): return s.replace('.', '-')
    symbol1 = fix_symbol(symbol1)
    symbol2 = fix_symbol(symbol2)

    data = download_pair_data(symbol1, symbol2, start_date, end_date)

    # Perform cointegration test
    if data is not None:
        # Bypass the cointegration check and proceed with the backtest
        print("Cointegration check bypassed. Proceeding with the backtest.")
        run_backtest(data)
    else:
        print("Failed to retrieve and prepare data.")