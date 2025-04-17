import pandas as pd
import numpy as np
import yfinance as yf  # For fetching financial data
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy


class BollingerBandStrategy(Strategy):
    """
    Bollinger Band trading strategy.

    Buy when the price crosses below the lower band, and sell when it crosses above the upper band.
    """

    def init(self):
        super().init()
        self.period = 20  # Lookback period for moving average
        self.std_dev = 2  # Standard deviation multiplier

        # Precompute Bollinger Bands
        self.close_prices = self.data.Close
        self.sma = np.zeros_like(self.close_prices)
        self.std = np.zeros_like(self.close_prices)

        for i in range(self.period, len(self.close_prices)):
            self.sma[i] = np.mean(self.close_prices[i - self.period:i])
            self.std[i] = np.std(self.close_prices[i - self.period:i])

        self.upper_band = self.sma + (self.std * self.std_dev)
        self.lower_band = self.sma - (self.std * self.std_dev)

        # Generate signals
        self.signal = np.zeros(len(self.close_prices))
        self.signal[self.close_prices < self.lower_band] = 1   # Buy
        self.signal[self.close_prices > self.upper_band] = -1  # Sell

        self.sl = 0.02  # 2% trailing stop-loss
        self.trailing_sl_active = False
        self.trade_prices = []
        self.bar_index = 0  # Track bar index

    def next(self):
        if self.bar_index > self.period:
            # Entry signals
            if self.signal[self.bar_index] == 1 and not self.position:
                self.buy()
                self.trailing_sl_active = True
                self.trade_prices.append(self.data.Close[self.bar_index])

            elif self.signal[self.bar_index] == -1 and self.position:
                self.sell()
                self.trailing_sl_active = False
                self.trade_prices.append(self.data.Close[self.bar_index])

        # Trailing Stop-Loss Management
        for trade in self.trades:
            if self.trailing_sl_active:
                if trade.is_long and self.data.Low[self.bar_index] < trade.entry_price * (1 - self.sl):
                    trade.close()
                    self.trailing_sl_active = False
                elif trade.is_short and self.data.High[self.bar_index] > trade.entry_price * (1 + self.sl):
                    trade.close()
                    self.trailing_sl_active = False

        # Dynamic SL Adjustment
        for trade in self.trades:
            if trade.is_long:
                new_sl = self.data.Close[self.bar_index] * (1 - self.sl)
                if trade.sl is None or new_sl > trade.sl:
                    trade.sl = new_sl
            elif trade.is_short:
                new_sl = self.data.Close[self.bar_index] * (1 + self.sl)
                if trade.sl is None or new_sl < trade.sl:
                    trade.sl = new_sl

        self.bar_index += 1


def download_stock_data(symbol, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"No data found for symbol {symbol} between {start_date} and {end_date}.")
            return None

        # Handle possible MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns:
                df = df['Adj Close'].to_frame()
            else:
                available_cols = df.columns.levels[0].intersection(
                    ['Open', 'High', 'Low', 'Close', 'Volume'])
                df = df[available_cols]
                df.columns = available_cols

        return df

    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None


def run_backtest(data):
    """
    Runs the backtest for the Bollinger Band strategy.
    """
    bt = Backtest(data, BollingerBandStrategy,
                  cash=10000, commission=0.002,
                  exclusive_orders=True)
    results = bt.run()
    print(results)
    bt.plot(resample=False)
    return bt


if __name__ == "__main__":
    # Prompt user for input
    symbol = input("Enter the symbol (e.g., AAPL, NVDA): ")
    start_date = input("Enter the start date (e.g., YYYY-MM-DD; 2020-01-01): ")
    end_date = input("Enter the end date (e.g., YYYY-MM-DD; 2024-12-12): ")

    data = download_stock_data(symbol, start_date, end_date)

    if data is None:
        print("Failed to retrieve data. Exiting.")
        exit()

    run_backtest(data)