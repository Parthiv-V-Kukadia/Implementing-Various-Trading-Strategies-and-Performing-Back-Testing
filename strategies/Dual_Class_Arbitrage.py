import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy


class DualClassArbitrageStrategy(Strategy):
    def init(self):
        # Store the custom Class C prices passed in data
        self.class_c = self.data.df['ClassC']
        self.price_diff = self.data.Close - self.class_c
        self.sma = self.I(lambda x: x.rolling(50).mean(), self.price_diff)
        self.atr = self.data.df['ATR']  # ATR precomputed in the data
        self.rsi = self.data.df['RSI']  # RSI precomputed in the data
        self.zscore_values = []  # Store z-scores for analysis

    def next(self):
        # print(f"Price Diff: {self.price_diff[-1]}, SMA: {self.sma[-1]}, ATR: {self.atr[-1]}, RSI: {self.rsi[-1]}")

        zscore = (self.price_diff[-1] - self.sma[-1]) / self.atr[-1]
        self.zscore_values.append(zscore)  # Store z-score
        # print(f"Z-score: {zscore}")  # Debugging z-score
        # print(f"Current Close: {self.data.Close[-1]}")

        if self.rsi[-1] < 90 and zscore > 2.0 and not self.position:  # Changed z-score threshold to 2
            size = round(self._broker._cash * 0.05 / self.data.Close[-1])
            print(
                f"BUY SIGNAL: Z-score: {zscore:.2f}, RSI: {self.rsi[-1]:.2f}, Size: {size}, Price: {self.data.Close[-1]:.2f}")  # Detailed log
            self.buy(size=size)
        elif self.rsi[-1] > 10 and zscore < -2.0 and self.position:  # Changed z-score threshold to -2
            print(f"SELL SIGNAL: Z-score: {zscore:.2f}, RSI: {self.rsi[-1]:.2f}, Price: {self.data.Close[-1]:.2f}")
            self.sell()

        # Implement Stop-Loss and Take-Profit
        if self.position:
            stop_loss = self.data.Close[-1] - 2 * self.atr[-1]
            take_profit = self.data.Close[-1] + 2 * self.atr[-1]
            # print(f"Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")  # Debugging SL/TP
            if self.data.Close[-1] < stop_loss or self.data.Close[-1] > take_profit:
                print(f"Closing position due to SL/TP at price: {self.data.Close[-1]:.2f}")
                self.position.close()


def download_stock_data(symbol_class_a, symbol_class_c, start_date, end_date):
    try:
        df_class_a = yf.download(symbol_class_a, start=start_date, end=end_date, group_by='ticker')
        df_class_c = yf.download(symbol_class_c, start=start_date, end=end_date, group_by='ticker')

        if df_class_a.empty or df_class_c.empty:
            print(f"Missing data for {symbol_class_a} or {symbol_class_c}")
            return None

        # Handle MultiIndex columns by flattening
        if isinstance(df_class_a.columns, pd.MultiIndex):
            df_class_a.columns = df_class_a.columns.get_level_values(1)
        if isinstance(df_class_c.columns, pd.MultiIndex):
            df_class_c.columns = df_class_c.columns.get_level_values(1)

        # Make sure required columns are present
        required_columns_a = ['Open', 'High', 'Low', 'Close', 'Volume']
        required_columns_c = ['Close']
        if not all(col in df_class_a.columns for col in required_columns_a) or not all(
                col in df_class_c.columns for col in required_columns_c):
            print(f"Missing expected columns in downloaded data.")
            print("Available columns in Class A:", df_class_a.columns)
            print("Available columns in Class C:", df_class_c.columns)
            return None

        # Combine into one DataFrame, aligning on index (Date)
        df = df_class_a[required_columns_a].copy()
        df['ClassC'] = df_class_c['Close']

        # Calculate ATR (Average True Range)
        df['TR'] = np.maximum(df['High'] - df['Low'],
                           np.maximum(np.abs(df['High'] - df['Close'].shift()),
                                      np.abs(df['Low'] - df['Close'].shift())))
        df['ATR'] = df['TR'].rolling(14).mean()  # ATR calculation

        # Correct RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Download error: {e}")
        return None


def run_backtest(data, initial_cash):
    bt = Backtest(data, DualClassArbitrageStrategy,
                  cash=initial_cash, commission=0.00,
                  exclusive_orders=True)
    results = bt.run()
    print(results)
    bt.plot(resample=False)  # Keep resample=False for higher fidelity
    # Plot the z-scores
    plt.figure(figsize=(10, 6))
    plt.plot(bt._strategy.zscore_values, label='Z-score')  # Access from the instance
    plt.axhline(2, color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(-2, color='g', linestyle='--', label='Lower Threshold')
    plt.xlabel('Time')
    plt.ylabel('Z-score')
    plt.title('Z-score over Time')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # User input
    initial_cash = float(input("Enter initial capital (e.g., 10000): "))
    symbol_class_a = input("Enter the symbol for Class A share (e.g., GOOGL): ")
    symbol_class_c = input("Enter the symbol for Class C share (e.g., GOOG): ")
    start_date = input("Enter the start date (e.g.,YYYY-MM-DD): ")
    end_date = input("Enter the end date (e.g.,YYYY-MM-DD): ")

    data = download_stock_data(symbol_class_a, symbol_class_c, start_date, end_date)

    if data is not None:
        run_backtest(data, initial_cash)
    else:
        print("Failed to retrieve and prepare data.")