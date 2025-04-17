from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint  # Cointegration test

app = Flask(__name__)

# Define the Bollinger Bands Strategy
class BollingerBandsStrategy(Strategy):
    def init(self):
        self.sma = self.I(lambda x: x.rolling(20).mean(), self.data.Close)
        self.std = self.I(lambda x: x.rolling(20).std(), self.data.Close)
        self.upper_band = self.sma + (self.std * 2)
        self.lower_band = self.sma - (self.std * 2)

    def next(self):
        if self.data.Close[-1] < self.lower_band[-1] and not self.position:
            self.buy()
        elif self.data.Close[-1] > self.upper_band[-1] and self.position:
            self.sell()

# Define the Dual Class Arbitrage Strategy
class DualClassArbitrageStrategy(Strategy):
    def init(self):
        self.class_c = self.data.df['ClassC']
        self.price_diff = self.data.Close - self.class_c
        self.sma = self.I(lambda x: x.rolling(50).mean(), self.price_diff)
        self.atr = self.data.df['ATR']
        self.rsi = self.data.df['RSI']
        self.zscore_values = []

    def next(self):
        zscore = (self.price_diff[-1] - self.sma[-1]) / self.atr[-1]
        self.zscore_values.append(zscore)

        if self.rsi[-1] < 90 and zscore > 2.0 and not self.position:
            size = round(self._broker._cash * 0.05 / self.data.Close[-1])
            self.buy(size=size)
        elif self.rsi[-1] > 10 and zscore < -2.0 and self.position:
            self.sell()

        if self.position:
            stop_loss = self.data.Close[-1] - 2 * self.atr[-1]
            take_profit = self.data.Close[-1] + 2 * self.atr[-1]
            if self.data.Close[-1] < stop_loss or self.data.Close[-1] > take_profit:
                self.position.close()

# Define the Sector-Based Pairs Trading Strategy
class SectorPairsTradingStrategy(Strategy):
    stop_loss_pct = 0.02
    take_profit_pct = 0.05

    def init(self):
        self.asset2 = self.data.df['Asset2']
        self.spread = self.data.Close - self.asset2

        self.mean = self.I(lambda x: x.rolling(30).mean(), self.spread)
        self.std = self.I(lambda x: x.rolling(30).std(), self.spread)
        self.zscore = self.I(lambda: (self.spread - self.mean) / self.std)

        self.macd = self.I(self.calculate_macd, self.data.Close)
        self.signal = self.I(self.calculate_signal, self.data.Close)

        self.upper_bb = self.I(lambda x: x.rolling(20).mean() + 2 * x.rolling(20).std(), self.spread)
        self.lower_bb = self.I(lambda x: x.rolling(20).mean() - 2 * x.rolling(20).std(), self.spread)

    def calculate_macd(self, price):
        ema_12 = self.ema(price, 12)
        ema_26 = self.ema(price, 26)
        return ema_12 - ema_26

    def calculate_signal(self, price):
        macd = self.macd
        return self.ema(macd, 9)

    def ema(self, series, span):
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

        if z > 0.7 and macd_value > signal_value and not position:
            self.sell()
            self.buy()

        elif z < -0.7 and macd_value < signal_value and not position:
            risk = 0.3
            capital = self.equity
            expected_return = 0.1
            size = self.calculate_position_size(capital, risk, expected_return)
            self.buy(size=size)
            self.sell(size=size)

        elif abs(z) < 0.6 and position:
            position.close()

        elif position:
            pnl_pct = position.pl / self.equity
            if pnl_pct <= -self.stop_loss_pct:
                position.close()
            elif pnl_pct >= self.take_profit_pct:
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

        if isinstance(df1.columns, pd.MultiIndex):
            df1.columns = df1.columns.get_level_values(1)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.get_level_values(1)

        df = df1[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Asset2'] = df2['Close']
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Download error: {e}")
        return None
    
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

def download_dual_data(symbol_class_a, symbol_class_c, start_date, end_date):
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

    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return None

def run_backtest(data, strategy_class, initial_cash):
    bt = Backtest(data, strategy_class,
                  cash=initial_cash, commission=0.00,
                  exclusive_orders=True)
    results = bt.run()
    bt.plot(resample=False)
    return results

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/run_strategy', methods=['POST'])
def run_strategy():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid or missing JSON payload'}), 400

    try:
        initial_cash = float(data.get('initial_cash', 0))
        strategy_type = data.get('strategy_type')
        start_date = data.get('start_date')
        end_date = data.get('end_date')

        if strategy_type == 'bollinger-bands':
            symbol = data.get('symbol')
            stock_data = download_stock_data(symbol, start_date, end_date)
            results = run_backtest(stock_data, BollingerBandsStrategy, initial_cash)

        elif strategy_type == 'dual-class':
            symbol_class_a = data.get('symbol_class_a')
            symbol_class_c = data.get('symbol_class_c')
            stock_data = download_dual_data(symbol_class_a, symbol_class_c, start_date, end_date)
            results = run_backtest(stock_data, DualClassArbitrageStrategy, initial_cash)

        elif strategy_type == 'pairs':
            symbol1 = data.get('symbol1')
            symbol2 = data.get('symbol2')
            stock_data = download_pair_data(symbol1, symbol2, start_date, end_date)
            results = run_backtest(stock_data, SectorPairsTradingStrategy, initial_cash)

        else:
            return jsonify({'error': 'Unknown strategy type'}), 400

        return jsonify({
            'strategy_type': strategy_type,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': initial_cash,
            'results': results._strategy._equity_curve.to_dict()  # or however you structure your results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == "__main__":
    app.run(debug=True)