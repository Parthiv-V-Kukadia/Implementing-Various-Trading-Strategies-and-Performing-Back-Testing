# Quantitative Trading Strategies with Python

This repository contains a collection of algorithmic trading strategies developed using Python, `backtesting.py`, and `yfinance`. Each strategy is designed to leverage different market inefficiencies, technical indicators, and statistical techniques.

## Strategies Implemented
### 1. **Bollinger Band Strategy with Trailing Stop Loss**
A classic mean-reversion strategy utilizing Bollinger Bands.

- **Buy**: When price crosses below the lower band.
- **Sell**: When price crosses above the upper band.
- **Features**:
  - 20-period moving average
  - 2 standard deviations for band width
  - Trailing stop-loss logic (2% trail)
- **Backtest Example**:
  ```bash
  python Bollinger_Band_Strategy.py
  ```

### 2. **Dual-Class Arbitrage Strategy**
Designed for arbitraging the spread between Class A and Class C shares of the same company (e.g., GOOGL vs GOOG).

- **Indicators Used**:
  - Price difference and rolling mean
  - Average True Range (ATR)
  - Relative Strength Index (RSI)
  - Z-score of the spread
- **Logic**:
  - Buy when Z-score > 2 and RSI < 90
  - Sell when Z-score < -2 and RSI > 10
  - Includes stop-loss and take-profit based on ATR
- **Backtest Example**:
  ```bash
  python Dual_Class_Arbitrage.py
  ```

### 3. **Sector Pairs Trading Strategy** *(Work in Progress)*
Mean-reversion strategy applied to two co-integrated stocks from the same sector.

- **Key Concepts**:
  - Statistical arbitrage using spread between two stocks
  - Cointegration test (Engle-Granger)
  - Z-score of spread
  - Bollinger Bands on spread
  - MACD and signal line
- **Planned Features**:
  - Entry on Z-score extremes + MACD divergence
  - Exit using Bollinger Bands or fixed risk parameters
  - Stop-loss: 2%
  - Take-profit: 5%
- **Backtest Example**:
  ```bash
  python Sector_Based_Pairs_Trading.py
  ```


## Technologies Used

- Python
- `yfinance` – for historical stock data
- `backtesting.py` – for strategy simulation
- `matplotlib` – for data visualization
- `numpy`, `pandas` – for computation and data wrangling
- `statsmodels` – for cointegration test


## How to Run
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a strategy**:
   ```bash
   python <strategy_file>.py
   ```

3. **User Input**:
   You’ll be prompted to enter:
   - Initial capital (for strategies involving sizing)
   - Stock ticker(s)
   - Date range (e.g., 2020-01-01 to 2024-12-31)


## Output

- Backtest summary with key metrics (return, Sharpe ratio, etc.)
- Trade logs (where applicable)
- Interactive performance plots
- Additional indicator plots (e.g., Z-scores)


## File Structure

```
.
├── Bollinger_Band_Strategy.py           # Bollinger Band with Trailing SL
├── Dual_Class_Arbitrage.py              # Class A vs Class C Arbitrage Strategy
├── Sector_Based_Pairs_Trading.py        # Sector-based pairs trading (WIP)
├── README.md                            # You're here!
└── requirements.txt                     # Python dependencies
```

## Notes
- Ensure the tickers you use exist on Yahoo Finance.
- Performance will vary depending on data quality and market conditions.
- Always validate strategies on out-of-sample data.
