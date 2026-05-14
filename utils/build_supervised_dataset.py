import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

etf_list = ['QQQ', 'GLD', 'IWM', 'XLE', 'XLF', 'TLT', 'SPY']
etfs = {}

for etf in etf_list:
    df = pd.read_csv(f"../data/{etf}_with_regime.csv", index_col=0, parse_dates=True)
    df.rename(columns={etf: 'Close'}, inplace=True)  # standardize
    df.dropna(inplace=True)

    # Basic Features
    df[f'{etf}_MACD'] = MACD(df['Close']).macd()
    df[f'{etf}_RSI'] = RSIIndicator(df['Close']).rsi()
    df[f'{etf}_Bollinger_Width'] = BollingerBands(df['Close']).bollinger_wband()
    df[f'{etf}_Sharpe'] = (df['Close'].pct_change().mean() / df['Close'].pct_change().std()) * np.sqrt(252)

    etfs[etf] = df[[f'{etf}_MACD', f'{etf}_RSI', f'{etf}_Bollinger_Width', f'{etf}_Sharpe', 'Regime']]

# from functools import reduce

# # Start with QQQ
# master_df = etfs['QQQ'].copy()

# # Merge others
# for etf in etf_list:
#     if etf == 'QQQ':
#         continue
#     master_df = master_df.join(etfs[etf], how='inner')

# # Add QQQ future return direction as target (binary)
# master_df['QQQ_Return'] = master_df['QQQ_MACD'].shift(-1)  # future MACD direction
# master_df['Target'] = (master_df['QQQ_Return'] > 0).astype(int)
# master_df.dropna(inplace=True)

# master_df.to_csv("../data/qqq_supervised.csv")
# print("✅ Saved: qqq_supervised.csv with cross-ETF features")
