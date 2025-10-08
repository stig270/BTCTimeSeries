import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

symbol = "BTC-USD"
df = yf.download(symbol, start="2008-01-01", end=dt.datetime.now().strftime("%Y-%m-%d"), interval="1d")

df["Daily Return"] = (df["Close"] - df["Open"]) / df["Open"]
df["Volatility"] = np.log(df["Close"] / df["Open"]).rolling(window=7).std()
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
df["RSI"] = 1 - (1 / (1 + df["Daily Return"].rolling(window=14).apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if -x[x < 0].sum() != 0 else 0)))
df["Bollinger Band Width"] = (df["Close"].rolling(window=20).std() * 4) / df["Close"].rolling(window=20).mean()
df["OBV"] = (np.sign(df["Close"] - df["Open"]) * df["Volume"]).fillna(0).cumsum()
df["ATR"] = (df["High"] - df["Low"]).rolling(window=14).mean()
df["Stochastic oscillator"] = ((df["Close"] - df["Low"].rolling(window=14).min()) / (df["High"].rolling(window=14).max() - df["Low"].rolling(window=14).min()))

with open("coremarketdata.csv", "w") as f:
    df.to_csv(f)
