import pandas as pd
import datetime as dt

min_date = dt.datetime(2020, 3, 25)
max_date = dt.datetime(2025, 10, 7)

coremarketdata_df = pd.read_csv("coremarketdata.csv", parse_dates=["Date"], index_col="Date")
coremarketdata_df.index = pd.to_datetime(coremarketdata_df.index, format='ISO8601')
coremarketdata_df.sort_index(inplace=True)
coremarketdata_df = coremarketdata_df[(coremarketdata_df.index >= min_date) & (coremarketdata_df.index <= max_date)]

derivatives_df = pd.read_csv("derivatives.csv", parse_dates=["Date"], index_col="Date")
derivatives_df.index = pd.to_datetime(derivatives_df.index, format ='ISO8601')
derivatives_df.sort_index(inplace=True)
derivatives_df = derivatives_df[(derivatives_df.index >= min_date) & (derivatives_df.index <= max_date)]


onchain_df = pd.read_csv("onchainmetrics.csv", parse_dates=["Date"], index_col="Date")
onchain_df.index = pd.to_datetime(onchain_df.index, format ='ISO8601')
onchain_df.sort_index(inplace=True)
onchain_df = onchain_df[(onchain_df.index >= min_date) & (onchain_df.index <= max_date)]



futures_df = pd.read_csv("futures.csv", parse_dates=["timestamp"], index_col="timestamp")
futures_df.sort_index(inplace=True)
futures_df.index = pd.to_datetime(futures_df.index, format ='ISO8601')
futures_df.index.name = "Date"
futures_df["funding_rate"] = futures_df["estimated_sell_volume"] / futures_df["estimated_buy_volume"] - 1
futures_df = futures_df.resample("D").agg({
        'funding_rate': 'mean',             
        'cvd': 'sum',                        
        'cumulative_liquidations': 'sum',  
    }).reset_index()

futures_df = futures_df.rename(columns={'funding_rate': 'Funding Rate', 'cvd': 'CVD', 'cumulative_liquidations': 'Cumulative Liquidations'})
futures_df.index = pd.DatetimeIndex(futures_df["Date"]).normalize()
futures_df.sort_index(inplace=True)
futures_df = futures_df[(futures_df.index >= min_date) & (futures_df.index <= max_date)]
futures_df.set_index("Date", inplace=True)
futures_df.to_csv("futures.csv") #old file was too big for git hub

greeks_df = pd.read_csv("greeks.csv", parse_dates=["date"], index_col="date")
greeks_df.sort_index(inplace=True)
greeks_df.index.name = "Date"
greeks_df.index = pd.to_datetime(greeks_df.index).tz_localize(None).normalize()
greeks_df["Delta"] = greeks_df["delta_call"] + greeks_df["delta_put"]
greeks_df = greeks_df.resample("D").agg({
        'Delta': 'last',
        'gamma': 'last',
        'vega': 'last'
    }).reset_index()

greeks_df.rename(columns={'gamma': 'Gamma', 'vega': 'Vega'}, inplace=True)
greeks_df.index = pd.DatetimeIndex(greeks_df["Date"]).normalize()
greeks_df.sort_index(inplace=True)
greeks_df = greeks_df[(greeks_df.index >= min_date) & (greeks_df.index <= max_date)]
greeks_df.set_index("Date", inplace=True)
greeks_df.to_csv("greeks.csv")

metrics = pd.merge(futures_df, greeks_df, left_on="Date", right_on="Date", how="inner")

metrics["Close"] = coremarketdata_df["Close"]
metrics["Open"] = coremarketdata_df["Open"]
metrics["High"] = coremarketdata_df["High"]
metrics["Low"] = coremarketdata_df["Low"]
metrics["Volume"] = coremarketdata_df["Volume"]
metrics["Daily Return"] = coremarketdata_df["Daily Return"]
metrics["Volatility"] = coremarketdata_df["Volatility"]
metrics["SMA50"] = coremarketdata_df["SMA50"]
metrics["SMA200"] = coremarketdata_df["SMA200"]
metrics["RSI"] = coremarketdata_df["RSI"]
metrics["EMA20"] = coremarketdata_df["EMA20"]
metrics["MACD"] = coremarketdata_df["MACD"]
metrics["Bollinger Band Width"] = coremarketdata_df["Bollinger Band Width"]
metrics["OBV"] = coremarketdata_df["OBV"]
metrics["ATR"] = coremarketdata_df["ATR"]
metrics["Stochastic Oscillator"] = coremarketdata_df["Stochastic oscillator"]

metrics["Activate Addresses"] = onchain_df["Active Addresses"]
metrics["Hash Rate"] = onchain_df["Hash Rate"]
metrics["Circulating Supply"] = onchain_df["Circulating Supply"]
metrics["Transaction Volume"] = onchain_df["Transaction Volume"]
metrics["Exchange Inflow"] = onchain_df["Exchange Inflow"]
metrics["Exchange Outflow"] = onchain_df["Exchange Outflow"]

metrics["Open Interest"] = derivatives_df["Open Interest"]
metrics["Bid-Ask Spread"] = derivatives_df["Bid-Ask Spread"]
metrics["Depth at 1%"] = derivatives_df["Depth at 1%"]
metrics["Put-Call Ratio"] = derivatives_df["Put-Call Ratio"]

epsilon = 1e-3
metrics["Funding Rate"] = (metrics["Funding Rate"] - metrics["Funding Rate"].rolling(window = 30).mean()) / (metrics["Funding Rate"].rolling(window = 30).std() + epsilon)
metrics["Funding Rate"] = metrics["Funding Rate"].clip(metrics["Funding Rate"].quantile(0.1), metrics["Funding Rate"].quantile(0.9))
metrics["CVD"] = (metrics["CVD"] - metrics["CVD"].rolling(window = 30).mean()) / (metrics["CVD"].rolling(window = 30).std() + epsilon)
metrics["CVD"] = metrics["CVD"].clip(metrics["CVD"].quantile(0.01), metrics["CVD"].quantile(0.99))
metrics["Cumulative Liquidations"] = (metrics["Cumulative Liquidations"] - metrics["Cumulative Liquidations"].rolling(window = 30).mean()) / (metrics["Cumulative Liquidations"].rolling(window = 30).std() + epsilon)
metrics["Cumulative Liquidations"] = metrics["Cumulative Liquidations"].clip(metrics["Cumulative Liquidations"].quantile(0.01), metrics["Cumulative Liquidations"].quantile(0.99))
metrics["Close"] = (metrics["Close"] - metrics["Close"].rolling(window = 30).mean()) / (metrics["Close"].rolling(window = 30).std() + epsilon)
metrics["Close"] = metrics["Close"].clip(metrics["Close"].quantile(0.01), metrics["Close"].quantile(0.99))
metrics["Open"] = (metrics["Open"] - metrics["Open"].rolling(window = 30).mean()) / (metrics["Open"].rolling(window = 30).std() + epsilon)
metrics["Open"] = metrics["Open"].clip(metrics["Open"].quantile(0.01), metrics["Open"].quantile(0.99))
metrics["High"] = (metrics["High"] - metrics["High"].rolling(window = 30).mean()) / (metrics["High"].rolling(window = 30).std() + epsilon)
metrics["High"] = metrics["High"].clip(metrics["High"].quantile(0.01), metrics["High"].quantile(0.99))
metrics["Low"] = (metrics["Low"] - metrics["Low"].rolling(window = 30).mean()) / (metrics["Low"].rolling(window = 30).std() + epsilon)
metrics["Low"] = metrics["Low"].clip(metrics["Low"].quantile(0.01), metrics["Low"].quantile(0.99))
metrics["Volume"] = (metrics["Volume"] - metrics["Volume"].rolling(window = 30).mean()) / (metrics["Volume"].rolling(window = 30).std() + epsilon)
metrics["Volume"] = metrics["Volume"].clip(metrics["Volume"].quantile(0.01), metrics["Volume"].quantile(0.99))
metrics["Daily Return"] = (metrics["Daily Return"] - metrics["Daily Return"].rolling(window = 30).mean()) / (metrics["Daily Return"].rolling(window = 30).std() + epsilon)
metrics["Daily Return"] = metrics["Daily Return"].clip(metrics["Daily Return"].quantile(0.01), metrics["Daily Return"].quantile(0.99))
metrics["Volatility"] = (metrics["Volatility"] - metrics["Volatility"].rolling(window = 30).mean()) / (metrics["Volatility"].rolling(window = 30).std() + epsilon)
metrics["Volatility"] = metrics["Volatility"].clip(metrics["Volatility"].quantile(0.01), metrics["Volatility"].quantile(0.99))
metrics["SMA50"] = (metrics["SMA50"] - metrics["SMA50"].rolling(window = 30).mean()) / (metrics["SMA50"].rolling(window = 30).std() + epsilon)
metrics["SMA50"] = metrics["SMA50"].clip(metrics["SMA50"].quantile(0.01), metrics["SMA50"].quantile(0.99))
metrics["SMA200"] = (metrics["SMA200"] - metrics["SMA200"].rolling(window = 30).mean()) / (metrics["SMA200"].rolling(window = 30).std() + epsilon)
metrics["SMA200"] = metrics["SMA200"].clip(metrics["SMA200"].quantile(0.01), metrics["SMA200"].quantile(0.99))
metrics["RSI"] = (metrics["RSI"] - metrics["RSI"].rolling(window = 30).mean()) / (metrics["RSI"].rolling(window = 30).std() + epsilon)
metrics["RSI"] = metrics["RSI"].clip(metrics["RSI"].quantile(0.01), metrics["RSI"].quantile(0.99))
metrics["EMA20"] = (metrics["EMA20"] - metrics["EMA20"].rolling(window = 30).mean()) / (metrics["EMA20"].rolling(window = 30).std() + epsilon)
metrics["EMA20"] = metrics["EMA20"].clip(metrics["EMA20"].quantile(0.01), metrics["EMA20"].quantile(0.99))
metrics["MACD"] = (metrics["MACD"] - metrics["MACD"].rolling(window = 30).mean()) / (metrics["MACD"].rolling(window = 30).std() + epsilon)
metrics["MACD"] = metrics["MACD"].clip(metrics["MACD"].quantile(0.01), metrics["MACD"].quantile(0.99))
metrics["Bollinger Band Width"] = (metrics["Bollinger Band Width"] - metrics["Bollinger Band Width"].rolling(window = 30).mean()) / (metrics["Bollinger Band Width"].rolling(window = 30).std() + epsilon)
metrics["Bollinger Band Width"] = metrics["Bollinger Band Width"].clip(metrics["Bollinger Band Width"].quantile(0.01), metrics["Bollinger Band Width"].quantile(0.99))
metrics["OBV"] = (metrics["OBV"] - metrics["OBV"].rolling(window = 30).mean()) / (metrics["OBV"].rolling(window = 30).std() + epsilon)
metrics["OBV"] = metrics["OBV"].clip(metrics["OBV"].quantile(0.01), metrics["OBV"].quantile(0.99))
metrics["ATR"] = (metrics["ATR"] - metrics["ATR"].rolling(window = 30).mean()) / (metrics["ATR"].rolling(window = 30).std() + epsilon)
metrics["ATR"] = metrics["ATR"].clip(metrics["ATR"].quantile(0.01), metrics["ATR"].quantile(0.99))
metrics["Stochastic Oscillator"] = (metrics["Stochastic Oscillator"] - metrics["Stochastic Oscillator"].rolling(window = 30).mean()) / (metrics["Stochastic Oscillator"].rolling(window = 30).std() + epsilon)
metrics["Stochastic Oscillator"] = metrics["Stochastic Oscillator"].clip(metrics["Stochastic Oscillator"].quantile(0.01), metrics["Stochastic Oscillator"].quantile(0.99))
metrics["Activate Addresses"] = (metrics["Activate Addresses"] - metrics["Activate Addresses"].rolling(window = 30).mean()) / (metrics["Activate Addresses"].rolling(window = 30).std() + epsilon)
metrics["Activate Addresses"] = metrics["Activate Addresses"].clip(metrics["Activate Addresses"].quantile(0.01), metrics["Activate Addresses"].quantile(0.99))
metrics["Hash Rate"] = (metrics["Hash Rate"] - metrics["Hash Rate"].rolling(window = 30).mean()) / (metrics["Hash Rate"].rolling(window = 30).std() + epsilon)
metrics["Hash Rate"] = metrics["Hash Rate"].clip(metrics["Hash Rate"].quantile(0.01), metrics["Hash Rate"].quantile(0.99))
metrics["Circulating Supply"] = (metrics["Circulating Supply"] - metrics["Circulating Supply"].rolling(window = 30).mean()) / (metrics["Circulating Supply"].rolling(window = 30).std() + epsilon)
metrics["Circulating Supply"] = metrics["Circulating Supply"].clip(metrics["Circulating Supply"].quantile(0.01), metrics["Circulating Supply"].quantile(0.99))
metrics["Transaction Volume"] = (metrics["Transaction Volume"] - metrics["Transaction Volume"].rolling(window = 30).mean()) / (metrics["Transaction Volume"].rolling(window = 30).std() + epsilon)
metrics["Transaction Volume"] = metrics["Transaction Volume"].clip(metrics["Transaction Volume"].quantile(0.01), metrics["Transaction Volume"].quantile(0.99))
metrics["Exchange Inflow"] = (metrics["Exchange Inflow"] - metrics["Exchange Inflow"].rolling(window = 30).mean()) / (metrics["Exchange Inflow"].rolling(window = 30).std() + epsilon)
metrics["Exchange Inflow"] = metrics["Exchange Inflow"].clip(metrics["Exchange Inflow"].quantile(0.01), metrics["Exchange Inflow"].quantile(0.99))
metrics["Exchange Outflow"] = (metrics["Exchange Outflow"] - metrics["Exchange Outflow"].rolling(window = 30).mean()) / (metrics["Exchange Outflow"].rolling(window = 30).std() + epsilon)
metrics["Exchange Outflow"] = metrics["Exchange Outflow"].clip(metrics["Exchange Outflow"].quantile(0.01), metrics["Exchange Outflow"].quantile(0.99))
metrics["Open Interest"] = (metrics["Open Interest"] - metrics["Open Interest"].rolling(window = 30).mean()) / (metrics["Open Interest"].rolling(window = 30).std() + epsilon)
metrics["Open Interest"] = metrics["Open Interest"].clip(metrics["Open Interest"].quantile(0.01), metrics["Open Interest"].quantile(0.99))
metrics["Bid-Ask Spread"] = (metrics["Bid-Ask Spread"] - metrics["Bid-Ask Spread"].rolling(window = 30).mean()) / (metrics["Bid-Ask Spread"].rolling(window = 30).std() + epsilon)
metrics["Bid-Ask Spread"] = metrics["Bid-Ask Spread"].clip(metrics["Bid-Ask Spread"].quantile(0.01), metrics["Bid-Ask Spread"].quantile(0.99))
metrics["Depth at 1%"] = (metrics["Depth at 1%"] - metrics["Depth at 1%"].rolling(window = 30).mean()) / (metrics["Depth at 1%"].rolling(window = 30).std() + epsilon)
metrics["Depth at 1%"] = metrics["Depth at 1%"].clip(metrics["Depth at 1%"].quantile(0.01), metrics["Depth at 1%"].quantile(0.99))
metrics["Put-Call Ratio"] = (metrics["Put-Call Ratio"] - metrics["Put-Call Ratio"].rolling(window = 30).mean()) / (metrics["Put-Call Ratio"].rolling(window = 30).std() + epsilon)
metrics["Put-Call Ratio"] = metrics["Put-Call Ratio"].clip(metrics["Put-Call Ratio"].quantile(0.01), metrics["Put-Call Ratio"].quantile(0.99))
metrics["Delta"] = (metrics["Delta"] - metrics["Delta"].rolling(window = 30).mean()) / (metrics["Delta"].rolling(window = 30).std() + epsilon)
metrics["Delta"] = metrics["Delta"].clip(metrics["Delta"].quantile(0.01), metrics["Delta"].quantile(0.99))
metrics["Gamma"] = (metrics["Gamma"] - metrics["Gamma"].rolling(window = 30).mean()) / (metrics["Gamma"].rolling(window = 30).std() + epsilon)
metrics["Gamma"] = metrics["Gamma"].clip(metrics["Gamma"].quantile(0.01), metrics["Gamma"].quantile(0.99))
metrics["Vega"] = (metrics["Vega"] - metrics["Vega"].rolling(window = 30).mean()) / (metrics["Vega"].rolling(window = 30).std() + epsilon)
metrics["Vega"] = metrics["Vega"].clip(metrics["Vega"].quantile(0.01), metrics["Vega"].quantile(0.99))

metrics.dropna(inplace=True)

with open("metrics.csv", "w") as f:
    metrics.to_csv(f)