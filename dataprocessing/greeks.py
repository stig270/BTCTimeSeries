import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import requests
from scipy.stats import norm
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class BybitDataDownloaderOptimized:
    
    
    def __init__(self, symbol='BTCUSDT', testnet=False):
        self.session = HTTP(testnet=testnet)
        self.symbol = symbol
        self.start_date = datetime(2019, 9, 9)
        self.end_date = datetime(2025, 10, 30)
        
    def download_funding_rates(self):
       
        all_funding = []
        start_time = int(self.start_date.timestamp() * 1000)
        end_time = int(self.end_date.timestamp() * 1000)
        current_time = start_time
        
        with tqdm(desc="Funding rates", unit="requests") as pbar:
            while current_time < end_time:
                
                    response = self.session.get_funding_rate_history(
                        category="linear",
                        symbol=self.symbol,
                        startTime=current_time,
                        endTime=end_time,
                        limit=200
                    )
                    
                    pbar.update(1)
                    
                    if response['retCode'] == 0 and response['result']['list']:
                        data = response['result']['list']
                        all_funding.extend(data)
                        
                        last_timestamp = int(data[-1]['fundingRateTimestamp'])
                        
                        
                        current_time = last_timestamp + 1
                    
                    
                    time.sleep(0.12) 
                    
              
        
        if all_funding:
            df = pd.DataFrame(all_funding)
            df['timestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(float), unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float)
            df = df[['timestamp', 'funding_rate', 'symbol']].sort_values('timestamp').reset_index(drop=True)
           
            return df
        
    
    def download_klines(self, interval='60'):
        
        
        all_klines = []
        start_time = int(self.start_date.timestamp() * 1000)
        end_time = int(self.end_date.timestamp() * 1000)
        current_time = start_time
        
        interval_ms = int(interval) * 60 * 1000
        total_candles = (end_time - start_time) // interval_ms
        expected_requests = (total_candles // 200) + 1
        
        with tqdm(total=expected_requests, desc="Klines", unit="requests") as pbar:
            while current_time < end_time:
                    try:
                        response = self.session.get_kline(
                            category="linear",
                            symbol=self.symbol,
                            interval=interval,
                            start=current_time,
                            limit=200
                        )
                    except Exception as e:
                        continue
                    pbar.update(1)
                    
                    if response['retCode'] == 0 and response['result']['list']:
                        data = response['result']['list']
                        all_klines.extend(data)
                        
                        last_timestamp = int(data[-1][0])
                        
                        if len(data) < 200:
                            break
                        
                        current_time = last_timestamp + interval_ms
                        
                        if current_time >= end_time:
                            break
                    else:
                        break
                    
                    time.sleep(0.12)
                    
               
        
        if all_klines:
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
          
            return df
        
    
    def estimate_cvd_from_klines(self, klines_df):
        
       
        df = klines_df.copy()
        
        df['price_change'] = df['close'] - df['open']
        df['price_change_pct'] = (df['price_change'] / df['open']) * 100
        
        df['candle_strength'] = df['price_change'] / (df['high'] - df['low'] + 1e-10)
        
        df['estimated_buy_volume'] = df.apply(
            lambda row: row['volume'] * (0.5 + 0.5 * row['candle_strength']) 
            if row['price_change'] >= 0 
            else row['volume'] * (0.5 + 0.5 * row['candle_strength']),
            axis=1
        )
        
        df['estimated_sell_volume'] = df['volume'] - df['estimated_buy_volume']
        df['volume_delta'] = df['estimated_buy_volume'] - df['estimated_sell_volume']
        
        df['cvd'] = df['volume_delta'].cumsum()
        
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['total_wick'] = df['upper_wick'] + df['lower_wick']
        df['wick_ratio'] = df['total_wick'] / (df['body_size'] + 1e-10)
        
        df['estimated_liquidation_volume'] = df.apply(
            lambda row: row['volume'] * min(row['wick_ratio'] / 10, 1.0) 
            if row['wick_ratio'] > 2 else 0,
            axis=1
        )
        
        df['cumulative_liquidations'] = df['estimated_liquidation_volume'].cumsum()
       
        return df
    
    def merge_all_data(self, funding_df, klines_cvd_df):
      
        merged = klines_cvd_df.copy()
        
        if not funding_df.empty:
            funding_merge = funding_df[['timestamp', 'funding_rate']].copy()
            merged = pd.merge(merged, funding_merge, on='timestamp', how='left')
            
            merged['funding_rate'] = merged['funding_rate'].fillna(method='ffill')
        
        final_columns = [
            'timestamp',
            'open', 'high', 'low', 'close', 'volume',
            'funding_rate',
            'cvd',
            'volume_delta',
            'estimated_buy_volume',
            'estimated_sell_volume',
            'cumulative_liquidations',
            'estimated_liquidation_volume',
            'wick_ratio'
        ]
        
        final_columns = [col for col in final_columns if col in merged.columns]
        merged = merged[final_columns]
  
        return merged
    



class GreeksCalculator:
    
    def __init__(self, time_to_expiration=1.0):
       
        self.T = time_to_expiration
    
    @staticmethod
    def get_treasury_rate(date):
        
        if date.timestamp() < datetime(2020, 3, 1).timestamp():
            return 0.0175  # Pre-COVID: ~1.75%
        elif date.timestamp() < datetime(2021, 1, 1).timestamp():
            return 0.0015  # COVID era: ~0.15%
        elif date.timestamp() < datetime(2022, 1, 1).timestamp():
            return 0.0010  # 2021: ~0.10%
        elif date.timestamp() < datetime(2023, 1, 1).timestamp():
            return 0.0200  # 2022 rate hikes: ~2.0%
        elif date.timestamp() < datetime(2024, 1, 1).timestamp():
            return 0.0480  # 2023: ~4.8%
        else:
            return 0.0450  # 2024+: ~4.5%
    
    @staticmethod
    def calculate_historical_volatility(prices, window=252):
    
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        return volatility
    
    def black_scholes_call(self, S, K, r, sigma):
     
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * self.T) * norm.cdf(d2)
        return call_price
    
    def calculate_delta(self, S, K, r, sigma, option_type='call'):
      
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def calculate_gamma(self, S, K, r, sigma):
      
        
            
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(self.T))
        return gamma
    
    def calculate_vega(self, S, K, r, sigma):
       
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        
        vega = S * norm.pdf(d1) * np.sqrt(self.T) / 100  # Divided by 100 for 1% change
        return vega


def download_historical_data(ticker, start_date, end_date):
    
    stock = yf.Ticker(ticker)
    
    
    df = stock.history(start=start_date, end=end_date)
    
    return df


def calculate_greeks_timeseries(ticker='SPY', start_date='2019-09-01', end_date=None,
                                moneyness_levels=[0.90, 0.95, 1.00, 1.05, 1.10],
                                volatility_window=252):
   
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    

    hist_data = download_historical_data(ticker, start_date, end_date)
    
    hist_data['returns'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
    hist_data['volatility'] = hist_data['returns'].rolling(window=volatility_window).std() * np.sqrt(252)
    
    hist_data = hist_data[hist_data.index >= start_date]
    
    calc = GreeksCalculator(time_to_expiration=1.0)
    
    results = []
    
    for date, row in hist_data.iterrows():
        S = row['Close']  
        sigma = row['volatility'] 
        
        if np.isnan(sigma) or sigma <= 0:
            continue
        
        r = calc.get_treasury_rate(date)
        
        for moneyness in moneyness_levels:
            K = S * moneyness  
            
            delta_call = calc.calculate_delta(S, K, r, sigma, option_type='call')
            delta_put = calc.calculate_delta(S, K, r, sigma, option_type='put')
            gamma = calc.calculate_gamma(S, K, r, sigma)
            vega = calc.calculate_vega(S, K, r, sigma)
            
            results.append({
                'date': date,
                'ticker': ticker,
                'spot_price': S,
                'strike_price': K,
                'moneyness': moneyness,
                'moneyness_pct': f"{int(moneyness*100)}%",
                'volatility': sigma,
                'risk_free_rate': r,
                'delta_call': delta_call,
                'delta_put': delta_put,
                'gamma': gamma,
                'vega': vega
            })
    
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    
    return results_df


    
    
  

#downloader = BybitDataDownloaderOptimized(symbol='BTCUSDT', testnet=False)
    
#funding_df = downloader.download_funding_rates()
#klines_df = downloader.download_klines(interval='60')
#klines_cvd_df = downloader.estimate_cvd_from_klines(klines_df)
  
    
#final_df = downloader.merge_all_data(funding_df, klines_cvd_df)
#final_df.to_csv('futures.csv')
    
end_date = datetime(2025, 11, 30)
    
start_date = datetime(2019, 5, 1)

collector = GreeksCalculator()

greeks_df = calculate_greeks_timeseries(
        ticker='BTC-USD',
        start_date='2013-09-01',
        end_date=None,  
        moneyness_levels=[0.90, 0.95, 1.00, 1.05, 1.10],
        volatility_window=252
    )
greeks_df.to_csv('greeks.csv')