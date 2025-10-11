import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta
from typing import List, Optional
import os
from pathlib import Path
import time

class BinanceHistoricalDownloader:
    
    def __init__(self, data_dir: str = "./binance_data"):
        self.base_url = "https://data.binance.vision/data"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_and_extract_zip(self, url: str) -> Optional[pd.DataFrame]:
        
        response = requests.get(url, timeout=30)
        time.sleep(0.1)    
            
            
        response.raise_for_status()
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
            
            return df
            
        
    
    def get_funding_rate_history(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        all_data = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            year_month = current.strftime("%Y-%m")
            url = f"{self.base_url}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{year_month}.zip"
            
            df = self.download_and_extract_zip(url)
            if df is not None:
                all_data.append(df)
            
            current += timedelta(days=32)
            current = current.replace(day=1)
            time.sleep(0.5)  
        
       
        
        combined = pd.concat(all_data, ignore_index=True)
        combined['calc_time'] = pd.to_datetime(combined['calc_time'], unit='ms')
        
        return combined
    
   
    
    def get_trades_history(self, symbol: str, start_date: str, end_date: str, 
                          sample_days: int = 30) -> Optional[pd.DataFrame]:
        
        all_data = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_since_sample = 0
        
        while current <= end:
            if days_since_sample % sample_days == 0:
                date_str = current.strftime("%Y-%m-%d")
                url = f"{self.base_url}/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
                
                df = self.download_and_extract_zip(url)
                if df is not None:
                    df['date'] = date_str
                    all_data.append(df)
            
            current += timedelta(days=1)
            days_since_sample += 1
            time.sleep(0.3)
        
       
        combined = pd.concat(all_data, ignore_index=True)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'], unit='ms')
        
        
        return combined
    
    



def aggregate_to_daily(funding_df: pd.DataFrame, 
                          trades_df: pd.DataFrame, 
                          start_date = str, 
                          end_date = str) -> Optional[pd.DataFrame]:
        
        daily_features = []
        
       
        dates = pd.date_range(
            start=start_date.date(),
            end=end_date.date(),
            freq='D'
        )
       
        for date in dates:
            features = {'date': date.date()}
            
            if not funding_df.empty:
                day_funding = funding_df[funding_df['calc_time'].dt.date == date.date()]
                if not day_funding.empty:
                    features['funding_avg'] = day_funding['funding_rate'].mean()
                    features['funding_std'] = day_funding['funding_rate'].std()
                    features['funding_sum'] = day_funding['funding_rate'].sum()
                    features['funding_count'] = len(day_funding)
            
            
            
           
            if trades_df is not None and not trades_df.empty:
                day_trades = trades_df[trades_df['timestamp'].dt.date == date.date()]
                if not day_trades.empty:
                    buys = day_trades[~day_trades['is_buyer_maker']]['quantity'].sum()
                    sells = day_trades[day_trades['is_buyer_maker']]['quantity'].sum()
                    features['cvd'] = buys - sells
                    total_vol = day_trades['quantity'].sum()
                    features['cvd_normalized'] = features['cvd'] / total_vol if total_vol > 0 else 0
            
            daily_features.append(features)
        
        df = pd.DataFrame(daily_features)
      
        return df


def download_complete_history(symbol: str, 
                              start_date: str,
                              end_date: str):
    
    downloader = BinanceHistoricalDownloader()
    
    funding_df = downloader.get_funding_rate_history(symbol, start_date, end_date)
    trades_df = downloader.get_trades_history(symbol, start_date, end_date, sample_days=30)
  
    metrics = aggregate_to_daily(
        funding_df,  trades_df, start_date, end_date
    )
    
    
    
    return metrics



metrics = download_complete_history(
    symbol="BTCUSDT",
    start_date="2019-09-01",  
    end_date="2025-09-30"
)
    
with open("futures.csv", "w") as f:
    metrics.to_csv(f)