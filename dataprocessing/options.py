import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
from scipy.stats import norm
import json

class DeribitHistoricalDownloader:
    
    def __init__(self, data_dir: str = "./deribit_data"):
        self.base_url = "https://www.deribit.com/api/v2"
        self.hist_url = "https://history.deribit.com/api/v2"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def get_historical_volatility(self, currency: str, start_timestamp: int, 
                                  end_timestamp: int) -> pd.DataFrame:
       
        url = f"{self.base_url}/public/get_historical_volatility"
        params = {
            'currency': currency,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        }
            
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
            
        if 'result' in data:
            df = pd.DataFrame(data['result'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
          
            
       
    
    def get_instruments_by_date(self, currency: str, date: datetime) -> List[Dict]:
      
        timestamp = int(date.timestamp() * 1000)
            
        url = f"{self.base_url}/public/get_instruments"
        params = {
            'currency': currency,
            'kind': 'option',
            'expired': False
        }
            
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
            
        if 'result' in data:
            instruments = data['result']
                
            active_instruments = []
            for inst in instruments:
                creation = inst.get('creation_timestamp', 0)
                expiration = inst.get('expiration_timestamp', float('inf'))
                    
                if creation <= timestamp <= expiration:
                    active_instruments.append(inst)
                
            return active_instruments
            
       
            
        
    def get_ticker_by_instrument(self, instrument_name: str, timestamp: int) -> Optional[Dict]:
        
        url = f"{self.base_url}/public/ticker"
        params = {
            'instrument_name': instrument_name
            }
            
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
            
        if 'result' in data:
            return data['result']
            
            
            
        
    
    def get_order_book_snapshot(self, instrument_name: str, timestamp: int) -> Optional[Dict]:
        
        url = f"{self.base_url}/public/get_order_book"
        params = {
            'instrument_name': instrument_name,
            'depth': 1  
            }
            
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
            
        if 'result' in data:
            return data['result']
            
          
    
    def download_options_snapshot_for_date(self, currency: str, date: datetime) -> pd.DataFrame:
        
        instruments = self.get_instruments_by_date(currency, date)
        
        options_data = []
        timestamp = int(date.timestamp() * 1000)
        
        for i, inst in enumerate(instruments):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(instruments)}")
            
           
            ticker = self.get_ticker_by_instrument(inst['instrument_name'], timestamp)
                
            if ticker:
                    
                parts = inst['instrument_name'].split('-')
                    
                option_data = {
                        'date': date.date(),
                        'timestamp': timestamp,
                        'instrument_name': inst['instrument_name'],
                        'currency': currency,
                        'strike': inst['strike'],
                        'option_type': inst['option_type'], 
                        'expiration_timestamp': inst['expiration_timestamp'],
                        'expiration_date': datetime.fromtimestamp(inst['expiration_timestamp'] / 1000),
                        
                        'underlying_price': ticker.get('underlying_price'),
                        'mark_price': ticker.get('mark_price'),
                        'mark_iv': ticker.get('mark_iv'),
                        'bid_price': ticker.get('best_bid_price'),
                        'ask_price': ticker.get('best_ask_price'),
                        'last_price': ticker.get('last_price'),
                        
                        'open_interest': ticker.get('open_interest', 0),
                        'volume_24h': ticker.get('stats', {}).get('volume', 0),
                        'volume_usd': ticker.get('stats', {}).get('volume_usd', 0),
                        
                        'delta': ticker.get('greeks', {}).get('delta'),
                        'gamma': ticker.get('greeks', {}).get('gamma'),
                        'vega': ticker.get('greeks', {}).get('vega'),
                        'theta': ticker.get('greeks', {}).get('theta'),
                        'rho': ticker.get('greeks', {}).get('rho'),
                    }
                    
                options_data.append(option_data)
                
                time.sleep(0.1)  
                
           
        
        df = pd.DataFrame(options_data)
        
        return df
    
    def download_historical_options_range(self, currency: str, start_date: str, 
                                         end_date: str, frequency_days: int = 1) -> pd.DataFrame:
       
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_snapshots = []
        current = start
        
        while current <= end:
            snapshot = self.download_options_snapshot_for_date(currency, current)
            
            if not snapshot.empty:
                all_snapshots.append(snapshot)
                
                snapshot_file = self.data_dir / f"{currency}_options_snapshot_{current.date()}.csv"
                snapshot.to_csv(snapshot_file, index=False)
            
            current += timedelta(days=frequency_days)
            time.sleep(1)  
        
        
        
        combined = pd.concat(all_snapshots, ignore_index=True)
        
        
        return combined


class OptionsMetricsCalculator:
    
    @staticmethod
    def calculate_put_call_ratio(options_df: pd.DataFrame, metric: str = 'volume') -> Dict:
        
        if options_df.empty:
            return {}
        
        puts = options_df[options_df['option_type'] == 'put']
        calls = options_df[options_df['option_type'] == 'call']
        
        if metric == 'volume':
            put_metric = puts['volume_24h'].sum()
            call_metric = calls['volume_24h'].sum()
        else:  
            put_metric = puts['open_interest'].sum()
            call_metric = calls['open_interest'].sum()
        
        pcr = put_metric / call_metric if call_metric > 0 else 0
        
        return {
            f'put_{metric}': put_metric,
            f'call_{metric}': call_metric,
            f'pcr_{metric}': pcr
        }
    
    @staticmethod
    def calculate_total_greeks_exposure(options_df: pd.DataFrame) -> Dict:
      
        valid_data = options_df.dropna(subset=['delta', 'gamma', 'vega', 'open_interest'])
        
        
        underlying_price = valid_data['underlying_price'].iloc[0]
        
        total_delta = (valid_data['delta'].abs() * valid_data['open_interest'] * underlying_price).sum()
        total_gamma = (valid_data['gamma'].abs() * valid_data['open_interest'] * underlying_price**2).sum()
        total_vega = (valid_data['vega'].abs() * valid_data['open_interest']).sum()
        
        net_delta = (valid_data['delta'] * valid_data['open_interest'] * underlying_price).sum()
        net_gamma = (valid_data['gamma'] * valid_data['open_interest'] * underlying_price**2).sum()
        net_vega = (valid_data['vega'] * valid_data['open_interest']).sum()
        
        return {
            'total_delta_exposure': total_delta,
            'total_gamma_exposure': total_gamma,
            'total_vega_exposure': total_vega,
            'net_delta': net_delta,
            'net_gamma': net_gamma,
            'net_vega': net_vega,
            'underlying_price': underlying_price
        }
    
    @staticmethod
    def calculate_greeks_by_moneyness(options_df: pd.DataFrame) -> Dict:
      
        valid_data = options_df.dropna(subset=['strike', 'underlying_price', 'open_interest'])
        valid_data = valid_data.copy()
        valid_data['moneyness'] = valid_data['strike'] / valid_data['underlying_price']
        
        atm_data = valid_data[(valid_data['moneyness'] >= 0.95) & (valid_data['moneyness'] <= 1.05)]
        otm_calls = valid_data[(valid_data['option_type'] == 'call') & (valid_data['moneyness'] > 1.05)]
        otm_puts = valid_data[(valid_data['option_type'] == 'put') & (valid_data['moneyness'] < 0.95)]
        itm_calls = valid_data[(valid_data['option_type'] == 'call') & (valid_data['moneyness'] < 0.95)]
        itm_puts = valid_data[(valid_data['option_type'] == 'put') & (valid_data['moneyness'] > 1.05)]
        
        underlying = valid_data['underlying_price'].iloc[0]
        
        def calc_exposure(df):
            if df.empty or 'gamma' not in df.columns:
                return {'delta': 0, 'gamma': 0, 'vega': 0, 'oi': 0}
            return {
                'delta': (df['delta'].fillna(0) * df['open_interest'] * underlying).sum(),
                'gamma': (df['gamma'].fillna(0) * df['open_interest'] * underlying**2).sum(),
                'vega': (df['vega'].fillna(0) * df['open_interest']).sum(),
                'oi': df['open_interest'].sum()
            }
        
        return {
            'atm': calc_exposure(atm_data),
            'otm_calls': calc_exposure(otm_calls),
            'otm_puts': calc_exposure(otm_puts),
            'itm_calls': calc_exposure(itm_calls),
            'itm_puts': calc_exposure(itm_puts)
        }
    
    @staticmethod
    def aggregate_options_to_daily(options_df: pd.DataFrame) -> pd.DataFrame:
        
        if options_df.empty:
            return pd.DataFrame()
        
        daily_metrics = []
        
        for date, group in options_df.groupby('date'):
            metrics = {'date': date}
            
            pcr_volume = OptionsMetricsCalculator.calculate_put_call_ratio(group, 'volume')
            pcr_oi = OptionsMetricsCalculator.calculate_put_call_ratio(group, 'oi')
            metrics.update(pcr_volume)
            metrics.update(pcr_oi)
            
            greeks = OptionsMetricsCalculator.calculate_total_greeks_exposure(group)
            metrics.update(greeks)
            
            greeks_by_moneyness = OptionsMetricsCalculator.calculate_greeks_by_moneyness(group)
            if greeks_by_moneyness:
                metrics['atm_gamma'] = greeks_by_moneyness['atm']['gamma']
                metrics['atm_delta'] = greeks_by_moneyness['atm']['delta']
                metrics['otm_call_oi'] = greeks_by_moneyness['otm_calls']['oi']
                metrics['otm_put_oi'] = greeks_by_moneyness['otm_puts']['oi']
            
            metrics['total_volume_24h'] = group['volume_24h'].sum()
            metrics['total_volume_usd'] = group['volume_usd'].sum()
            metrics['total_open_interest'] = group['open_interest'].sum()
            
            metrics['avg_iv'] = group['mark_iv'].mean()
            metrics['iv_std'] = group['mark_iv'].std()
            
            metrics['num_contracts'] = len(group)
            metrics['num_calls'] = len(group[group['option_type'] == 'call'])
            metrics['num_puts'] = len(group[group['option_type'] == 'put'])
            
            daily_metrics.append(metrics)
        
        return pd.DataFrame(daily_metrics)


def download_and_process_options_history(currency: str = "BTC",
                                         start_date: str = "2020-01-01",
                                         end_date: str = "2025-09-30",
                                         frequency_days: int = 7):
   
    downloader = DeribitHistoricalDownloader()
    options_df = downloader.download_historical_options_range(
        currency=currency,
        start_date=start_date,
        end_date=end_date,
        frequency_days=frequency_days
    )
    
  
    calculator = OptionsMetricsCalculator()
    daily_metrics = calculator.aggregate_options_to_daily(options_df)
    
  
    return daily_metrics




metrics = download_and_process_options_history(
    currency="BTC",
    start_date="2024-01-01",
    end_date="2025-09-30",
    frequency_days=7  
    )
    
with open("options.csv", "w") as f:
    metrics.to_csv(f)
