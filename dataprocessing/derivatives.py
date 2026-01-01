import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path

class DeribitUnifiedDownloader:
    
    def __init__(self, data_dir: str = "./deribit_data"):
        self.base_url = "https://www.deribit.com/api/v2"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    
    def get_funding_rate_history(self, instrument: str, start_timestamp: int, 
                                 end_timestamp: int) -> pd.DataFrame:
        all_data = []
        
        url = f"{self.base_url}/public/get_funding_rate_history"
        params = {
            'instrument_name': instrument,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'result' in data and data['result']:
            all_data.extend(data['result'])
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_ohlcv_history(self, currency: str, start_timestamp: int,
                         end_timestamp: int, resolution: int = 60) -> pd.DataFrame:
       
        
        url = f"{self.base_url}/public/get_tradingview_chart_data"
        
        resolution_map = {
            1: '1', 5: '5', 15: '15', 30: '30', 60: '60', 
            120: '120', 180: '180', 360: '360', 720: '720', 1440: '1D'
        }
        
        params = {
            'instrument_name': f'{currency}-PERPETUAL',
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'resolution': resolution_map[resolution]
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
            
        if 'result' in data:
                result = data['result']
                
                df = pd.DataFrame({
                    'timestamp': result['ticks'],
                    'open': result['open'],
                    'high': result['high'],
                    'low': result['low'],
                    'close': result['close'],
                    'volume': result['volume']
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                df['price_change'] = df['close'].pct_change()
                df['price_drop'] = df['price_change'].clip(upper=0).abs()
                df['volatility'] = df['price_change'].abs()
                
                df['volume_ma'] = df['volume'].rolling(24, min_periods=1).mean()
                df['volume_spike'] = df['volume'] / df['volume_ma']
                df['liquidation_proxy'] = df['volume'] * df['price_drop'] * df['volume_spike']
                
                return df
        
        
        return pd.DataFrame()
    
    def get_volatility_index(self, currency: str, start_timestamp: int,
                            end_timestamp: int) -> pd.DataFrame:
       
        
        url = f"{self.base_url}/public/get_historical_volatility"
        
        params = {
            'currency': currency,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
            
        if 'result' in data and data['result']:
                result = data['result']
                
                df = pd.DataFrame(result, columns=['timestamp', 'volatility'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                return df
        
        
        return pd.DataFrame()
    
    def get_open_interest_snapshots(self, instrument: str, start_timestamp: int,
                                   end_timestamp: int) -> pd.DataFrame:
       
        all_data = []
        current = start_timestamp
        
        sample_interval = 4 * 3600 * 1000  
        
        while current < end_timestamp:
            url = f"{self.base_url}/public/ticker"
            params = {'instrument_name': instrument}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'result' in data:
                result = data['result']
                all_data.append({
                    'timestamp': current,
                    'open_interest': result.get('open_interest', 0),
                    'instrument_name': instrument
                })
            
            current += sample_interval
            time.sleep(5.0)
        
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_orderbook_depth_samples(self, instrument: str, start_timestamp: int,
                                   end_timestamp: int, sample_days: int = 1) -> pd.DataFrame:
        
        all_samples = []
        current = start_timestamp
        
        sample_interval = sample_days * 86400000
        
        while current < end_timestamp:
            url = f"{self.base_url}/public/get_order_book"
            params = {
                'instrument_name': instrument,
                'depth': 20
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'result' in data:
                result = data['result']
                
                bids = result.get('bids', [])
                asks = result.get('asks', [])
                
                if bids and asks:
                    best_bid = bids[0][0] if bids else 0
                    best_ask = asks[0][0] if asks else 0
                    mid_price = (best_bid + best_ask) / 2
                    
                    spread = best_ask - best_bid
                    spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
                    
                    one_pct_below = mid_price * 0.99
                    one_pct_above = mid_price * 1.01
                    
                    bid_depth_1pct = sum(qty for price, qty in bids if price >= one_pct_below)
                    ask_depth_1pct = sum(qty for price, qty in asks if price <= one_pct_above)
                    
                    all_samples.append({
                        'timestamp': current,
                        'best_bid': best_bid,
                        'best_ask': best_ask,
                        'mid_price': mid_price,
                        'spread_bps': spread_bps,
                        'bid_depth_1pct': bid_depth_1pct,
                        'ask_depth_1pct': ask_depth_1pct,
                        'total_depth_1pct': bid_depth_1pct + ask_depth_1pct,
                        'depth_imbalance': (bid_depth_1pct - ask_depth_1pct) / (bid_depth_1pct + ask_depth_1pct) if (bid_depth_1pct + ask_depth_1pct) > 0 else 0
                    })
            
            current += sample_interval
            time.sleep(1.0)
        
        df = pd.DataFrame(all_samples)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_instruments_by_date(self, currency: str, date: datetime) -> List[Dict]:
        timestamp = int(date.timestamp() * 1000)
    
        url = f"{self.base_url}/public/get_instruments"
        params = {
            'currency': currency,
            'kind': 'option',
        }
        
        response = requests.get(url, params=params)
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
        
        return []
    
    def get_ticker_by_instrument(self, instrument_name: str) -> Optional[Dict]:
        url = f"{self.base_url}/public/ticker"
        params = {'instrument_name': instrument_name}
        
        response = requests.get(url, params=params)
        data = response.json()
        
        return data['result']
    
        
    def download_options_snapshot(self, currency: str, date: datetime) -> pd.DataFrame:
        
            url = f"{self.base_url}/public/get_instruments"
            params = {
                'currency': currency,
                'kind': 'option',
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            
            instruments = data['result']
            url = f"{self.base_url}/public/get_book_summary_by_currency"
            params = {
                'currency': currency,
                'kind': 'option'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            book_summary = {item['instrument_name']: item for item in data['result']}
            
            options_data = []
            timestamp = int(date.timestamp() * 1000)  
            
            for inst in instruments:
                inst_name = inst['instrument_name']
                if inst_name not in book_summary:
                    continue
                
                ticker = book_summary[inst_name]
                
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
                    'bid_price': ticker.get('bid_price'),
                    'ask_price': ticker.get('ask_price'),
                    'last_price': ticker.get('last'),
                    
                    'open_interest': ticker.get('open_interest', 0),
                    'volume_24h': ticker.get('volume', 0),
                    'volume_usd': ticker.get('volume_usd', 0),
                    
                    'delta': None,
                    'gamma': None,
                    'vega': None,
                    'theta': None,
                }
                
                options_data.append(option_data)
            
            return pd.DataFrame(options_data)

def aggregate_to_daily(funding_df: pd.DataFrame,
                      ohlcv_df: pd.DataFrame,
                      volatility_df: pd.DataFrame,
                      oi_df: pd.DataFrame,
                      orderbook_df: pd.DataFrame,
                      options_df: pd.DataFrame,
                      start_date: str,
                      end_date: str) -> pd.DataFrame:
    
    daily_features = []
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    for date in dates:
        features = {'date': date.date()}
        
        if not funding_df.empty:
            day_funding = funding_df[funding_df['timestamp'].dt.date == date.date()]
            if not day_funding.empty:
                features['funding_rate_avg'] = day_funding['interest_8h'].mean()
                features['funding_rate_std'] = day_funding['interest_8h'].std()
                features['funding_rate_1h_avg'] = day_funding['interest_1h'].mean() if 'interest_1h' in day_funding.columns else None
        
        if not ohlcv_df.empty:
            day_ohlcv = ohlcv_df[ohlcv_df['timestamp'].dt.date == date.date()]
            if not day_ohlcv.empty:
                features['volume'] = day_ohlcv['volume'].sum()
                features['trade_count'] = len(day_ohlcv) * 100  
                
                funding_for_day = funding_df[funding_df['timestamp'].dt.date == date.date()]['interest_8h'].mean() if not funding_df.empty else 0
                price_change = (day_ohlcv['close'].iloc[-1] - day_ohlcv['open'].iloc[0]) / day_ohlcv['open'].iloc[0]
                
                features['cvd'] = day_ohlcv['volume'].sum() * price_change
                features['cvd_normalized'] = price_change  
                
                liquidation_estimate = day_ohlcv['liquidation_proxy'].sum()
                features['liq_total_volume'] = liquidation_estimate
                
                negative_candles = day_ohlcv[day_ohlcv['close'] < day_ohlcv['open']]
                positive_candles = day_ohlcv[day_ohlcv['close'] > day_ohlcv['open']]
                
                features['liq_long_volume'] = negative_candles['liquidation_proxy'].sum()  
                features['liq_short_volume'] = positive_candles['liquidation_proxy'].sum()  
                
                
                volume_threshold = day_ohlcv['volume'].quantile(0.9)
                liquidation_events = day_ohlcv[day_ohlcv['volume'] > volume_threshold]
                features['liq_count'] = len(liquidation_events)
                features['liq_long_count'] = len(liquidation_events[liquidation_events['close'] < liquidation_events['open']])
                features['liq_short_count'] = len(liquidation_events[liquidation_events['close'] > liquidation_events['open']])
                
                if daily_features:
                    prev_cumulative = daily_features[-1].get('cumulative_liquidations', 0)
                    features['cumulative_liquidations'] = prev_cumulative + features['liq_total_volume']
                else:
                    features['cumulative_liquidations'] = features['liq_total_volume']
        
        if not volatility_df.empty:
            day_vol = volatility_df[volatility_df['timestamp'].dt.date == date.date()]
            if not day_vol.empty:
                features['volatility_index'] = day_vol['volatility'].mean()
        
        if not oi_df.empty:
            day_oi = oi_df[oi_df['timestamp'].dt.date == date.date()]
            if not day_oi.empty:
                features['open_interest_futures'] = day_oi['open_interest'].mean()
                features['oi_change'] = day_oi['open_interest'].diff().sum()
        
        if not orderbook_df.empty:
            day_book = orderbook_df[orderbook_df['timestamp'].dt.date == date.date()]
            if not day_book.empty:
                features['bid_ask_spread_bps'] = day_book['spread_bps'].mean()
                features['bid_ask_spread_max_bps'] = day_book['spread_bps'].max()
                features['depth_1pct'] = day_book['total_depth_1pct'].mean()
                features['bid_depth_1pct'] = day_book['bid_depth_1pct'].mean()
                features['ask_depth_1pct'] = day_book['ask_depth_1pct'].mean()
                features['depth_imbalance'] = day_book['depth_imbalance'].mean()
        
        if not options_df.empty:
            day_options = options_df[options_df['date'] == date.date()]
            if not day_options.empty:
                puts = day_options[day_options['option_type'] == 'put']
                calls = day_options[day_options['option_type'] == 'call']
                
                put_oi = puts['open_interest'].sum()
                call_oi = calls['open_interest'].sum()
                features['put_call_ratio'] = put_oi / call_oi if call_oi > 0 else 0
                
                put_vol = puts['volume_24h'].sum()
                call_vol = calls['volume_24h'].sum()
                features['put_call_ratio_volume'] = put_vol / call_vol if call_vol > 0 else 0
                
                valid_greeks = day_options.dropna(subset=['delta', 'gamma', 'vega', 'open_interest', 'underlying_price'])
                
                if not valid_greeks.empty:
                    underlying_price = valid_greeks['underlying_price'].iloc[0]
                    
                    features['delta_exposure'] = (valid_greeks['delta'] * valid_greeks['open_interest'] * underlying_price).sum()
                    features['delta_exposure_abs'] = (valid_greeks['delta'].abs() * valid_greeks['open_interest'] * underlying_price).sum()
                    
                    features['gamma_exposure'] = (valid_greeks['gamma'] * valid_greeks['open_interest'] * underlying_price**2).sum()
                    features['gamma_exposure_abs'] = (valid_greeks['gamma'].abs() * valid_greeks['open_interest'] * underlying_price**2).sum()
                    
                    features['vega_exposure'] = (valid_greeks['vega'] * valid_greeks['open_interest']).sum()
                    features['vega_exposure_abs'] = (valid_greeks['vega'].abs() * valid_greeks['open_interest']).sum()
                    
                    features['underlying_price'] = underlying_price
                
                features['options_total_oi'] = day_options['open_interest'].sum()
                features['options_total_volume'] = day_options['volume_24h'].sum()
        
        daily_features.append(features)
    
    return pd.DataFrame(daily_features)


def download_complete_deribit_history(perpetual_instrument: str = "BTC-PERPETUAL",
                                     currency: str = "BTC",
                                     start_date: str = "2019-06-01",
                                     end_date: str = "2025-10-14",
                                     options_sample_days: int = 1,
                                     ohlcv_resolution: int = 60):
    
    downloader = DeribitUnifiedDownloader()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    funding_df = downloader.get_funding_rate_history(perpetual_instrument, start_ts, end_ts)
    time.sleep(2.0)
    
    ohlcv_df = downloader.get_ohlcv_history(currency, start_ts, end_ts, resolution=ohlcv_resolution)
    time.sleep(2.0)
    
    volatility_df = downloader.get_volatility_index(currency, start_ts, end_ts)
    time.sleep(2.0)
   
    oi_df = downloader.get_open_interest_snapshots(perpetual_instrument, start_ts, end_ts)
    time.sleep(2.0)
    
    orderbook_df = downloader.get_orderbook_depth_samples(perpetual_instrument, start_ts, end_ts, sample_days=options_sample_days)
    time.sleep(2.0)
    
    all_options = []
    current = start_dt
    
    while current <= end_dt:
        
        snapshot = downloader.download_options_snapshot(currency, current)
        
        if not snapshot.empty:
            all_options.append(snapshot)
        
        current += timedelta(days=options_sample_days)
        time.sleep(5.0)
    
    options_df = pd.concat(all_options, ignore_index=True)
   
    
    metrics = aggregate_to_daily(
        funding_df, ohlcv_df, volatility_df, oi_df, orderbook_df, options_df,
        start_date, end_date
    )
    
    
    return metrics


metrics = download_complete_deribit_history(
    perpetual_instrument="BTC-PERPETUAL",
    currency="BTC",
    start_date="2019-06-01", 
    end_date="2025-10-16", 
    options_sample_days=1,
    ohlcv_resolution=60  
)

final_metrics = metrics[[
    'date',
    'open_interest_futures',
    'funding_rate_avg',
    'bid_ask_spread_bps',
    'depth_1pct',
    'cumulative_liquidations',
    'put_call_ratio',
    'cvd'
]].copy()

final_metrics.columns = [
    'date',
    'Open Interest',
    'Funding Rate',
    'Bid-Ask Spread',
    'Depth at 1%',
    'Cumulative Liquidations',
    'Put-Call Ratio',
    'CVD'
]

final_metrics = final_metrics.set_index('date')

full_date_range = pd.date_range(
    start=final_metrics.index.min(),
    end=final_metrics.index.max(),
    freq='D'
)

final_metrics = final_metrics.reindex(full_date_range)
final_metrics = final_metrics.fillna(method='ffill')

with open("derivatives.csv", "w") as f:
    final_metrics.to_csv(f)
