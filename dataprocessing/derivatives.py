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
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    def get_trades_history(self, currency: str, start_timestamp: int, 
                          end_timestamp: int, kind: str = 'future') -> pd.DataFrame:
        all_data = []
        current_start = start_timestamp

        url = f"{self.base_url}/public/get_last_trades_by_currency_and_time"
        chunk_size = 86400000
        while current_start < end_timestamp:
            current_end = min(current_start + chunk_size, end_timestamp)

            params = {
                'currency': currency,
                'start_timestamp': current_start,
                'end_timestamp': current_end,  
                'count': 1000,
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
                
            if 'result' in data and 'trades' in data['result']:
                trades = data['result']['trades']
                all_data.extend(trades)
            
                if len(trades) < 1000:
                    current_start = current_end +1
                else:
                    current_start = trades[-1]['timestamp'] + 1
                        
            time.sleep(5.0)
           
        
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if 'liquidation' not in df.columns:
            df['liquidation'] = [t.get('liquidation', None) for t in all_data]
        return df
    
    
            
        
        
    
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
       
          
    def get_ticker_by_instrument(self, instrument_name: str) -> Optional[Dict]:
        
            url = f"{self.base_url}/public/ticker"
            params = {'instrument_name': instrument_name}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            return data['result']
       
        
    
    def download_options_snapshot(self, currency: str, date: datetime) -> pd.DataFrame:
        instruments = self.get_instruments_by_date(currency, date)
        
        url = f"{self.base_url}/public/get_book_summary_by_currency"
        payload = {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "public/get_book_summary_by_currency",
                    "params": {
                        'currency': currency,
                        'kind': 'option'
                    }
                }
                
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        options_data = []
        timestamp = int(date.timestamp() * 1000)
        book_summary = {item['instrument_name']: item for item in data['result']}

        
        for inst in instruments:
            inst_name = inst['instrument_name']
            ticker = book_summary[inst_name]

            #ticker = self.get_ticker_by_instrument(inst['instrument_name']) #Too slow
            
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
                    'volume_24h': ticker.get('stats', {}).get('volume', 0) if ticker.get('stats') else 0,
                    'volume_usd': ticker.get('stats', {}).get('volume_usd', 0) if ticker.get('stats') else 0,
                    
                    'delta': ticker.get('greeks', {}).get('delta') if ticker.get('greeks') else None,
                    'gamma': ticker.get('greeks', {}).get('gamma') if ticker.get('greeks') else None,
                    'vega': ticker.get('greeks', {}).get('vega') if ticker.get('greeks') else None,
                    'theta': ticker.get('greeks', {}).get('theta') if ticker.get('greeks') else None,
                }
                
            options_data.append(option_data)
            
        
        return pd.DataFrame(options_data)


def aggregate_to_daily(funding_df: pd.DataFrame,
                      trades_df: pd.DataFrame,
                      liquidations_df: pd.DataFrame,
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
        
        if not trades_df.empty:
            day_trades = trades_df[trades_df['timestamp'].dt.date == date.date()]
            if not day_trades.empty:
                buys = day_trades[day_trades['direction'] == 'buy']['amount'].sum()
                sells = day_trades[day_trades['direction'] == 'sell']['amount'].sum()
                features['cvd'] = buys - sells
                total_vol = day_trades['amount'].sum()
                features['cvd_normalized'] = features['cvd'] / total_vol if total_vol > 0 else 0
                features['trade_count'] = len(day_trades)
                features['volume'] = total_vol
        
        if not liquidations_df.empty:
            day_liq = liquidations_df[liquidations_df['timestamp'].dt.date == date.date()]
            if not day_liq.empty:
                long_liqs = day_liq[day_liq['direction'] == 'sell']
                short_liqs = day_liq[day_liq['direction'] == 'buy']
                
                features['liq_long_volume'] = long_liqs['amount'].sum()
                features['liq_short_volume'] = short_liqs['amount'].sum()
                features['liq_total_volume'] = day_liq['amount'].sum()
                features['liq_count'] = len(day_liq)
                features['liq_long_count'] = len(long_liqs)
                features['liq_short_count'] = len(short_liqs)
                
                if daily_features:
                    prev_cumulative = daily_features[-1].get('cumulative_liquidations', 0)
                    features['cumulative_liquidations'] = prev_cumulative + features['liq_total_volume']
                else:
                    features['cumulative_liquidations'] = features['liq_total_volume']
        
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
                                     ):
    
    
    downloader = DeribitUnifiedDownloader()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    
    
    
    funding_df = downloader.get_funding_rate_history(perpetual_instrument, start_ts, end_ts)
    time.sleep(100.0)
    trades_df = downloader.get_trades_history(currency, start_ts, end_ts, kind='future')
    time.sleep(100.0)
    liquidations_df = trades_df[trades_df['liquidation'].notna() & (trades_df['liquidation'] != False)].copy()
    oi_df = downloader.get_open_interest_snapshots(perpetual_instrument, start_ts, end_ts)
    time.sleep(100.0)
    orderbook_df = downloader.get_orderbook_depth_samples(perpetual_instrument, start_ts, end_ts, sample_days=1)
    time.sleep(100.0)


    
    processed_dates = set()
    all_options = []
    current = start_dt
        
    while current <= end_dt:
            
        if current.date() in processed_dates:
            current += timedelta(days=options_sample_days)
            continue
            
            
        snapshot = downloader.download_options_snapshot(currency, current)
                
        if not snapshot.empty:
            all_options.append(snapshot)
                            
        current += timedelta(days=options_sample_days)
        time.sleep(5.0)  
        
    if all_options:
        options_df = pd.concat(all_options, ignore_index=True)
            
            
        
    
    
    
    metrics = aggregate_to_daily(
        funding_df, trades_df, liquidations_df, oi_df, orderbook_df, options_df,
        start_date, end_date
    )
    
  
    return metrics


metrics = download_complete_deribit_history(
        perpetual_instrument="BTC-PERPETUAL",
        currency="BTC",
        start_date="2019-05-01", 
        end_date="2025-10-16", 
        options_sample_days=1
)
    
   
final_metrics = metrics[[
            'date',
            'open_interest_futures',
            'funding_rate_avg',
            'bid_ask_spread_bps',
            'depth_1pct',
            'cumulative_liquidations',
            'put_call_ratio',
            'cvd',
            'gamma_exposure',
            'delta_exposure',
            'vega_exposure'
            
        ]].copy()
        
final_metrics.columns = [
            'date',
            'Open Interest',
            'Funding Rate',
            'Bid-Ask Spread',
            'Depth at 1%',
            'Cumulative Liquidations',
            'Put-Call Ratio',
            'CVD',
            'Gamma Exposure',
            'Delta Exposure',
            'Vega Exposure'
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
    
 