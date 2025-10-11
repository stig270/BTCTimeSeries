import numpy as np
import pandas as pd
import datetime as dt   
from google.cloud import bigquery

projectID = "btcdataprocessing"



class CompleteDailyBitcoinMetrics:
    
   
    
    def __init__(self, project_id, start_date='2008-01-01', end_date=dt.datetime.now().strftime('%Y-%m-%d')):
        self.client = bigquery.Client(project=project_id)
        self.start_date = start_date
        self.end_date = end_date
        
    def get_transaction_metrics(self):
        
        query = f"""
        SELECT 
            DATE(block_timestamp) as date,
            COUNT(*) as transaction_count,
            COUNT(DISTINCT ARRAY_TO_STRING(inputs.addresses, ',')) as active_sending_addresses,
            SUM(output_value) / 1e8 as transaction_volume_btc
        FROM `bigquery-public-data.crypto_bitcoin.transactions`,
            UNNEST(inputs) as inputs
        WHERE block_timestamp >= '{self.start_date}'
          AND block_timestamp < '{self.end_date}'
        GROUP BY date
        ORDER BY date
        """
        
        df = self.client.query(query).to_dataframe()
        
        return df
    
    def get_hash_rate_proxy(self):
        
        query = f"""
        
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as block_count,
            COUNT(*)*100/144 as hash_rate_th_s
        FROM `bigquery-public-data.crypto_bitcoin.blocks`
        WHERE timestamp >= '{self.start_date}'
          AND timestamp < '{self.end_date}'
        GROUP BY date
        ORDER BY date
        """
        
        df = self.client.query(query).to_dataframe()
        
        return df[['date', 'hash_rate_th_s']]
    
    def get_circulating_supply(self):
        
        query = f"""
        WITH daily_supply AS (
            SELECT 
                DATE(block_timestamp) as date,
                SUM(CASE WHEN is_coinbase THEN output_value ELSE 0 END) / 1e8 as daily_mined
            FROM `bigquery-public-data.crypto_bitcoin.transactions`
            WHERE block_timestamp >= '2009-01-03'  -- Genesis block
              AND block_timestamp < '{self.end_date}'
            GROUP BY date
        )
        SELECT 
            date,
            SUM(daily_mined) OVER (ORDER BY date) as circulating_supply_btc
        FROM daily_supply
        WHERE date >= '{self.start_date}'
        ORDER BY date
        """
        
      
        df = self.client.query(query).to_dataframe()
      
        
        return df
    
    def estimate_exchange_flows_from_patterns(self, tx_data):
        df = tx_data.copy()
        
        df['tx_vol_ma7'] = df['transaction_volume_btc'].rolling(7, min_periods=1).mean()
        df['tx_vol_ma30'] = df['transaction_volume_btc'].rolling(30, min_periods=1).mean()
        df['addr_ma7'] = df['active_sending_addresses'].rolling(7, min_periods=1).mean()
        df['volume_spike'] = (df['transaction_volume_btc'].astype(np.float64) - df['tx_vol_ma30']) / df['tx_vol_ma30']
        df['volume_spike'] = df['volume_spike'].clip(lower=0)  

        vol_norm = (df['transaction_volume_btc']- df['transaction_volume_btc'].min()) / \
                   (df['transaction_volume_btc'].max() - df['transaction_volume_btc'].min())
        
        spike_norm = df['volume_spike'] / df['volume_spike'].max() if df['volume_spike'].max() > 0 else 0
        
        df['inflow_score'] = 0.6 * vol_norm.astype(np.float64) + 0.4 * spike_norm.astype(np.float64)
        
        df['exchange_inflow_btc'] = df['transaction_volume_btc'].astype(np.float64) * (0.15 + 0.15 * df['inflow_score'])
        
        df['below_avg_volume'] = (df['transaction_volume_btc'].astype(np.float64) < df['tx_vol_ma30']).astype(float)
        df['addr_stability'] = (df['active_sending_addresses'] / df['addr_ma7']).clip(0.5, 1.5) - 0.5
        
        df['outflow_score'] = 0.5 * df['below_avg_volume'] + 0.5 * df['addr_stability']
        
        df['exchange_outflow_btc'] = df['transaction_volume_btc'].astype(np.float64) * (0.10 + 0.10 * df['outflow_score'])
        
        
        return df
    
    def get_all_metrics(self):
        tx_metrics = self.get_transaction_metrics()
        hash_rate = self.get_hash_rate_proxy()
        supply = self.get_circulating_supply()
        
        complete = tx_metrics.merge(hash_rate, on='date', how='left')
        complete = complete.merge(supply, on='date', how='left')
        
        complete = self.estimate_exchange_flows_from_patterns(complete)
        
        final_metrics = complete[[
            'date',
            'active_sending_addresses',
            'hash_rate_th_s',
            'circulating_supply_btc',
            'transaction_volume_btc',
            'exchange_inflow_btc',
            'exchange_outflow_btc'
        ]].copy()
        
        final_metrics.columns = [
            'date',
            'Active Addresses',
            'Hash Rate',
            'Circulating Supply',
            'Transaction Volume',
            'Exchange Inflow',
            'Exchange Outflow'
        ]
        
        final_metrics = final_metrics.set_index('date')
        
        full_date_range = pd.date_range(
            start=final_metrics.index.min(),
            end=final_metrics.index.max(),
            freq='D'
        )
        
        final_metrics = final_metrics.reindex(full_date_range)
        
        final_metrics = final_metrics.fillna(method='ffill')
        
        missing = final_metrics.isnull().sum()
        
        print(f"  Date range: {final_metrics.index.min().date()} to {final_metrics.index.max().date()}")
        print(f"  Missing values per column:\n{missing}")
       
        
        return final_metrics



collector = CompleteDailyBitcoinMetrics(
    project_id=projectID
)

metrics = collector.get_all_metrics()

with open("onchainmetrics.csv", "w") as f:
    metrics.to_csv(f)