import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class DataPreprocessor:
    def __init__(self, start_time: str = "10:00", end_time: str = "15:30", n_levels: int = 5):
        """Initialize preprocessor with trading hours and number of levels."""
        self.start_time = pd.Timestamp(start_time).time()
        self.end_time = pd.Timestamp(end_time).time()
        self.n_levels = n_levels
        
        self.needed_columns = ['ts_event'] + [
            f'{side}_{type}_{i:02d}' 
            for side in ['bid', 'ask']
            for type in ['px', 'sz']
            for i in range(self.n_levels)
        ]
        
    def load_and_process_data(self, filepath: str) -> pd.DataFrame:
        """Load and process MBP data."""
        df = pd.read_parquet(filepath, columns=self.needed_columns)
        
        df['timestamp'] = pd.to_datetime(df['ts_event'], unit='ns')
        df = df.set_index('timestamp').sort_index()
        df = df.drop('ts_event', axis=1)
        
        mask = (df.index.time >= self.start_time) & (df.index.time <= self.end_time)
        df = df[mask]

        if df.empty:
            raise ValueError(f"No data remains after applying time mask from {self.start_time} to {self.end_time}. Check input data.")

        df = df.ffill()
        
        return df
    
    def compute_returns(self, df: pd.DataFrame, freq: str = '1min') -> pd.Series:
        """Compute log returns at specified frequency."""
        returns = np.log((df['bid_px_00'] + df['ask_px_00']).shift(-1) / 
                        (df['bid_px_00'] + df['ask_px_00']))
        
        if freq:
            returns = returns.resample(freq).sum()
            returns = returns[~returns.index.duplicated(keep='first')]
            
        return returns

    def process_multiple_stocks(self, 
                              data_dir: str, 
                              stocks: List[str], 
                              freq: str = '1min') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Process multiple stock data files."""
        returns = {}
        processed_data = {}
        
        for stock in stocks:
            print(f"Processing {stock}...")
            filepath = Path(data_dir) / f"{stock}_mbp10.parquet"
            df = self.load_and_process_data(filepath)
            processed_data[stock] = df
            returns[stock] = self.compute_returns(df, freq)
            
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df[~returns_df.index.duplicated(keep='first')]

        if returns_df.empty:
            raise ValueError("Returns DataFrame is empty. Ensure sufficient data exists for the specified frequency and timeframe.")
        
        return returns_df, processed_data
