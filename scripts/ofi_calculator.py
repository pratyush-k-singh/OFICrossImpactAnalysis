import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA

class OFICalculator:
    def __init__(self, n_levels: int = 5):
        """Initialize calculator for OFI metrics."""
        self.n_levels = n_levels

    def _compute_level_ofi(self, df: pd.DataFrame, level: int) -> pd.Series:
        """Compute OFI for a specific level following paper methodology."""            
        level_str = f"{level:02d}"
        
        bid_px = df[f'bid_px_{level_str}']
        bid_sz = df[f'bid_sz_{level_str}']
        
        bid_flow = np.where(
            bid_px > bid_px.shift(1), bid_sz,
            np.where(
                bid_px == bid_px.shift(1),
                bid_sz - bid_sz.shift(1),
                -bid_sz
            )
        )
        
        ask_px = df[f'ask_px_{level_str}']
        ask_sz = df[f'ask_sz_{level_str}']
        
        ask_flow = np.where(
            ask_px > ask_px.shift(1), -ask_sz,
            np.where(
                ask_px == ask_px.shift(1),
                ask_sz.shift(1) - ask_sz,
                ask_sz
            )
        )
        
        return pd.Series(bid_flow + ask_flow, index=df.index)

    def compute_multi_level_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OFI for all levels."""
        ofis = {}
        
        if df.empty:
            raise ValueError("Input DataFrame is empty. Ensure sufficient data is provided.")

        for level in range(self.n_levels):
            ofis[f'level_{level}'] = self._compute_level_ofi(df, level)
            
        return pd.DataFrame(ofis)

    def compute_integrated_ofi(self, df: pd.DataFrame, freq: str = '1min') -> pd.Series:
        """Compute integrated OFI with proper scaling."""
        multi_level = self.compute_multi_level_ofi(df)
        
        book_depth = np.mean([
            np.mean([df[f'bid_sz_{i:02d}'].mean() + df[f'ask_sz_{i:02d}'].mean()])
            for i in range(self.n_levels)
        ])
        
        scaled_ofi = multi_level / book_depth

        if scaled_ofi.empty or scaled_ofi.shape[0] < self.n_levels:
            raise ValueError("Not enough data for PCA. Ensure sufficient rows are available for OFI computation.")
        
        pca = PCA(n_components=1)
        w1 = pca.fit(scaled_ofi).components_[0]
        w1_normalized = w1 / np.sum(np.abs(w1))
        
        integrated_ofi = scaled_ofi @ w1_normalized
        
        if freq:
            integrated_ofi = integrated_ofi.resample(freq).sum()
            integrated_ofi = integrated_ofi[~integrated_ofi.index.duplicated(keep='first')]
        
        return integrated_ofi

    def process_multiple_stocks(self, processed_data: Dict[str, pd.DataFrame], 
                              freq: str = '1min') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Process multiple stocks and return OFIs."""
        integrated_ofis = {}
        multi_level_ofis = {}
        
        for stock, df in processed_data.items():
            print(f"Computing OFIs for {stock}...")
            multi_level = self.compute_multi_level_ofi(df)
            resampled_multi_level = multi_level.resample(freq).sum().dropna()
            multi_level_ofis[stock] = resampled_multi_level
            integrated_ofis[stock] = self.compute_integrated_ofi(df, freq)
        
        integrated_df = pd.DataFrame(integrated_ofis)
        integrated_df = integrated_df[~integrated_df.index.duplicated(keep='first')]
        
        return integrated_df, multi_level_ofis
