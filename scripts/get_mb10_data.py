import databento as db
from datetime import datetime
from tqdm import tqdm
import os

def download_mbp10_data(symbols, start_date, end_date, output_dir=None):
    """Download MBP-10 data from Databento for given symbols."""
    
    client = db.Historical('DATABENTO-KEY')

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    os.makedirs(output_dir, exist_ok=True)
    
    symbol_map = {
        'AAPL': {'symbol': 'AAPL', 'dataset': 'XNAS.ITCH'},
        'MSFT': {'symbol': 'MSFT', 'dataset': 'XNAS.ITCH'},
        'NVDA': {'symbol': 'NVDA', 'dataset': 'XNAS.ITCH'},
        'AMGN': {'symbol': 'AMGN', 'dataset': 'XNAS.ITCH'},
        'GILD': {'symbol': 'GILD', 'dataset': 'XNAS.ITCH'},
        'TSLA': {'symbol': 'TSLA', 'dataset': 'XNAS.ITCH'},
        'PEP':  {'symbol': 'PEP',  'dataset': 'XNAS.ITCH'},
        'JPM':  {'symbol': 'JPM',  'dataset': 'XNAS.ITCH'},
        'V':    {'symbol': 'V',    'dataset': 'XNAS.ITCH'},
        'XOM':  {'symbol': 'XOM',  'dataset': 'XNAS.ITCH'}
    }
    
    for symbol in tqdm(symbols, desc="Downloading data"):
        if symbol not in symbol_map:
            print(f"Symbol {symbol} not found in symbol_map. Skipping.")
            continue
        
        info = symbol_map[symbol]
        print(f"\nDownloading data for {symbol}")
        
        try:
            df = client.timeseries.get_range(
                dataset=info['dataset'],
                symbols=[info['symbol']],
                schema='mbp-10',
                start=start_date,
                end=end_date,
                limit=None
            )
            
            output_file = f"{output_dir}/{symbol}_mbp10.parquet"
            df.to_parquet(output_file)
            print(f"Saved to {output_file}")
            
        except Exception as e:
            print(f"Error downloading {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMGN', 'GILD', 
               'TSLA', 'PEP', 'JPM', 'V', 'XOM']
    
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2023, 12, 31)
    output_dir = '../data'
    
    download_mbp10_data(symbols, start_date, end_date, output_dir)
