```markdown
# Cross-Impact Analysis of Order Flow Imbalance (OFI)

Cross-Impact Analysis of Order Flow Imbalance (OFI) is a comprehensive market microstructure analysis platform that implements the methodology from "Cross-Impact of Order Flow Imbalance in Equity Markets" (Cont et al., 2023) for analyzing order flow dynamics and price formation.

## 🚀 Features

- **Order Flow Analysis**
  - Multi-level OFI calculation
  - PCA-based OFI integration
  - Level-specific impact analysis
  - Book depth normalization

- **Cross-Impact Analysis**
  - Contemporaneous impact modeling
  - Predictive cross-impact analysis
  - LASSO regularization for sparse estimation
  - Sector-level aggregation

- **Visualization & Results**
  - Correlation heatmaps
  - Impact analysis plots
  - Multi-level impact visualizations
  - Sector relationship networks

## 🛠 Technology Stack

- **Core Analysis**
  - Python 3.8+
  - NumPy/Pandas
  - scikit-learn
  - Matplotlib/Seaborn

- **Data Processing**
  - Databento API
  - Parquet file format
  - PCA dimensionality reduction
  - LASSO regression

- **Visualization**
  - Matplotlib
  - Seaborn
  - Network graphs
  - Correlation heatmaps

## 📋 Prerequisites

- Python 3.8+
- Databento API access
- 16GB+ RAM recommended
- Storage for market data

## 🔧 Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Databento access in scripts/get_mbp10_data.py:
```python
client = db.Historical('YOUR-DATABENTO-KEY')
```

## 🏗 Project Structure

```
project/
├── data/                    # Market data files
│   └── *.parquet           # MBP-10 data files
├── notebooks/              # Analysis notebooks
├── scripts/
│   ├── cross_impact_analyzer.py  # Cross-impact analysis
│   ├── data_preprocessor.py      # Data preprocessing
│   ├── get_mbp10_data.py        # Data downloader
│   └── ofi_calculator.py        # OFI computation
├── results/
│   ├── correlations/            # OFI correlations
│   ├── impact_analysis/         # Impact studies
│   └── multi_level_impacts/     # Level analysis
└── requirements.txt
```

## 🔍 Core Components

### Data Preprocessing
```python
from scripts.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(n_levels=5)
returns_df, processed_data = preprocessor.process_multiple_stocks(
    data_dir='data',
    stocks=['AAPL', 'MSFT', 'NVDA']
)
```

### OFI Calculation
```python
from scripts.ofi_calculator import OFICalculator

calculator = OFICalculator(n_levels=5)
integrated_ofis, multi_level_ofis = calculator.process_multiple_stocks(
    processed_data
)
```

### Impact Analysis
```python
from scripts.cross_impact_analyzer import CrossImpactAnalyzer

analyzer = CrossImpactAnalyzer(alpha=0.01)
results = analyzer.compute_contemporaneous_impact(
    returns_df,
    multi_level_ofis,
    integrated_ofis
)
```

## 📊 Generated Results

- **correlations/**
  - multi_level_ofi_correlations.png
  - level_correlation_structure.png

- **impact_analysis/**
  - contemporaneous_impact_analysis.png
  - predictive_impact_analysis.png
  - cross_impact_evolution.png

- **multi_level_impacts/**
  - Individual stock impact profiles
  - Level-specific visualizations

## 📈 Utilized Markets

- **Tech Sector**
  - AAPL, MSFT, NVDA
- **Healthcare**
  - AMGN, GILD
- **Consumer**
  - TSLA, PEP
- **Financials**
  - JPM, V
- **Energy**
  - XOM

## 🔍 Analysis Parameters

- Trading hours: 10:00-15:30 ET
- Order book depth: 5 levels
- LASSO alpha: 0.01
- Prediction horizons: [1, 2, 3, 5, 10, 20, 30] minutes
- Default frequency: 1-minute intervals

## 🙏 Acknowledgments

- Cont et al. for the OFI methodology
- Databento for market data access
```