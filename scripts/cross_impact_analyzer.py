import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression, Lasso

class CrossImpactAnalyzer:
    def __init__(self, alpha: float = 0.01, n_levels: int = 5):
        """
        Initialize analyzer with LASSO regularization parameter and number of levels.
        
        Args:
            alpha: LASSO regularization parameter
            n_levels: Number of order book levels to analyze
        """
        self.alpha = alpha
        self.n_levels = n_levels
        
    def _fit_single_stock_multi_level_model(self, 
                                          returns: pd.Series, 
                                          ofis: pd.DataFrame) -> Dict:
        """Fit single stock model using OLS with multiple levels."""
        model = LinearRegression()
        X = ofis.values
        y = returns.values
        
        model.fit(X, y)
        
        return {
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=ofis.columns),
            'r2': model.score(X, y)
        }
        
    def _fit_cross_impact_multi_level_model(self,
                                          returns: pd.Series,
                                          ofis: Dict[str, pd.DataFrame]) -> Dict:
        """Fit cross-impact model using LASSO with multiple levels."""
        # Combine all stocks' OFIs into one matrix
        combined_ofis = pd.concat([df for df in ofis.values()], axis=1)
        
        model = Lasso(alpha=self.alpha)
        X = combined_ofis.values
        y = returns.values
        
        model.fit(X, y)
        
        return {
            'intercept': model.intercept_,
            'coefficients': pd.Series(model.coef_, index=combined_ofis.columns),
            'r2': model.score(X, y)
        }

    def compute_contemporaneous_impact(self,
                                     returns: pd.DataFrame,
                                     multi_level_ofis: Dict[str, pd.DataFrame],
                                     integrated_ofis: pd.DataFrame) -> Dict:
        """
        Compute contemporaneous impact models with multiple OFI levels.
        
        Args:
            returns: DataFrame of stock returns
            multi_level_ofis: Dict of DataFrames containing multi-level OFIs for each stock
            integrated_ofis: DataFrame of integrated OFIs
        """
        results = {}
        
        for stock in returns.columns:
            stock_results = {}
            
            stock_results['PI_ML'] = self._fit_single_stock_multi_level_model(
                returns[stock],
                multi_level_ofis[stock]
            )
            
            stock_results['PI_I'] = self._fit_single_stock_multi_level_model(
                returns[stock],
                integrated_ofis[[stock]]
            )
            
            stock_results['CI_ML'] = self._fit_cross_impact_multi_level_model(
                returns[stock],
                multi_level_ofis
            )
            
            stock_results['CI_I'] = self._fit_cross_impact_multi_level_model(
                returns[stock],
                {k: integrated_ofis[[k]] for k in integrated_ofis.columns}
            )
            
            results[stock] = stock_results
            
        return results

    def compute_predictive_impact(self, returns: pd.DataFrame, 
                            multi_level_ofis: Dict[str, pd.DataFrame],
                            integrated_ofis: pd.DataFrame,
                            lags: List[int] = [1, 2, 3, 5, 10, 20, 30]) -> Dict:
        """
        Compute predictive cross-impact following equations (12-15) in paper.
        
        Args:
            returns: DataFrame of stock returns
            multi_level_ofis: Dict of DataFrames containing multi-level OFIs for each stock
            integrated_ofis: DataFrame of integrated OFIs
            lags: List of prediction horizons to analyze
        """
        results = {}
        
        for stock in returns.columns:
            stock_results = {}
            
            for f in lags:
                X_ml = multi_level_ofis[stock].shift(f)
                y = returns[stock]
                
                model_fpi_ml = LinearRegression()
                mask = ~X_ml.isna().any(axis=1)
                model_fpi_ml.fit(X_ml[mask], y[mask])
                
                X_i = integrated_ofis[[stock]].shift(f)
                
                model_fpi_i = LinearRegression()
                mask = ~X_ml.isna().any(axis=1)
                model_fpi_i.fit(X_i[mask], y[mask])
                
                X_ml_cross = pd.concat([df.shift(f) for df in multi_level_ofis.values()], axis=1)
                
                model_fci_ml = Lasso(alpha=self.alpha)
                mask = ~X_ml_cross.isna().any(axis=1)
                model_fci_ml.fit(X_ml_cross[mask], y[mask])
                
                X_i_cross = integrated_ofis.shift(f)
                
                model_fci_i = Lasso(alpha=self.alpha)
                mask = ~X_i_cross.isna().any(axis=1)
                model_fci_i.fit(X_i_cross[mask], y[mask])
                
                stock_results[f] = {
                    'FPI_ML': {
                        'coefficients': pd.Series(model_fpi_ml.coef_, 
                                            index=multi_level_ofis[stock].columns),
                        'intercept': model_fpi_ml.intercept_,
                        'r2': model_fpi_ml.score(X_ml[mask], y[mask])
                    },
                    'FPI_I': {
                        'coefficient': model_fpi_i.coef_[0],
                        'intercept': model_fpi_i.intercept_,
                        'r2': model_fpi_i.score(X_i[mask], y[mask])
                    },
                    'FCI_ML': {
                        'coefficients': pd.Series(model_fci_ml.coef_,
                                            index=X_ml_cross.columns),
                        'intercept': model_fci_ml.intercept_,
                        'r2': model_fci_ml.score(X_ml_cross[mask], y[mask])
                    },
                    'FCI_I': {
                        'coefficients': pd.Series(model_fci_i.coef_,
                                            index=X_i_cross.columns),
                        'intercept': model_fci_i.intercept_,
                        'r2': model_fci_i.score(X_i_cross[mask], y[mask])
                    }
                }
            
            results[stock] = stock_results
        
        return results