"""
Time Series Analysis Module for Mortality Rates
Forecasting and time series modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class MortalityTimeSeriesAnalysis:
    """Time series analysis for mortality data"""
    
    def __init__(self, processed_data_path="data/processed/mortality_data_processed.csv"):
        self.processed_data_path = Path(processed_data_path)
        self.df = None
        self.models = {}
        self.forecasts = {}
    
    def load_data(self):
        """Load processed data"""
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.processed_data_path}")
        self.df = pd.read_csv(self.processed_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def prepare_time_series(self, county=None):
        """Prepare time series data"""
        if county:
            ts_data = self.df[self.df['county'] == county].copy()
        else:
            ts_data = self.df.groupby('date')['overall_mortality_rate'].mean().reset_index()
            ts_data['county'] = 'All Counties'
        
        ts_data = ts_data.sort_values('date')
        ts_data = ts_data.set_index('date')
        
        return ts_data
    
    def decompose_time_series(self, ts_data, county_name="All Counties"):
        """Decompose time series into trend, seasonal, and residual"""
        print(f"Decomposing time series for {county_name}...")
        
        # Resample to monthly if needed
        if isinstance(ts_data, pd.Series):
            monthly_ts = ts_data.resample('M').mean()
        else:
            monthly_ts = ts_data['overall_mortality_rate'].resample('M').mean()
        
        # Remove NaN values
        monthly_ts = monthly_ts.dropna()
        
        if len(monthly_ts) < 24:
            print(f"Not enough data for decomposition (need at least 24 months)")
            return None
        
        try:
            decomposition = seasonal_decompose(
                monthly_ts,
                model='additive',
                period=12
            )
            
            return {
                'original': monthly_ts,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
        except Exception as e:
            print(f"Decomposition failed: {e}")
            return None
    
    def test_stationarity(self, ts):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(ts.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def fit_arima_model(self, ts, order=(1, 1, 1)):
        """Fit ARIMA model"""
        print(f"Fitting ARIMA{order} model...")
        
        try:
            model = ARIMA(ts.dropna(), order=order)
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=12)
            forecast_index = pd.date_range(
                start=ts.index[-1] + pd.DateOffset(months=1),
                periods=12,
                freq='M'
            )
            
            return {
                'model': fitted_model,
                'forecast': forecast,
                'forecast_index': forecast_index,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            return None
    
    def evaluate_forecast(self, actual, predicted):
        """Evaluate forecast accuracy"""
        # Align indices
        common_index = actual.index.intersection(predicted.index)
        if len(common_index) == 0:
            return None
        
        actual_aligned = actual.loc[common_index]
        predicted_aligned = predicted.loc[common_index]
        
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def forecast_county_mortality(self, county, forecast_months=12):
        """Forecast mortality for a specific county"""
        print(f"Forecasting mortality for {county}...")
        
        ts_data = self.prepare_time_series(county=county)
        monthly_ts = ts_data['overall_mortality_rate'].resample('M').mean().dropna()
        
        if len(monthly_ts) < 24:
            print(f"Insufficient data for {county}")
            return None
        
        # Test stationarity
        stationarity = self.test_stationarity(monthly_ts)
        
        # Fit ARIMA model
        arima_result = self.fit_arima_model(monthly_ts)
        
        if arima_result:
            self.models[county] = arima_result
            self.forecasts[county] = {
                'forecast': arima_result['forecast'],
                'forecast_index': arima_result['forecast_index'],
                'stationarity': stationarity
            }
        
        return arima_result
    
    def forecast_all_counties(self, top_n=10):
        """Forecast for top N counties by population"""
        print("Forecasting for multiple counties...")
        
        # Get top counties by average population
        county_pop = self.df.groupby('county')['population'].mean().sort_values(ascending=False)
        top_counties = county_pop.head(top_n).index.tolist()
        
        results = {}
        for county in top_counties:
            result = self.forecast_county_mortality(county)
            if result:
                results[county] = result
        
        return results
    
    def plot_forecast(self, county, save_path=None):
        """Plot forecast for a county"""
        if county not in self.forecasts:
            print(f"No forecast available for {county}")
            return
        
        ts_data = self.prepare_time_series(county=county)
        monthly_ts = ts_data['overall_mortality_rate'].resample('M').mean().dropna()
        
        forecast_data = self.forecasts[county]
        
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_ts.index, monthly_ts.values, label='Historical', linewidth=2)
        plt.plot(forecast_data['forecast_index'], forecast_data['forecast'].values,
                label='Forecast', linewidth=2, linestyle='--', color='red')
        plt.fill_between(forecast_data['forecast_index'],
                        forecast_data['forecast'].values * 0.9,
                        forecast_data['forecast'].values * 1.1,
                        alpha=0.2, color='red', label='Confidence Interval')
        plt.xlabel('Date')
        plt.ylabel('Mortality Rate')
        plt.title(f'Mortality Rate Forecast for {county}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self):
        """Run complete time series analysis"""
        print("="*60)
        print("Starting Time Series Analysis")
        print("="*60)
        
        self.load_data()
        
        # Analyze overall trend
        overall_ts = self.prepare_time_series()
        decomposition = self.decompose_time_series(overall_ts['overall_mortality_rate'], "All Counties")
        
        # Forecast for top counties
        forecasts = self.forecast_all_counties(top_n=5)
        
        print(f"\nCompleted forecasts for {len(forecasts)} counties")
        
        return {
            'decomposition': decomposition,
            'forecasts': forecasts,
            'models': self.models
        }

if __name__ == "__main__":
    analyzer = MortalityTimeSeriesAnalysis()
    results = analyzer.run_analysis()

