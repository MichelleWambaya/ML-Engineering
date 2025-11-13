"""
Statistical Analysis Module for Mortality Rates
Performs comprehensive statistical analysis on mortality data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class MortalityStatisticalAnalysis:
    """Statistical analysis for mortality data"""
    
    def __init__(self, processed_data_path="data/processed/mortality_data_processed.csv"):
        self.processed_data_path = Path(processed_data_path)
        self.df = None
        self.results = {}
    
    def load_data(self):
        """Load processed data"""
        if not self.processed_data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {self.processed_data_path}")
        self.df = pd.read_csv(self.processed_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Loaded {len(self.df)} records")
        return self.df
    
    def descriptive_statistics(self):
        """Calculate descriptive statistics"""
        print("Calculating descriptive statistics...")
        
        numeric_cols = [
            'overall_mortality_rate', 'infant_mortality_rate',
            'child_mortality_rate', 'adult_mortality_rate',
            'maternal_mortality_rate', 'hospitals_per_100k',
            'doctors_per_100k'
        ]
        
        stats_dict = {}
        for col in numeric_cols:
            if col in self.df.columns:
                stats_dict[col] = {
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'q25': self.df[col].quantile(0.25),
                    'q75': self.df[col].quantile(0.75)
                }
        
        self.results['descriptive_stats'] = stats_dict
        return stats_dict
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("Performing correlation analysis...")
        
        numeric_cols = [
            'overall_mortality_rate', 'infant_mortality_rate',
            'hospitals_per_100k', 'doctors_per_100k',
            'health_facilities', 'population'
        ]
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        self.results['correlation_matrix'] = corr_matrix
        self.results['strong_correlations'] = strong_correlations
        
        return corr_matrix, strong_correlations
    
    def urban_rural_comparison(self):
        """Compare urban vs rural mortality rates"""
        print("Comparing urban vs rural mortality...")
        
        comparison = self.df.groupby('is_urban').agg({
            'overall_mortality_rate': ['mean', 'std', 'median'],
            'infant_mortality_rate': ['mean', 'std', 'median'],
            'hospitals_per_100k': 'mean',
            'doctors_per_100k': 'mean'
        }).round(2)
        
        # Statistical test
        urban_rates = self.df[self.df['is_urban'] == True]['overall_mortality_rate']
        rural_rates = self.df[self.df['is_urban'] == False]['overall_mortality_rate']
        
        t_stat, p_value = stats.ttest_ind(urban_rates, rural_rates)
        
        self.results['urban_rural_comparison'] = {
            'summary': comparison,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return comparison, t_stat, p_value
    
    def county_ranking(self):
        """Rank counties by mortality rates"""
        print("Ranking counties...")
        
        county_stats = self.df.groupby('county').agg({
            'overall_mortality_rate': 'mean',
            'infant_mortality_rate': 'mean',
            'total_deaths': 'sum',
            'population': 'mean'
        }).reset_index()
        
        county_stats['mortality_rank'] = county_stats['overall_mortality_rate'].rank(ascending=True)
        county_stats = county_stats.sort_values('overall_mortality_rate')
        
        # Categorize counties
        county_stats['mortality_category'] = pd.cut(
            county_stats['overall_mortality_rate'],
            bins=[0, 10, 15, 20, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        self.results['county_ranking'] = county_stats
        return county_stats
    
    def seasonal_analysis(self):
        """Analyze seasonal patterns"""
        print("Analyzing seasonal patterns...")
        
        seasonal_stats = self.df.groupby('month').agg({
            'overall_mortality_rate': ['mean', 'std'],
            'total_deaths': 'sum'
        }).reset_index()
        
        seasonal_stats.columns = ['month', 'avg_mortality_rate', 'std_mortality_rate', 'total_deaths']
        seasonal_stats = seasonal_stats.sort_values('month')
        
        # Test for seasonality
        monthly_means = self.df.groupby('month')['overall_mortality_rate'].mean()
        f_stat, p_value = stats.f_oneway(*[group['overall_mortality_rate'].values 
                                          for name, group in self.df.groupby('month')])
        
        self.results['seasonal_analysis'] = {
            'monthly_stats': seasonal_stats,
            'f_statistic': f_stat,
            'p_value': p_value,
            'has_seasonality': p_value < 0.05
        }
        
        return seasonal_stats, f_stat, p_value
    
    def trend_analysis(self):
        """Analyze temporal trends"""
        print("Analyzing trends...")
        
        # Yearly trends
        yearly_trends = self.df.groupby('year').agg({
            'overall_mortality_rate': 'mean',
            'total_deaths': 'sum'
        }).reset_index()
        
        # Calculate trend
        x = yearly_trends['year'].values
        y = yearly_trends['overall_mortality_rate'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        trend_direction = "increasing" if slope > 0 else "decreasing"
        
        self.results['trend_analysis'] = {
            'yearly_trends': yearly_trends,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'trend_direction': trend_direction,
            'significant': p_value < 0.05
        }
        
        return yearly_trends, slope, r_value**2, p_value
    
    def risk_factor_analysis(self):
        """Identify risk factors for high mortality"""
        print("Analyzing risk factors...")
        
        # Create high mortality flag
        median_mortality = self.df['overall_mortality_rate'].median()
        self.df['high_mortality'] = self.df['overall_mortality_rate'] > median_mortality
        
        # Compare high vs low mortality groups
        risk_factors = self.df.groupby('high_mortality').agg({
            'hospitals_per_100k': 'mean',
            'doctors_per_100k': 'mean',
            'health_facilities': 'mean',
            'population': 'mean',
            'is_urban': lambda x: (x == True).sum() / len(x)
        }).round(2)
        
        # Statistical tests
        high_mort_hospitals = self.df[self.df['high_mortality'] == True]['hospitals_per_100k']
        low_mort_hospitals = self.df[self.df['high_mortality'] == False]['hospitals_per_100k']
        
        t_stat, p_value = stats.ttest_ind(high_mort_hospitals, low_mort_hospitals)
        
        self.results['risk_factors'] = {
            'comparison': risk_factors,
            'hospital_t_test': {'t_statistic': t_stat, 'p_value': p_value}
        }
        
        return risk_factors
    
    def run_all_analyses(self):
        """Run all statistical analyses"""
        print("="*60)
        print("Starting Statistical Analysis")
        print("="*60)
        
        self.load_data()
        
        self.descriptive_statistics()
        self.correlation_analysis()
        self.urban_rural_comparison()
        self.county_ranking()
        self.seasonal_analysis()
        self.trend_analysis()
        self.risk_factor_analysis()
        
        print("\n" + "="*60)
        print("Statistical Analysis Complete")
        print("="*60)
        
        # Print summary
        print("\nKey Findings:")
        print(f"- Average mortality rate: {self.results['descriptive_stats']['overall_mortality_rate']['mean']:.2f}")
        print(f"- Trend: {self.results['trend_analysis']['trend_direction']}")
        print(f"- Seasonality detected: {self.results['seasonal_analysis']['has_seasonality']}")
        
        return self.results

if __name__ == "__main__":
    analyzer = MortalityStatisticalAnalysis()
    results = analyzer.run_all_analyses()

