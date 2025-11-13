"""
ETL Process Module for Mortality Rates Analysis
Extract, Transform, Load operations for mortality data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class MortalityETL:
    """ETL process for mortality data"""
    
    def __init__(self, raw_data_path="data/raw/mortality_data_raw.csv",
                 processed_data_path="data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def extract(self):
        """Extract data from source"""
        print("Extracting data...")
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        
        df = pd.read_csv(self.raw_data_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Extracted {len(df)} records")
        return df
    
    def transform(self, df):
        """Transform and clean data"""
        print("Transforming data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        df = df.fillna({
            'infant_deaths': 0,
            'child_deaths': 0,
            'adult_deaths': 0,
            'maternal_deaths': 0
        })
        
        # Calculate derived metrics
        df['year_month'] = df['date'].dt.to_period('M')
        df['quarter'] = df['date'].dt.quarter
        df['month_name'] = df['date'].dt.strftime('%B')
        
        # Calculate cumulative metrics
        df = df.sort_values(['county', 'date'])
        df['cumulative_deaths'] = df.groupby('county')['total_deaths'].cumsum()
        
        # Calculate moving averages
        df['mortality_rate_ma3'] = df.groupby('county')['overall_mortality_rate'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['mortality_rate_ma12'] = df.groupby('county')['overall_mortality_rate'].transform(
            lambda x: x.rolling(window=12, min_periods=1).mean()
        )
        
        # Calculate year-over-year change
        df['yoy_change'] = df.groupby(['county', 'month'])['overall_mortality_rate'].pct_change() * 100
        
        # Create mortality categories
        df['mortality_category'] = pd.cut(
            df['overall_mortality_rate'],
            bins=[0, 10, 15, 20, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Calculate death rate per age group
        df['infant_death_rate'] = (df['infant_deaths'] / df['population']) * 1000
        df['child_death_rate'] = (df['child_deaths'] / df['population']) * 1000
        df['adult_death_rate'] = (df['adult_deaths'] / df['population']) * 1000
        
        print(f"Transformed data: {len(df)} records, {len(df.columns)} columns")
        return df
    
    def create_aggregated_datasets(self, df):
        """Create aggregated datasets for analysis"""
        print("Creating aggregated datasets...")
        
        # Annual aggregation
        annual_df = df.groupby(['county', 'year']).agg({
            'total_deaths': 'sum',
            'infant_deaths': 'sum',
            'child_deaths': 'sum',
            'adult_deaths': 'sum',
            'maternal_deaths': 'sum',
            'population': 'mean',
            'overall_mortality_rate': 'mean',
            'infant_mortality_rate': 'mean',
            'hospitals_per_100k': 'mean',
            'doctors_per_100k': 'mean',
            'health_facilities': 'mean'
        }).reset_index()
        
        annual_df['annual_mortality_rate'] = (annual_df['total_deaths'] / annual_df['population']) * 1000
        
        # County-level summary
        county_summary = df.groupby('county').agg({
            'total_deaths': 'sum',
            'overall_mortality_rate': 'mean',
            'infant_mortality_rate': 'mean',
            'population': 'mean',
            'hospitals_per_100k': 'mean',
            'doctors_per_100k': 'mean',
            'health_facilities': 'mean',
            'is_urban': 'first'
        }).reset_index()
        
        county_summary['avg_annual_deaths'] = county_summary['total_deaths'] / df['year'].nunique()
        county_summary = county_summary.sort_values('overall_mortality_rate', ascending=False)
        
        # Monthly trends
        monthly_trends = df.groupby(['year', 'month']).agg({
            'total_deaths': 'sum',
            'overall_mortality_rate': 'mean',
            'population': 'sum'
        }).reset_index()
        
        monthly_trends['date'] = pd.to_datetime(
            monthly_trends['year'].astype(str) + '-' + 
            monthly_trends['month'].astype(str) + '-01'
        )
        
        return {
            'annual': annual_df,
            'county_summary': county_summary,
            'monthly_trends': monthly_trends
        }
    
    def load(self, df, aggregated_data):
        """Load processed data to storage"""
        print("Loading processed data...")
        
        # Save main dataset
        main_file = self.processed_data_path / "mortality_data_processed.csv"
        df.to_csv(main_file, index=False)
        print(f"Saved main dataset to {main_file}")
        
        # Save aggregated datasets
        annual_file = self.processed_data_path / "mortality_annual.csv"
        aggregated_data['annual'].to_csv(annual_file, index=False)
        print(f"Saved annual data to {annual_file}")
        
        county_file = self.processed_data_path / "mortality_county_summary.csv"
        aggregated_data['county_summary'].to_csv(county_file, index=False)
        print(f"Saved county summary to {county_file}")
        
        monthly_file = self.processed_data_path / "mortality_monthly_trends.csv"
        aggregated_data['monthly_trends'].to_csv(monthly_file, index=False)
        print(f"Saved monthly trends to {monthly_file}")
        
        return {
            'main': main_file,
            'annual': annual_file,
            'county_summary': county_file,
            'monthly_trends': monthly_file
        }
    
    def run_etl(self):
        """Run complete ETL process"""
        print("="*60)
        print("Starting ETL Process")
        print("="*60)
        
        # Extract
        df = self.extract()
        
        # Transform
        df = self.transform(df)
        
        # Create aggregated datasets
        aggregated_data = self.create_aggregated_datasets(df)
        
        # Load
        file_paths = self.load(df, aggregated_data)
        
        print("\n" + "="*60)
        print("ETL Process Complete")
        print("="*60)
        
        return df, aggregated_data, file_paths

if __name__ == "__main__":
    etl = MortalityETL()
    df, aggregated_data, file_paths = etl.run_etl()
    print("\nSample processed data:")
    print(df.head())

