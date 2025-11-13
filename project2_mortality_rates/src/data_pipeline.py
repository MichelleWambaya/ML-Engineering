"""
Data Pipeline Module for Mortality Rates Analysis
Handles data collection, validation, and initial processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

class MortalityDataPipeline:
    """Data pipeline for mortality rate data"""
    
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.counties = [
            "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika",
            "Malindi", "Kitale", "Garissa", "Kakamega", "Nyeri", "Embu",
            "Meru", "Machakos", "Kitui", "Makueni", "Taita Taveta", "Kwale",
            "Kilifi", "Lamu", "Tana River", "Wajir", "Mandera", "Marsabit",
            "Isiolo", "Samburu", "Turkana", "West Pokot", "Uasin Gishu",
            "Trans Nzoia", "Baringo", "Laikipia", "Nandi", "Kericho", "Bomet",
            "Narok", "Kajiado", "Nyandarua", "Murang'a", "Kiambu", "Kirinyaga",
            "Nyamira", "Kisii", "Homa Bay", "Migori", "Siaya", "Busia",
            "Vihiga", "Bungoma", "Trans Mara"
        ]
    
    def generate_mortality_data(self, years=5):
        """Generate synthetic mortality data for Kenyan counties"""
        np.random.seed(42)
        
        data = []
        start_date = datetime(2019, 1, 1)
        
        for county in self.counties:
            # Base mortality rates vary by county characteristics
            # Urban counties typically have lower mortality
            is_urban = county in ["Nairobi", "Mombasa", "Kisumu", "Nakuru"]
            
            # Base rates per 1000 population
            if is_urban:
                base_infant_mortality = np.random.uniform(20, 35)
                base_child_mortality = np.random.uniform(5, 12)
                base_adult_mortality = np.random.uniform(8, 15)
                base_maternal_mortality = np.random.uniform(200, 400)
            else:
                base_infant_mortality = np.random.uniform(35, 65)
                base_child_mortality = np.random.uniform(10, 25)
                base_adult_mortality = np.random.uniform(12, 25)
                base_maternal_mortality = np.random.uniform(400, 800)
            
            # Population estimates (in thousands)
            population = np.random.uniform(100, 5000)
            
            # Healthcare indicators
            hospitals_per_100k = np.random.uniform(0.5, 3.0) if is_urban else np.random.uniform(0.2, 1.5)
            doctors_per_100k = np.random.uniform(10, 50) if is_urban else np.random.uniform(2, 15)
            health_facilities = int(population * hospitals_per_100k / 100)
            
            # Generate monthly data for the past years
            for year in range(years):
                for month in range(1, 13):
                    date = datetime(2019 + year, month, 1)
                    
                    # Seasonal variation (higher in certain months)
                    seasonal_factor = 1.0
                    if month in [1, 2, 6, 7]:  # Dry seasons
                        seasonal_factor = 1.15
                    elif month in [3, 4, 5]:  # Rainy season
                        seasonal_factor = 1.05
                    
                    # Random variation
                    random_factor = np.random.uniform(0.85, 1.15)
                    
                    # Calculate deaths
                    infant_deaths = int((base_infant_mortality * population / 1000 / 12) * 
                                      seasonal_factor * random_factor)
                    child_deaths = int((base_child_mortality * population / 1000 / 12) * 
                                     seasonal_factor * random_factor)
                    adult_deaths = int((base_adult_mortality * population / 1000 / 12) * 
                                     seasonal_factor * random_factor)
                    maternal_deaths = int((base_maternal_mortality * population / 1000 / 12 / 1000) * 
                                        seasonal_factor * random_factor)
                    
                    total_deaths = infant_deaths + child_deaths + adult_deaths
                    
                    # Calculate rates
                    infant_mortality_rate = (infant_deaths / population) * 1000 * 12
                    child_mortality_rate = (child_deaths / population) * 1000 * 12
                    adult_mortality_rate = (adult_deaths / population) * 1000 * 12
                    maternal_mortality_rate = (maternal_deaths / population) * 1000 * 12 * 1000
                    overall_mortality_rate = (total_deaths / population) * 1000 * 12
                    
                    data.append({
                        'county': county,
                        'year': 2019 + year,
                        'month': month,
                        'date': date.strftime('%Y-%m-%d'),
                        'population': round(population, 2),
                        'infant_deaths': infant_deaths,
                        'child_deaths': child_deaths,
                        'adult_deaths': adult_deaths,
                        'maternal_deaths': maternal_deaths,
                        'total_deaths': total_deaths,
                        'infant_mortality_rate': round(infant_mortality_rate, 2),
                        'child_mortality_rate': round(child_mortality_rate, 2),
                        'adult_mortality_rate': round(adult_mortality_rate, 2),
                        'maternal_mortality_rate': round(maternal_mortality_rate, 2),
                        'overall_mortality_rate': round(overall_mortality_rate, 2),
                        'hospitals_per_100k': round(hospitals_per_100k, 2),
                        'doctors_per_100k': round(doctors_per_100k, 2),
                        'health_facilities': health_facilities,
                        'is_urban': is_urban
                    })
        
        df = pd.DataFrame(data)
        return df
    
    def validate_data(self, df):
        """Validate mortality data"""
        print("Validating data...")
        
        # Check for negative values
        numeric_cols = ['infant_deaths', 'child_deaths', 'adult_deaths', 
                       'total_deaths', 'population']
        for col in numeric_cols:
            if (df[col] < 0).any():
                print(f"Warning: Negative values found in {col}")
                df[col] = df[col].clip(lower=0)
        
        # Check mortality rate ranges
        df['infant_mortality_rate'] = df['infant_mortality_rate'].clip(0, 200)
        df['overall_mortality_rate'] = df['overall_mortality_rate'].clip(0, 50)
        
        # Validate date format
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Validated {len(df)} records")
        return df
    
    def save_data(self, df, filename="mortality_data_raw.csv"):
        """Save data to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def run_pipeline(self):
        """Run the complete data pipeline"""
        print("Starting mortality data pipeline...")
        df = self.generate_mortality_data(years=5)
        df = self.validate_data(df)
        self.save_data(df)
        print(f"Pipeline complete: {len(df)} records processed")
        return df

if __name__ == "__main__":
    pipeline = MortalityDataPipeline()
    df = pipeline.run_pipeline()
    print("\nSample data:")
    print(df.head(10))

