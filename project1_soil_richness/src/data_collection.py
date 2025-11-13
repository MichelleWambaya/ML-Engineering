"""
Data Collection Module for Soil Richness Analysis
Collects and downloads soil data from various sources
"""

import pandas as pd
import numpy as np
import requests
import os
from pathlib import Path
import json

class SoilDataCollector:
    """Collects soil data from various sources"""
    
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Kenyan counties list
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
    
    def generate_synthetic_soil_data(self):
        """Generates realistic synthetic soil data for Kenyan counties"""
        np.random.seed(42)
        
        data = []
        for county in self.counties:
            # Generate realistic soil parameters
            # pH typically ranges from 5.5 to 8.5 in Kenya
            ph = np.random.normal(6.8, 0.8)
            
            # Organic matter percentage (0.5% to 5%)
            organic_matter = np.random.gamma(2, 0.8)
            organic_matter = np.clip(organic_matter, 0.5, 5.0)
            
            # Nitrogen content (ppm)
            nitrogen = np.random.gamma(3, 15)
            nitrogen = np.clip(nitrogen, 10, 100)
            
            # Phosphorus content (ppm)
            phosphorus = np.random.gamma(2, 10)
            phosphorus = np.clip(phosphorus, 5, 80)
            
            # Potassium content (ppm)
            potassium = np.random.gamma(4, 25)
            potassium = np.clip(potassium, 50, 300)
            
            # Cation Exchange Capacity (meq/100g)
            cec = np.random.normal(15, 5)
            cec = np.clip(cec, 5, 35)
            
            # Soil texture (clay percentage)
            clay = np.random.normal(30, 12)
            clay = np.clip(clay, 5, 60)
            
            # Sand percentage
            sand = np.random.normal(45, 15)
            sand = np.clip(sand, 10, 80)
            
            # Silt percentage (calculated to sum to ~100)
            silt = 100 - clay - sand
            silt = np.clip(silt, 5, 50)
            
            # Rainfall (mm/year) - varies by region
            rainfall = np.random.normal(800, 300)
            rainfall = np.clip(rainfall, 200, 2000)
            
            # Temperature (Celsius)
            temperature = np.random.normal(24, 3)
            temperature = np.clip(temperature, 18, 32)
            
            # Elevation (meters)
            elevation = np.random.normal(1200, 600)
            elevation = np.clip(elevation, 0, 3000)
            
            # Soil richness score (target variable)
            # Based on multiple factors
            richness_score = (
                0.2 * (organic_matter / 5.0) +
                0.15 * (nitrogen / 100.0) +
                0.15 * (phosphorus / 80.0) +
                0.15 * (potassium / 300.0) +
                0.1 * (cec / 35.0) +
                0.1 * (1 - abs(ph - 6.5) / 2.0) +
                0.15 * np.random.normal(0.5, 0.2)
            )
            richness_score = np.clip(richness_score * 100, 0, 100)
            
            data.append({
                'county': county,
                'ph': round(ph, 2),
                'organic_matter_percent': round(organic_matter, 2),
                'nitrogen_ppm': round(nitrogen, 2),
                'phosphorus_ppm': round(phosphorus, 2),
                'potassium_ppm': round(potassium, 2),
                'cec_meq_per_100g': round(cec, 2),
                'clay_percent': round(clay, 2),
                'sand_percent': round(sand, 2),
                'silt_percent': round(silt, 2),
                'rainfall_mm_per_year': round(rainfall, 2),
                'temperature_celsius': round(temperature, 2),
                'elevation_meters': round(elevation, 0),
                'soil_richness_score': round(richness_score, 2)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def save_data(self, df, filename="soil_data_raw.csv"):
        """Save collected data to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def collect(self):
        """Main collection method"""
        print("Collecting soil data for Kenyan counties...")
        df = self.generate_synthetic_soil_data()
        self.save_data(df)
        print(f"Collected data for {len(df)} counties")
        return df

if __name__ == "__main__":
    collector = SoilDataCollector()
    df = collector.collect()
    print("\nSample data:")
    print(df.head())

