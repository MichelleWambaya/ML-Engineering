"""
Data Ingestion Module for Literacy Rates Analysis
Collects and processes literacy and education data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class LiteracyDataIngestion:
    """Data ingestion for literacy rate analysis"""
    
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
    
    def generate_literacy_data(self, years=5):
        """Generate synthetic literacy data for Kenyan counties"""
        np.random.seed(42)
        
        data = []
        start_year = 2019
        
        for county in self.counties:
            # County characteristics
            is_urban = county in ["Nairobi", "Mombasa", "Kisumu", "Nakuru"]
            
            # Base literacy rates (percentage)
            if is_urban:
                base_literacy_rate = np.random.uniform(75, 90)
                base_male_literacy = np.random.uniform(80, 95)
                base_female_literacy = np.random.uniform(70, 88)
            else:
                base_literacy_rate = np.random.uniform(50, 80)
                base_male_literacy = np.random.uniform(55, 85)
                base_female_literacy = np.random.uniform(45, 75)
            
            # Population (in thousands)
            population = np.random.uniform(200, 6000)
            
            # Education infrastructure
            primary_schools = int(population * np.random.uniform(0.8, 1.5))
            secondary_schools = int(population * np.random.uniform(0.1, 0.3))
            teachers_per_school = np.random.uniform(8, 25)
            total_teachers = int((primary_schools + secondary_schools) * teachers_per_school)
            
            # Student-teacher ratio
            student_teacher_ratio = np.random.uniform(25, 50)
            total_students = int(total_teachers * student_teacher_ratio)
            
            # Enrollment rates
            primary_enrollment_rate = np.random.uniform(85, 98)
            secondary_enrollment_rate = np.random.uniform(40, 75)
            
            # Economic indicators (affect literacy)
            gdp_per_capita = np.random.uniform(500, 3000)  # USD
            poverty_rate = np.random.uniform(15, 60)  # percentage
            
            # Infrastructure
            libraries = int(population * np.random.uniform(0.01, 0.05))
            internet_penetration = np.random.uniform(20, 80)  # percentage
            
            # Generate yearly data
            for year in range(years):
                year_val = start_year + year
                
                # Trend: gradual improvement over time
                improvement_factor = 1 + (year * 0.01)  # 1% improvement per year
                random_variation = np.random.uniform(0.95, 1.05)
                
                # Calculate literacy rates
                literacy_rate = min(100, base_literacy_rate * improvement_factor * random_variation)
                male_literacy = min(100, base_male_literacy * improvement_factor * random_variation)
                female_literacy = min(100, base_female_literacy * improvement_factor * random_variation)
                
                # Gender gap
                gender_gap = male_literacy - female_literacy
                
                # Age group literacy
                youth_literacy = literacy_rate + np.random.uniform(-5, 5)
                adult_literacy = literacy_rate + np.random.uniform(-10, 5)
                
                # Education quality indicators
                dropout_rate = np.random.uniform(5, 25)
                completion_rate = 100 - dropout_rate
                
                data.append({
                    'county': county,
                    'year': year_val,
                    'population': round(population, 2),
                    'literacy_rate': round(literacy_rate, 2),
                    'male_literacy_rate': round(male_literacy, 2),
                    'female_literacy_rate': round(female_literacy, 2),
                    'gender_gap': round(gender_gap, 2),
                    'youth_literacy_rate': round(youth_literacy, 2),
                    'adult_literacy_rate': round(adult_literacy, 2),
                    'primary_schools': primary_schools,
                    'secondary_schools': secondary_schools,
                    'total_teachers': total_teachers,
                    'total_students': total_students,
                    'student_teacher_ratio': round(student_teacher_ratio, 2),
                    'primary_enrollment_rate': round(primary_enrollment_rate, 2),
                    'secondary_enrollment_rate': round(secondary_enrollment_rate, 2),
                    'dropout_rate': round(dropout_rate, 2),
                    'completion_rate': round(completion_rate, 2),
                    'libraries': libraries,
                    'internet_penetration': round(internet_penetration, 2),
                    'gdp_per_capita': round(gdp_per_capita, 2),
                    'poverty_rate': round(poverty_rate, 2),
                    'is_urban': is_urban
                })
        
        df = pd.DataFrame(data)
        return df
    
    def validate_data(self, df):
        """Validate literacy data"""
        print("Validating data...")
        
        # Ensure rates are between 0 and 100
        rate_columns = [
            'literacy_rate', 'male_literacy_rate', 'female_literacy_rate',
            'youth_literacy_rate', 'adult_literacy_rate',
            'primary_enrollment_rate', 'secondary_enrollment_rate',
            'dropout_rate', 'completion_rate', 'internet_penetration',
            'poverty_rate'
        ]
        
        for col in rate_columns:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)
        
        # Ensure positive values for counts
        count_columns = ['population', 'primary_schools', 'secondary_schools',
                        'total_teachers', 'total_students', 'libraries']
        for col in count_columns:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        print(f"Validated {len(df)} records")
        return df
    
    def save_data(self, df, filename="literacy_data_raw.csv"):
        """Save data to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def ingest(self):
        """Main ingestion method"""
        print("Ingesting literacy data for Kenyan counties...")
        df = self.generate_literacy_data(years=5)
        df = self.validate_data(df)
        self.save_data(df)
        print(f"Ingested data for {len(df)} county-year combinations")
        return df

if __name__ == "__main__":
    ingestion = LiteracyDataIngestion()
    df = ingestion.ingest()
    print("\nSample data:")
    print(df.head())

