"""
Data Processing Module for Soil Richness Analysis
Cleans, transforms, and prepares data for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class SoilDataProcessor:
    """Processes and cleans soil data"""
    
    def __init__(self, raw_data_path="data/raw/soil_data_raw.csv", 
                 processed_data_path="data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """Load raw data"""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, df):
        """Clean and validate data"""
        print("Cleaning data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        print(f"Removed {initial_count - len(df)} duplicate records")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        print(f"Removed {missing_before} missing values")
        
        # Validate ranges
        df['ph'] = df['ph'].clip(4.0, 9.0)
        df['organic_matter_percent'] = df['organic_matter_percent'].clip(0, 10)
        df['nitrogen_ppm'] = df['nitrogen_ppm'].clip(0, 200)
        df['phosphorus_ppm'] = df['phosphorus_ppm'].clip(0, 150)
        df['potassium_ppm'] = df['potassium_ppm'].clip(0, 500)
        
        # Ensure texture percentages sum correctly
        texture_sum = df[['clay_percent', 'sand_percent', 'silt_percent']].sum(axis=1)
        for col in ['clay_percent', 'sand_percent', 'silt_percent']:
            df[col] = (df[col] / texture_sum * 100).round(2)
        
        print(f"Cleaned data: {len(df)} records")
        return df
    
    def engineer_features(self, df):
        """Create additional features"""
        print("Engineering features...")
        
        # Nutrient ratios
        df['np_ratio'] = df['nitrogen_ppm'] / (df['phosphorus_ppm'] + 1e-6)
        df['nk_ratio'] = df['nitrogen_ppm'] / (df['potassium_ppm'] + 1e-6)
        df['pk_ratio'] = df['phosphorus_ppm'] / (df['potassium_ppm'] + 1e-6)
        
        # Nutrient balance index
        df['nutrient_balance'] = (
            (df['nitrogen_ppm'] / 100) * 0.4 +
            (df['phosphorus_ppm'] / 80) * 0.3 +
            (df['potassium_ppm'] / 300) * 0.3
        )
        
        # Climate suitability
        df['climate_index'] = (
            (df['rainfall_mm_per_year'] / 1000) * 0.5 +
            (1 - abs(df['temperature_celsius'] - 25) / 10) * 0.5
        )
        
        # pH categories
        df['ph_category'] = pd.cut(
            df['ph'],
            bins=[0, 6.0, 7.0, 8.0, 10],
            labels=['Acidic', 'Neutral', 'Alkaline', 'Highly Alkaline']
        )
        
        # Soil texture classification
        def classify_texture(row):
            clay, sand, silt = row['clay_percent'], row['sand_percent'], row['silt_percent']
            if clay >= 40:
                return 'Clay'
            elif sand >= 70:
                return 'Sandy'
            elif silt >= 50:
                return 'Silty'
            elif clay >= 27 and sand < 52:
                return 'Clay Loam'
            elif sand >= 43 and sand < 52:
                return 'Loam'
            else:
                return 'Sandy Loam'
        
        df['texture_class'] = df.apply(classify_texture, axis=1)
        
        # Richness categories
        df['richness_category'] = pd.cut(
            df['soil_richness_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        print(f"Created {len(df.columns)} features")
        return df
    
    def prepare_for_modeling(self, df, target_col='soil_richness_score', 
                            test_size=0.2, random_state=42):
        """Prepare data for machine learning"""
        print("Preparing data for modeling...")
        
        # Select features
        feature_cols = [
            'ph', 'organic_matter_percent', 'nitrogen_ppm', 'phosphorus_ppm',
            'potassium_ppm', 'cec_meq_per_100g', 'clay_percent', 'sand_percent',
            'silt_percent', 'rainfall_mm_per_year', 'temperature_celsius',
            'elevation_meters', 'np_ratio', 'nk_ratio', 'pk_ratio',
            'nutrient_balance', 'climate_index'
        ]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Encode categorical if needed
        if 'county' in df.columns:
            df['county_encoded'] = self.label_encoder.fit_transform(df['county'])
            X['county_encoded'] = df['county_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }
    
    def save_processed_data(self, df, filename="soil_data_processed.csv"):
        """Save processed data"""
        filepath = self.processed_data_path / filename
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
        return filepath
    
    def process(self):
        """Main processing pipeline"""
        df = self.load_data()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        self.save_processed_data(df)
        return df

if __name__ == "__main__":
    processor = SoilDataProcessor()
    df = processor.process()
    print("\nProcessed data sample:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

