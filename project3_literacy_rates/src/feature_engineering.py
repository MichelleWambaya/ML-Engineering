"""
Feature Engineering Module for Literacy Rates Analysis
Creates and transforms features for machine learning
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class LiteracyFeatureEngineering:
    """Feature engineering for literacy prediction"""
    
    def __init__(self, raw_data_path="data/raw/literacy_data_raw.csv",
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
    
    def create_features(self, df):
        """Create engineered features"""
        print("Creating features...")
        
        # Infrastructure density features
        df['schools_per_100k'] = ((df['primary_schools'] + df['secondary_schools']) / 
                                  df['population'] * 100000).round(2)
        df['teachers_per_100k'] = (df['total_teachers'] / df['population'] * 100000).round(2)
        df['students_per_100k'] = (df['total_students'] / df['population'] * 100000).round(2)
        df['libraries_per_100k'] = (df['libraries'] / df['population'] * 100000).round(2)
        
        # Education quality indicators
        df['school_ratio'] = (df['secondary_schools'] / 
                             (df['primary_schools'] + 1)).round(2)
        df['enrollment_gap'] = df['primary_enrollment_rate'] - df['secondary_enrollment_rate']
        
        # Economic and infrastructure composite
        df['economic_index'] = (
            (df['gdp_per_capita'] / 3000) * 0.5 +
            ((100 - df['poverty_rate']) / 100) * 0.3 +
            (df['internet_penetration'] / 100) * 0.2
        ).round(3)
        
        # Education infrastructure index
        df['education_infrastructure_index'] = (
            (df['schools_per_100k'] / 200) * 0.3 +
            (df['teachers_per_100k'] / 500) * 0.3 +
            (df['libraries_per_100k'] / 10) * 0.2 +
            (df['internet_penetration'] / 100) * 0.2
        ).round(3)
        
        # Accessibility features
        df['school_accessibility'] = (
            (df['schools_per_100k'] / 200) * 0.6 +
            (1 / (df['student_teacher_ratio'] / 50)) * 0.4
        ).round(3)
        
        # Gender equality index
        df['gender_equality_index'] = (1 - (df['gender_gap'] / 100)).clip(0, 1).round(3)
        
        # Education efficiency
        df['education_efficiency'] = (
            (df['completion_rate'] / 100) * 0.5 +
            (1 / (df['dropout_rate'] / 100 + 0.01)) * 0.3 +
            (1 / (df['student_teacher_ratio'] / 50)) * 0.2
        ).round(3)
        
        # Lag features (previous year's literacy)
        df = df.sort_values(['county', 'year'])
        df['literacy_rate_lag1'] = df.groupby('county')['literacy_rate'].shift(1)
        df['literacy_rate_change'] = df.groupby('county')['literacy_rate'].diff()
        
        # Rolling averages
        df['literacy_rate_ma2'] = df.groupby('county')['literacy_rate'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean()
        )
        
        # Year features
        df['years_since_2019'] = df['year'] - 2019
        df['is_recent'] = (df['year'] >= 2022).astype(int)
        
        # County encoding
        df['county_encoded'] = self.label_encoder.fit_transform(df['county'])
        
        # Literacy categories
        df['literacy_category'] = pd.cut(
            df['literacy_rate'],
            bins=[0, 50, 70, 85, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        print(f"Created {len(df.columns)} features")
        return df
    
    def select_features(self, df, target_col='literacy_rate'):
        """Select features for modeling"""
        feature_cols = [
            # Demographics
            'population',
            
            # Infrastructure
            'primary_schools', 'secondary_schools', 'total_teachers',
            'schools_per_100k', 'teachers_per_100k', 'students_per_100k',
            'libraries_per_100k', 'school_ratio',
            
            # Education metrics
            'primary_enrollment_rate', 'secondary_enrollment_rate',
            'enrollment_gap', 'student_teacher_ratio',
            'dropout_rate', 'completion_rate',
            
            # Economic
            'gdp_per_capita', 'poverty_rate', 'internet_penetration',
            
            # Composite indices
            'economic_index', 'education_infrastructure_index',
            'school_accessibility', 'gender_equality_index',
            'education_efficiency',
            
            # Temporal
            'years_since_2019', 'is_recent',
            
            # Lag features
            'literacy_rate_lag1', 'literacy_rate_change', 'literacy_rate_ma2',
            
            # Categorical
            'county_encoded', 'is_urban'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        return available_features
    
    def prepare_for_modeling(self, df, target_col='literacy_rate',
                            test_size=0.2, random_state=42):
        """Prepare data for machine learning"""
        print("Preparing data for modeling...")
        
        # Select features
        feature_cols = self.select_features(df, target_col)
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
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
        print(f"Features: {len(feature_cols)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols
        }
    
    def save_processed_data(self, df, filename="literacy_data_processed.csv"):
        """Save processed data"""
        filepath = self.processed_data_path / filename
        df.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
        return filepath
    
    def engineer(self):
        """Main feature engineering pipeline"""
        df = self.load_data()
        df = self.create_features(df)
        self.save_processed_data(df)
        return df

if __name__ == "__main__":
    engineer = LiteracyFeatureEngineering()
    df = engineer.engineer()
    print("\nProcessed data sample:")
    print(df.head())
    print("\nFeature columns:", len(df.columns))

