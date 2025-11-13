"""
Model Training Module for Soil Richness Analysis
Trains and evaluates multiple ML models
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers

class SoilRichnessModelTrainer:
    """Trains ML models for soil richness prediction"""
    
    def __init__(self, processed_data_path="data/processed/soil_data_processed.csv",
                 models_dir="models"):
        self.processed_data_path = Path(processed_data_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
    
    def load_processed_data(self):
        """Load processed data"""
        from src.data_processing import SoilDataProcessor
        
        processor = SoilDataProcessor()
        df = processor.load_data()
        df = processor.clean_data(df)
        df = processor.engineer_features(df)
        
        data_dict = processor.prepare_for_modeling(df)
        return data_dict
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results = {
            'model': rf,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
        
        print(f"Random Forest - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return results
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting...")
        
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = gb.predict(X_train)
        y_pred_test = gb.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results = {
            'model': gb,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
        
        print(f"Gradient Boosting - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results = {
            'model': xgb_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'predictions': y_pred_test
        }
        
        print(f"XGBoost - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return results
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Predictions
        y_pred_train = model.predict(X_train, verbose=0).flatten()
        y_pred_test = model.predict(X_test, verbose=0).flatten()
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        results = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'predictions': y_pred_test,
            'history': history
        }
        
        print(f"Neural Network - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return results
    
    def train_all_models(self):
        """Train all models"""
        print("Loading data...")
        data_dict = self.load_processed_data()
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples\n")
        
        # Train models
        self.results['random_forest'] = self.train_random_forest(X_train, y_train, X_test, y_test)
        self.results['gradient_boosting'] = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.results['xgboost'] = self.train_xgboost(X_train, y_train, X_test, y_test)
        self.results['neural_network'] = self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_models()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def save_models(self):
        """Save trained models"""
        print("\nSaving models...")
        for name, result in self.results.items():
            model = result['model']
            if name == 'neural_network':
                model_path = self.models_dir / f"{name}.h5"
                model.save(model_path)
            else:
                model_path = self.models_dir / f"{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
    
    def print_summary(self):
        """Print model comparison summary"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
        print("-"*60)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['test_r2']:<12.4f} {result['test_rmse']:<12.4f} {result['test_mae']:<12.4f}")
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
        print("="*60)

if __name__ == "__main__":
    trainer = SoilRichnessModelTrainer()
    results = trainer.train_all_models()

