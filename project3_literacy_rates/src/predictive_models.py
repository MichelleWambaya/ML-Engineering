"""
Predictive Models Module for Literacy Rates Analysis
Trains and evaluates ML models for literacy prediction
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import shap

class LiteracyPredictiveModels:
    """Predictive models for literacy rates"""
    
    def __init__(self, processed_data_path="data/processed/literacy_data_processed.csv",
                 models_dir="models"):
        self.processed_data_path = Path(processed_data_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
    
    def load_processed_data(self):
        """Load processed data"""
        from src.feature_engineering import LiteracyFeatureEngineering
        
        engineer = LiteracyFeatureEngineering()
        df = engineer.load_data()
        df = engineer.create_features(df)
        
        data_dict = engineer.prepare_for_modeling(df)
        return data_dict, engineer
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        print("Training Random Forest...")
        
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
        
        results = {
            'model': rf,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'feature_importance': feature_importance
        }
        
        print(f"Random Forest - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("Training XGBoost...")
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
        
        results = {
            'model': xgb_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'feature_importance': feature_importance
        }
        
        print(f"XGBoost - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model"""
        print("Training LightGBM...")
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = lgb_model.predict(X_train)
        y_pred_test = lgb_model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring='r2')
        
        results = {
            'model': lgb_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred_test,
            'feature_importance': feature_importance
        }
        
        print(f"LightGBM - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        print(f"CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model"""
        print("Training Neural Network...")
        
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
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
    
    def explain_model(self, model, X_test, model_name='random_forest'):
        """Generate SHAP explanations for model"""
        print(f"Generating SHAP explanations for {model_name}...")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Calculate feature importance from SHAP
            shap_importance = pd.DataFrame({
                'feature': X_test.columns,
                'shap_importance': np.abs(shap_values).mean(0)
            }).sort_values('shap_importance', ascending=False)
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'feature_importance': shap_importance
            }
        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return None
    
    def train_all_models(self):
        """Train all models"""
        print("Loading data...")
        data_dict, engineer = self.load_processed_data()
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples\n")
        
        # Train models
        self.results['random_forest'] = self.train_random_forest(X_train, y_train, X_test, y_test)
        self.results['xgboost'] = self.train_xgboost(X_train, y_train, X_test, y_test)
        self.results['lightgbm'] = self.train_lightgbm(X_train, y_train, X_test, y_test)
        self.results['neural_network'] = self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Generate SHAP explanations for tree-based models
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in self.results:
                shap_result = self.explain_model(
                    self.results[model_name]['model'],
                    X_test,
                    model_name
                )
                if shap_result:
                    self.results[model_name]['shap'] = shap_result
        
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
        print("\n" + "="*70)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*70)
        print(f"{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12} {'CV R²':<12}")
        print("-"*70)
        
        for name, result in self.results.items():
            cv_str = f"{result.get('cv_mean', 0):.4f}" if 'cv_mean' in result else "N/A"
            print(f"{name:<20} {result['test_r2']:<12.4f} {result['test_rmse']:<12.4f} "
                  f"{result['test_mae']:<12.4f} {cv_str:<12}")
        
        # Best model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_r2'])
        print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]['test_r2']:.4f})")
        print("="*70)

if __name__ == "__main__":
    trainer = LiteracyPredictiveModels()
    results = trainer.train_all_models()

