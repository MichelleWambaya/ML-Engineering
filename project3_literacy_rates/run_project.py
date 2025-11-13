"""
Main runner script for Literacy Rates Analysis Project
Runs the complete pipeline from data ingestion to model training
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_ingestion import LiteracyDataIngestion
from feature_engineering import LiteracyFeatureEngineering
from predictive_models import LiteracyPredictiveModels

def main():
    print("="*70)
    print("LITERACY RATES ANALYSIS PROJECT - KENYA")
    print("="*70)
    
    # Step 1: Data Ingestion
    print("\n[1/3] Ingesting data...")
    ingestion = LiteracyDataIngestion()
    ingestion.ingest()
    
    # Step 2: Feature Engineering
    print("\n[2/3] Engineering features...")
    engineer = LiteracyFeatureEngineering()
    engineer.engineer()
    
    # Step 3: Model Training
    print("\n[3/3] Training predictive models...")
    trainer = LiteracyPredictiveModels()
    trainer.train_all_models()
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check models/ folder for trained models")
    print("2. Launch dashboard: python src/dashboard_app.py")
    print("3. Open notebooks/ for detailed analysis")
    print("="*70)
    print("\nTo launch the interactive dashboard:")
    print("  python src/dashboard_app.py")
    print("  Then open http://127.0.0.1:8050 in your browser")
    print("="*70)

if __name__ == "__main__":
    main()

