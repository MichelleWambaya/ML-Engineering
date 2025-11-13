"""
Main runner script for Soil Richness Analysis Project
Runs the complete pipeline from data collection to visualization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_collection import SoilDataCollector
from data_processing import SoilDataProcessor
from model_training import SoilRichnessModelTrainer
from visualizations import SoilVisualizer

def main():
    print("="*70)
    print("SOIL RICHNESS ANALYSIS PROJECT - KENYA")
    print("="*70)
    
    # Step 1: Data Collection
    print("\n[1/4] Collecting data...")
    collector = SoilDataCollector()
    collector.collect()
    
    # Step 2: Data Processing
    print("\n[2/4] Processing data...")
    processor = SoilDataProcessor()
    processor.process()
    
    # Step 3: Model Training
    print("\n[3/4] Training models...")
    trainer = SoilRichnessModelTrainer()
    trainer.train_all_models()
    
    # Step 4: Visualizations
    print("\n[4/4] Generating visualizations...")
    visualizer = SoilVisualizer()
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check visualizations/ folder for charts")
    print("2. Check models/ folder for trained models")
    print("3. Open notebooks/ for detailed analysis")
    print("="*70)

if __name__ == "__main__":
    main()

