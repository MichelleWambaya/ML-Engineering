"""
Main runner script for Mortality Rates Analysis Project
Runs the complete pipeline from data collection to analysis
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline import MortalityDataPipeline
from etl_process import MortalityETL
from statistical_analysis import MortalityStatisticalAnalysis
from time_series_analysis import MortalityTimeSeriesAnalysis

def main():
    print("="*70)
    print("MORTALITY RATES ANALYSIS PROJECT - KENYA")
    print("="*70)
    
    # Step 1: Data Pipeline
    print("\n[1/4] Running data pipeline...")
    pipeline = MortalityDataPipeline()
    pipeline.run_pipeline()
    
    # Step 2: ETL Process
    print("\n[2/4] Running ETL process...")
    etl = MortalityETL()
    etl.run_etl()
    
    # Step 3: Statistical Analysis
    print("\n[3/4] Performing statistical analysis...")
    analyzer = MortalityStatisticalAnalysis()
    analyzer.run_all_analyses()
    
    # Step 4: Time Series Analysis
    print("\n[4/4] Performing time series analysis...")
    ts_analyzer = MortalityTimeSeriesAnalysis()
    ts_analyzer.run_analysis()
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Check data/processed/ for processed datasets")
    print("2. Review statistical analysis results in console output")
    print("3. Open notebooks/ for detailed analysis")
    print("="*70)

if __name__ == "__main__":
    main()

