# Mortality Rates Analysis in Kenyan Counties

## Project Overview
This project analyzes mortality rates across Kenyan counties using data engineering, statistical analysis, and predictive modeling. The project includes ETL pipelines, time series analysis, and identification of factors affecting mortality patterns.

## Project Structure
```
project2_mortality_rates/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data_pipeline.py
│   ├── etl_process.py
│   ├── statistical_analysis.py
│   └── time_series_analysis.py
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_statistical_analysis.ipynb
│   └── 03_time_series_forecasting.ipynb
├── dashboards/
├── requirements.txt
└── README.md
```

## Features
- **Data Engineering**: Automated ETL pipelines for mortality data
- **Statistical Analysis**: Comprehensive mortality pattern analysis
- **Time Series Analysis**: Forecasting and trend analysis
- **Geographic Analysis**: County-level mortality mapping
- **Risk Factor Identification**: Correlation with health indicators

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run ETL pipeline: `python src/etl_process.py`
2. Statistical analysis: `python src/statistical_analysis.py`
3. Time series analysis: `python src/time_series_analysis.py`
4. Explore notebooks: Open Jupyter notebooks in `notebooks/`

## Data Sources
- Kenya National Bureau of Statistics (KNBS)
- Ministry of Health mortality records
- WHO health statistics
- County health department reports

## Key Insights
- Infant mortality rates vary significantly across counties
- Strong correlation between healthcare access and mortality rates
- Seasonal patterns in mortality data
- Urban vs rural mortality disparities

