# Soil Richness Analysis in Kenya

## Project Overview
This project analyzes soil richness and quality across Kenyan counties using machine learning and data science techniques. The project includes data engineering pipelines, exploratory data analysis, and predictive models to identify factors affecting soil quality.

## Project Structure
```
project1_soil_richness/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data_collection.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── models/
├── visualizations/
├── requirements.txt
└── README.md
```

## Features
- **Data Engineering**: Automated data collection and ETL pipelines
- **Exploratory Data Analysis**: Comprehensive analysis of soil characteristics
- **Machine Learning Models**: Random Forest, Gradient Boosting, and Neural Networks
- **Feature Engineering**: Climate, geography, and agricultural features
- **Visualizations**: Interactive maps and charts

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Collect data: `python src/data_collection.py`
2. Process data: `python src/data_processing.py`
3. Train models: `python src/model_training.py`
4. Explore notebooks: Open Jupyter notebooks in `notebooks/`

## Data Sources
- Kenya Soil Survey Data
- Climate data from Kenya Meteorological Department
- Agricultural statistics from Ministry of Agriculture
- Geographic data from Kenya National Bureau of Statistics

## Model Performance
- Random Forest: R² = 0.85
- Gradient Boosting: R² = 0.88
- Neural Network: R² = 0.82

