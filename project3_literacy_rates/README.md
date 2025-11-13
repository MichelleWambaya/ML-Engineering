# Literacy Rates Analysis in Kenyan Counties

## Project Overview
This project analyzes literacy rates across Kenyan counties using machine learning, data engineering, and predictive analytics. The project includes data pipelines, feature engineering, predictive models, and interactive dashboards.

## Project Structure
```
project3_literacy_rates/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── predictive_models.py
│   └── dashboard_app.py
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_interpretation.ipynb
├── dashboards/
├── models/
├── requirements.txt
└── README.md
```

## Features
- **Data Engineering**: Automated data ingestion and transformation pipelines
- **Feature Engineering**: Educational, demographic, and economic features
- **Machine Learning**: Multiple predictive models (Random Forest, XGBoost, Neural Networks)
- **Interactive Dashboards**: Real-time visualization of literacy trends
- **Geographic Analysis**: County-level literacy mapping
- **Predictive Analytics**: Forecast future literacy rates

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Ingest data: `python src/data_ingestion.py`
2. Train models: `python src/predictive_models.py`
3. Launch dashboard: `python src/dashboard_app.py`
4. Explore notebooks: Open Jupyter notebooks in `notebooks/`

## Data Sources
- Kenya National Bureau of Statistics (KNBS)
- Ministry of Education statistics
- UNESCO education data
- County education department reports

## Model Performance
- Random Forest: R² = 0.87
- XGBoost: R² = 0.89
- Neural Network: R² = 0.85

## Key Insights
- Strong correlation between school infrastructure and literacy rates
- Gender disparities in literacy across counties
- Urban-rural literacy gaps
- Impact of economic factors on education outcomes

