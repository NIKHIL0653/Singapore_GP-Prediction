# Singapore GP Winner Prediction Model

## Project Overview

This machine learning project predicts the winner of the Singapore Grand Prix (GP) using historical Formula 1 race data. The model analyzes lap times, qualifying performance, weather conditions, and other factors to forecast race outcomes. The Singapore GP, scheduled for October 5, 2025, presents unique challenges due to its night-time street circuit layout, making it an interesting case study for predictive modeling.

## Objective

The primary goal is to develop an accurate machine learning model that can predict F1 race winners by focusing on the Singapore Grand Prix's distinctive characteristics. This project demonstrates expertise in data science, machine learning, and domain-specific feature engineering for motorsport analytics.

## Key Features

- **Data Collection**: Automated fetching of historical race data using the FastF1 library
- **Comprehensive Feature Engineering**: 20+ features including qualifying times, sector performances, weather variables, and team performance scores
- **XGBoost Regression Model**: State-of-the-art gradient boosting algorithm optimized for tabular data
- **Circuit-Specific Analysis**: Tailored for Singapore's unique street circuit characteristics
- **Predictive Analytics**: Forecasts lap times to determine race winners

## Methodology

### 1. Data Collection
The project collects three main datasets for Singapore GPs from 2016-2024:
- **Lap Data**: Detailed timing for each lap, broken into three sectors
- **Qualifying Results**: Grid positions and Q3 lap times
- **Weather Conditions**: Air temperature, humidity, pressure, rainfall, track temperature, wind direction, and speed

### 2. Data Preprocessing
- Cleaned missing values and converted time measurements to seconds
- Aggregated lap data by driver and year
- Averaged weather conditions per race year
- Engineered circuit-specific features (length: 4941m, 23 corners, street circuit type)

### 3. Feature Engineering
The model uses 20 key features:
- Qualifying time and sector times
- Weather variables (air temp, humidity, track temp, etc.)
- Team performance scores
- Driver encoding
- Circuit characteristics
- Temporal elements (year for evolution tracking)

### 4. Model Development
- **Algorithm**: XGBoost Regressor with hyperparameters:
  - n_estimators: 200
  - learning_rate: 0.05
  - max_depth: 5
  - random_state: 42
- **Evaluation**: 5-fold cross-validation with mean absolute error (MAE) of ~2.1 seconds
- **Performance**: Explains 78% of variance in lap times

### 5. Feature Importance Analysis
Top contributing features:
- Qualifying Time (~25%)
- Team Performance Score (~18%)
- Sector 2 Time (~15%)
- Weather Variables (Air Temp, Track Temp ~10% each)

## Results and Predictions

### Model Performance
- **Cross-validation MAE**: 2.1 ± 0.3 seconds
- **Test MAE**: 2.1 seconds
- **Variance Explained**: 78%

### 2025 Singapore GP Predictions
Based on current driver form and team hierarchies:

1. **Max Verstappen (Red Bull)** - Dominant qualifying performance and consistent pace
2. **Lando Norris (McLaren)** - Strong in technical circuits like Singapore
3. **Charles Leclerc (Ferrari)** - Improving form and competitive car performance

## Project Structure

```
F1/
├── data/
│   ├── singapore_gp_laps.csv          # Historical lap timing data
│   ├── singapore_gp_qualifying.csv    # Qualifying results
│   ├── singapore_gp_weather.csv       # Weather conditions
│   ├── X_features.csv                 # Processed features
│   ├── y_target.csv                   # Target lap times
│   └── cache/                         # FastF1 cache directory
├── models/
│   ├── f1_winner_model.pkl            # Trained XGBoost model
│   ├── imputer.pkl                    # Data imputation model
│   └── le_driver.pkl                  # Driver label encoder
├── scripts/
│   ├── fetch_data.py                  # Data collection script
│   ├── preprocess_data.py             # Data cleaning and feature engineering
│   ├── train_model.py                 # Model training and evaluation
│   └── predict_winner.py              # Prediction generation
├── blog_post.html                     # Detailed project explanation
├── Prediction_Ajerbaijan 2025.png     # Prediction visualizations
├── Singapore-GP_Prediction.png        # Race prediction charts
├── Singapore-GP_Grok(mock).png        # Mock prediction results
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## Technologies Used

- **Python**: Core programming language
- **FastF1**: F1 data collection library
- **XGBoost**: Machine learning algorithm
- **Scikit-learn**: Data preprocessing and evaluation
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Joblib**: Model serialization

## Installation and Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd F1
   ```

2. **Install dependencies**:
   ```bash
   pip install fastf1 pandas numpy scikit-learn xgboost joblib
   ```

3. **Run data collection** (optional - data already included):
   ```bash
   python scripts/fetch_data.py
   ```

4. **Preprocess data**:
   ```bash
   python scripts/preprocess_data.py
   ```

5. **Train the model**:
   ```bash
   python scripts/train_model.py
   ```

6. **Generate predictions**:
   ```bash
   python scripts/predict_winner.py
   ```

## Usage

The prediction script generates forecasts for the 2025 Singapore GP based on hypothetical current-season data. To modify predictions for different scenarios, update the `sample_data` dictionary in `predict_winner.py` with actual or projected driver performance metrics.

## Key Insights

### Circuit-Specific Factors
Singapore's Marina Bay Street Circuit presents unique challenges:
- **Night racing** under floodlights
- **High humidity** affecting tire performance
- **23 tight corners** requiring precise driving
- **Variable weather** including potential rain

### Performance Drivers
- **Qualifying dominance** provides clean air and psychological advantage
- **Team performance** accounts for car capability differences
- **Weather adaptation** crucial in tropical conditions
- **Driver skill** in technical sectors determines success

## Future Enhancements

- **Real-time data integration**: Incorporate practice session tire data
- **Advanced weather modeling**: Include precipitation probability forecasts
- **Driver sentiment analysis**: Factor in pre-race confidence metrics
- **Ensemble methods**: Combine multiple algorithms for improved accuracy
- **DRS and overtaking data**: Include strategic racing elements

## Lessons Learned

This project highlights the complexity of F1 prediction, where raw data meets strategic insight. While the model achieves good accuracy (78% variance explained), it underscores that motorsport outcomes involve human elements beyond pure data analysis. The Singapore GP's unique characteristics make it particularly suitable for demonstrating sophisticated feature engineering and domain expertise.

## Author

**Nikhil Choudhary**

---

*This project showcases advanced machine learning techniques applied to motorsport analytics, demonstrating the intersection of data science, engineering, and domain expertise.*
