import pandas as pd
import joblib

model = joblib.load('../models/f1_winner_model.pkl')
imputer = joblib.load('../models/imputer.pkl')
le_driver = joblib.load('../models/le_driver.pkl')

sample_data = {
    'Driver': ['PIA', 'NOR', 'VER', 'RUS', 'LEC', 'HAM', 'SAI', 'ALO', 'TSU', 'GAS'],
    'QualifyingTime (s)': [89.5, 89.6, 89.8, 90.0, 90.1, 90.2, 90.5, 90.8, 91.0, 91.2],
    'Sector1Time (s)': [26.4, 26.5, 26.6, 26.7, 26.8, 26.9, 27.0, 27.1, 27.2, 27.3],
    'Sector2Time (s)': [36.5, 36.6, 36.7, 36.8, 36.9, 37.0, 37.1, 37.2, 37.3, 37.4],
    'Sector3Time (s)': [26.6, 26.5, 26.5, 26.5, 26.4, 26.3, 26.4, 26.5, 26.5, 26.5],
    'RainProbability': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    'Temperature': [29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
    'TeamPerformanceScore': [100, 98, 90, 85, 82, 80, 70, 65, 60, 55],
    'Position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Year': [2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025],
    'AirTemp': [29, 29, 29, 29, 29, 29, 29, 29, 29, 29],
    'Humidity': [80, 80, 80, 80, 80, 80, 80, 80, 80, 80],
    'Pressure': [1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010, 1010],
    'Rainfall': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'TrackTemp': [34, 34, 34, 34, 34, 34, 34, 34, 34, 34],
    'WindDirection': [180, 180, 180, 180, 180, 180, 180, 180, 180, 180],
    'WindSpeed': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    'CircuitLength': [4941, 4941, 4941, 4941, 4941, 4941, 4941, 4941, 4941, 4941],
    'NumberOfCorners': [23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
    'CircuitType': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

df_sample = pd.DataFrame(sample_data)

df_sample['Driver_encoded'] = le_driver.transform(df_sample['Driver'])

X_pred = df_sample[['QualifyingTime (s)', 'Driver_encoded', 'Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)', 'RainProbability', 'Temperature', 'TeamPerformanceScore', 'Position', 'Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'CircuitLength', 'NumberOfCorners', 'CircuitType']]

X_pred_imputed = imputer.transform(X_pred)

predicted_lap_times = model.predict(X_pred_imputed)
df_sample['PredictedLapTime (s)'] = predicted_lap_times

df_sample = df_sample.sort_values(by='PredictedLapTime (s)')

print("\n🏆 Predicted 2025 Singapore GP Top 3 Winners 🏆\n")
top_3 = df_sample.head(3)[['Driver', 'PredictedLapTime (s)']]
print(top_3)

print("\n📊 Predicted 2025 Singapore GP Top 10 Racers 📊\n")
top_10 = df_sample.head(10)[['Driver', 'PredictedLapTime (s)']]
print(top_10)