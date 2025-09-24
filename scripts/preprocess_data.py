import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

laps_df = pd.read_csv('../data/singapore_gp_laps.csv')
qualifying_df = pd.read_csv('../data/singapore_gp_qualifying.csv')
weather_df = pd.read_csv('../data/singapore_gp_weather.csv')

print("Laps columns:", laps_df.columns.tolist())
print("Qualifying columns:", qualifying_df.columns.tolist())
print("Weather columns:", weather_df.columns.tolist())

laps_df = laps_df.dropna(subset=['Driver', 'LapTime (s)'])

for col in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
    if col in laps_df.columns:
        if laps_df[col].dtype == 'timedelta64[ns]':
            laps_df[f'{col} (s)'] = laps_df[col].dt.total_seconds()
        else:
            # If it's not a timedelta, try converting it
            try:
                laps_df[f'{col} (s)'] = pd.to_timedelta(laps_df[col]).dt.total_seconds()
            except:
                print(f"Couldn't convert {col} to seconds, keeping as is")
                laps_df[f'{col} (s)'] = laps_df[col]

avg_data = laps_df.groupby(['Driver', 'Year']).agg({
    'LapTime (s)': 'mean',
    'Sector1Time (s)': 'mean',
    'Sector2Time (s)': 'mean',
    'Sector3Time (s)': 'mean'
}).reset_index()

# Average weather per year
avg_weather = weather_df.groupby('Year').agg({
    'AirTemp': 'mean',
    'Humidity': 'mean',
    'Pressure': 'mean',
    'Rainfall': 'mean',
    'TrackTemp': 'mean',
    'WindDirection': 'mean',
    'WindSpeed': 'mean'
}).reset_index()

qualifying_df = qualifying_df.dropna(subset=['Abbreviation', 'Q3'])

qualifying_df['QualifyingTime (s)'] = pd.to_timedelta(qualifying_df['Q3']).dt.total_seconds()

merged_df = avg_data.merge(qualifying_df[['Abbreviation', 'QualifyingTime (s)', 'Position', 'Year']],
                            left_on=['Driver', 'Year'], right_on=['Abbreviation', 'Year'], how='inner')

merged_df = merged_df.merge(avg_weather, on='Year', how='left')

le_driver = LabelEncoder()
merged_df['Driver_encoded'] = le_driver.fit_transform(merged_df['Driver'])

y = merged_df['LapTime (s)']

# Add circuit-specific features (Singapore GP)
merged_df['CircuitLength'] = 4941  # meters
merged_df['NumberOfCorners'] = 23
merged_df['CircuitType'] = 1  # 1 for street circuit

merged_df['RainProbability'] = 0.1
merged_df['Temperature'] = 25

driver_to_team = {
    'VER': 'Red Bull', 'HAM': 'Mercedes', 'LEC': 'Ferrari', 'NOR': 'McLaren', 'ALO': 'Aston Martin',
    'RUS': 'Mercedes', 'SAI': 'Ferrari', 'TSU': 'Racing Bulls', 'OCO': 'Alpine', 'GAS': 'Alpine',
    'STR': 'Aston Martin', 'HUL': 'Kick Sauber', 'PIA': 'McLaren', 'ALB': 'Williams', 'BOT': 'Kick Sauber'
}
team_points = {
    'Red Bull': 100, 'Mercedes': 80, 'McLaren': 70, 'Ferrari': 60, 'Aston Martin': 40,
    'Alpine': 30, 'Racing Bulls': 20, 'Kick Sauber': 10, 'Williams': 5
}
merged_df['Team'] = merged_df['Driver'].map(driver_to_team).fillna('Unknown')
merged_df['TeamPerformanceScore'] = merged_df['Team'].map(team_points).fillna(0)

X = merged_df[['QualifyingTime (s)', 'Driver_encoded', 'Sector1Time (s)', 'Sector2Time (s)', 'Sector3Time (s)', 'RainProbability', 'Temperature', 'TeamPerformanceScore', 'Position', 'Year', 'AirTemp', 'Humidity', 'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed', 'CircuitLength', 'NumberOfCorners', 'CircuitType']]

X.to_csv('../data/X_features.csv', index=False)
y.to_csv('../data/y_target.csv', index=False)

import joblib
joblib.dump(le_driver, '../models/le_driver.pkl')

print("Data processing complete.")