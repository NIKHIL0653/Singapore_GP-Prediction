import fastf1
import pandas as pd
import os

os.makedirs('../data/cache', exist_ok=True)
fastf1.Cache.enable_cache('../data/cache')

years = range(2016, 2025)

all_laps = []
all_qualifying = []
all_weather = []

for year in years:
    try:
        session_race = fastf1.get_session(year, 'Singapore Grand Prix', 'R')
        session_race.load()
        laps = session_race.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
        laps['Year'] = year
        all_laps.append(laps)

        session_qual = fastf1.get_session(year, 'Singapore Grand Prix', 'Q')
        session_qual.load()
        qualifying = session_qual.results.copy()
        qualifying['Year'] = year
        all_qualifying.append(qualifying)

        weather = session_race.weather_data.copy()
        weather['Year'] = year
        all_weather.append(weather)

        print(f"Got data for {year}")
    except Exception as e:
        print(f"Couldn't load data for {year}: {e}")

if all_laps:
    combined_laps = pd.concat(all_laps, ignore_index=True)
    combined_laps['LapTime (s)'] = combined_laps['LapTime'].dt.total_seconds()
    combined_laps.to_csv('../data/singapore_gp_laps.csv', index=False)
    print("Saved lap data")

if all_qualifying:
    combined_qualifying = pd.concat(all_qualifying, ignore_index=True)
    combined_qualifying.to_csv('../data/singapore_gp_qualifying.csv', index=False)
    print("Saved qualifying data")

if all_weather:
    combined_weather = pd.concat(all_weather, ignore_index=True)
    combined_weather.to_csv('../data/singapore_gp_weather.csv', index=False)
    print("Saved weather data")
else:
    print("No data was loaded.")