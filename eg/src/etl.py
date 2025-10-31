import pandas as pd
import numpy as np
import os

def run_etl():
    csv_path = "/content/data/raw/World Energy Consumption.csv"
    df = pd.read_csv(csv_path)

    # Keep only rows with country, year, and consumption
    df = df.dropna(subset=['country', 'year', 'primary_energy_consumption', 'population'])
    df = df[df['country'] != 'World']

    # Convert year
    df['year'] = pd.to_datetime(df['year'], format='%Y')

    # Safe features (EXISTING COLUMNS)
    df['log_consumption'] = np.log1p(df['primary_energy_consumption'])
    df['gdp_per_capita'] = df['gdp'] / df['population']
    df['growth_rate'] = df.groupby('country')['primary_energy_consumption'].pct_change().rolling(3).mean()

    # Filter 2000+
    df = df[df['year'].dt.year >= 2000]

    # Save
    os.makedirs("/content/energyglobal/data/processed", exist_ok=True)
    df.to_parquet("/content/energyglobal/data/processed/energy_data.parquet")
    print("ETL DONE")

if __name__ == "__main__":
    run_etl()
