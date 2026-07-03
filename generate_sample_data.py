"""
generate_sample_data.py
------------------------
Generates a realistic synthetic house price dataset (house_prices.csv)
for use in the House Price Prediction System when no external dataset
is available.

To use your OWN dataset instead:
    1. Delete or ignore this script.
    2. Place your CSV file inside this "data/" folder.
    3. Name the target column one of: "Price", "SalePrice", "HousePrice".
    4. Update the DATA_PATH in main.py / src/data_preprocessing.py if the
       filename differs from "house_prices.csv".

Developed by: Sapna Jabeen
"""

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 2000

neighborhoods = ["Downtown", "Suburb", "Rural", "Uptown", "Lakeside", "Hillview"]
neighborhood_weights = [0.20, 0.30, 0.10, 0.15, 0.15, 0.10]

# Base premium multiplier per neighborhood (affects price)
neighborhood_premium = {
    "Downtown": 1.35,
    "Suburb": 1.00,
    "Rural": 0.75,
    "Uptown": 1.20,
    "Lakeside": 1.45,
    "Hillview": 1.10,
}

data = {}

data["SquareFeet"] = np.random.normal(1800, 650, N_SAMPLES).clip(400, 6000).round(0)
data["Bedrooms"] = np.random.choice([1, 2, 3, 4, 5, 6], N_SAMPLES,
                                     p=[0.05, 0.20, 0.35, 0.25, 0.10, 0.05])
data["Bathrooms"] = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], N_SAMPLES,
                                      p=[0.10, 0.10, 0.30, 0.20, 0.15, 0.10, 0.05])
data["YearBuilt"] = np.random.randint(1950, 2024, N_SAMPLES)
data["LotSize"] = np.random.normal(6000, 2500, N_SAMPLES).clip(500, 20000).round(0)
data["GarageCars"] = np.random.choice([0, 1, 2, 3], N_SAMPLES, p=[0.10, 0.30, 0.45, 0.15])
data["Neighborhood"] = np.random.choice(neighborhoods, N_SAMPLES, p=neighborhood_weights)
data["OverallQuality"] = np.random.randint(1, 11, N_SAMPLES)  # 1-10 scale
data["HasPool"] = np.random.choice([0, 1], N_SAMPLES, p=[0.85, 0.15])
data["DistanceToCityCenter_km"] = np.random.exponential(8, N_SAMPLES).clip(0.5, 50).round(2)
data["Stories"] = np.random.choice([1, 2, 3], N_SAMPLES, p=[0.40, 0.50, 0.10])

df = pd.DataFrame(data)

# ------------------------------------------------------------------
# Construct a realistic target price using a weighted formula + noise
# ------------------------------------------------------------------
age = 2024 - df["YearBuilt"]

base_price = (
    df["SquareFeet"] * 120
    + df["Bedrooms"] * 8000
    + df["Bathrooms"] * 12000
    + df["GarageCars"] * 7000
    + df["OverallQuality"] * 9000
    + df["HasPool"] * 15000
    + df["LotSize"] * 3
    - age * 500
    - df["DistanceToCityCenter_km"] * 1200
    + df["Stories"] * 4000
    + 50000
)

premium = df["Neighborhood"].map(neighborhood_premium)
price = base_price * premium

# Add realistic random noise (market variability)
noise = np.random.normal(0, 18000, N_SAMPLES)
price = (price + noise).clip(35000, None).round(-2)

df["SalePrice"] = price

# ------------------------------------------------------------------
# Deliberately inject a few data-quality issues so the preprocessing
# pipeline in this project has something meaningful to clean.
# ------------------------------------------------------------------
missing_idx = np.random.choice(df.index, size=int(0.03 * N_SAMPLES), replace=False)
df.loc[missing_idx, "LotSize"] = np.nan

missing_idx2 = np.random.choice(df.index, size=int(0.02 * N_SAMPLES), replace=False)
df.loc[missing_idx2, "GarageCars"] = np.nan

dup_rows = df.sample(15, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)

# A handful of extreme outliers
outlier_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_idx, "SalePrice"] = df.loc[outlier_idx, "SalePrice"] * 6

output_path = Path(__file__).resolve().parent / "house_prices.csv"
df.to_csv(output_path, index=False)
print(f"Sample dataset generated: {output_path.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
