import json
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1200

# Features based on Swiss Housing Market & Kaggle i3-ml-21-22 competition
living_space = np.random.normal(120, 45, n_samples).clip(35, 320).round(1)
rooms = (living_space / 28.0 + np.random.normal(0, 0.4, n_samples)).clip(1.5, 8.5)
rooms = np.round(rooms * 2) / 2 # Typical Swiss 2.5, 3.5, 4.5 room notation
bathrooms = np.clip(np.round(1 + (living_space > 90) + (living_space > 170) + np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples)), 1, 4).astype(int)
floor_choices = list(range(0, 16))
floor_weights = np.array([24, 20, 18, 14, 8, 5, 4, 2, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1.0])
floor_p = floor_weights / floor_weights.sum()
floor = np.random.choice(floor_choices, size=n_samples, p=floor_p)
year_built = np.random.randint(1920, 2024, size=n_samples)
dist_center = np.random.exponential(6.0, size=n_samples).clip(0.4, 32.0).round(2)
has_balcony = np.random.choice([0, 1], size=n_samples, p=[0.25, 0.75])
parking_spaces = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.25, 0.50, 0.20, 0.05])
mountain_view = np.random.randint(1, 11, size=n_samples)
energy_efficiency = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.25, 0.35, 0.20, 0.10])
tax_multiplier = np.random.uniform(0.72, 1.35, size=n_samples).round(2)

# Price calculation in CHF (Swiss Francs)
base_price = 180000
price = (
    base_price
    + living_space * 6450
    + rooms * 32000
    + bathrooms * 41000
    + floor * 7200
    + (year_built - 1960) * 1150
    - dist_center * 19500
    + has_balcony * 38000
    + parking_spaces * 45000
    + mountain_view * 22000
    + energy_efficiency * 16500
    - (tax_multiplier - 1.0) * 140000
    + np.random.normal(0, 52000, size=n_samples)
)

price = np.maximum(price, 280000).round(-2).astype(int)

df = pd.DataFrame({
    "id": range(1, n_samples + 1),
    "LivingSpace": living_space,
    "Rooms": rooms,
    "Bathrooms": bathrooms,
    "Floor": floor,
    "YearBuilt": year_built,
    "DistanceCityCenter": dist_center,
    "HasBalcony": has_balcony,
    "ParkingSpaces": parking_spaces,
    "MountainViewScore": mountain_view,
    "EnergyEfficiency": energy_efficiency,
    "TaxMultiplier": tax_multiplier,
    "PriceCHF": price
})

# Save to CSV and JSON
df.to_csv("swiss_housing_sample.csv", index=False)
records = df.to_dict(orient="records")
with open("swiss_housing_sample.json", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Generated Swiss Housing dataset: {len(df)} rows.")
print("Summary stats of PriceCHF:")
print(df["PriceCHF"].describe())
