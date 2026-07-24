"""
generate_data.py

Run:
    python generate_data.py
Produces:
    housing_data.csv  (in the same folder)
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 2500


def generate_housing_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area_sqft = rng.normal(1800, 650, n_samples).clip(400, 6000)
    bedrooms = rng.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.20, 0.32, 0.28, 0.12, 0.03])
    bathrooms = np.clip(np.round(bedrooms * rng.uniform(0.5, 1.1, n_samples)), 1, 5)
    stories = rng.choice([1, 2, 3, 4], n_samples, p=[0.35, 0.45, 0.15, 0.05])
    age_years = rng.exponential(15, n_samples).clip(0, 90)
    garage_spaces = rng.choice([0, 1, 2, 3], n_samples, p=[0.15, 0.30, 0.42, 0.13])


    location_score = rng.beta(2.2, 2.0, n_samples) * 9 + 1

    # amenities (0/1 flags)
    has_pool = rng.choice([0, 1], n_samples, p=[0.82, 0.18])
    has_basement = rng.choice([0, 1], n_samples, p=[0.55, 0.45])
    near_main_road = rng.choice([0, 1], n_samples, p=[0.65, 0.35])
    furnishing = rng.choice(["unfurnished", "semi-furnished", "furnished"], n_samples, p=[0.45, 0.35, 0.20])
    condition = rng.choice(["poor", "fair", "good", "excellent"], n_samples, p=[0.08, 0.27, 0.45, 0.20])

    condition_map = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
    furnishing_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    condition_score = np.vectorize(condition_map.get)(condition)
    furnishing_score = np.vectorize(furnishing_map.get)(furnishing)

    base_price = (
        area_sqft * 95
        + bedrooms * 8500
        + bathrooms * 6200
        + stories * 4000
        + garage_spaces * 5200
        + location_score * 9800
        + condition_score * 7000
        + furnishing_score * 4500
        + has_pool * 15000
        + has_basement * 9000
        - near_main_road * 6000
    )

    age_penalty = 1 - (0.55 * (1 - np.exp(-age_years / 20)))
    location_multiplier = 1 + (location_score / 10) ** 2 * 0.5

    price = base_price * age_penalty * location_multiplier

    noise = rng.normal(1.0, 0.07, n_samples)
    price = price * noise
    price = price.clip(25000, None)

    df = pd.DataFrame({
        "area_sqft": area_sqft.round(0).astype(int),
        "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int),
        "stories": stories.astype(int),
        "garage_spaces": garage_spaces.astype(int),
        "age_years": age_years.round(1),
        "location_score": location_score.round(2),
        "has_pool": has_pool.astype(int),
        "has_basement": has_basement.astype(int),
        "near_main_road": near_main_road.astype(int),
        "furnishing": furnishing,
        "condition": condition,
        "price": price.round(-2).astype(int),  # round to nearest 100
    })

    return df


if __name__ == "__main__":
    df = generate_housing_data()
    out_path = "housing_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print("\nPrice stats:")
    print(df["price"].describe())
