"""
OptiCrop - Synthetic Agricultural Dataset Generator
-----------------------------------------------------
Generates a realistic crop-recommendation dataset based on published
agronomic ranges for N, P, K, temperature, humidity, pH and rainfall
for 22 common crops. Each crop's parameters are sampled from a normal
distribution around agronomically accepted means, then clipped to
valid physical ranges.

Output: data/crop_data.csv
Columns: N, P, K, temperature, humidity, ph, rainfall, label
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Agronomic profiles: (N_mean, N_std, P_mean, P_std, K_mean, K_std,
#                       temp_mean, temp_std, humidity_mean, humidity_std,
#                       ph_mean, ph_std, rainfall_mean, rainfall_std)
CROP_PROFILES = {
    "rice":        (80, 15, 45, 10, 40, 10, 24, 2.5, 82, 5,  6.4, 0.4, 220, 30),
    "maize":       (80, 15, 40, 10, 20, 8,  23, 3.0, 63, 8,  6.2, 0.5, 90,  20),
    "chickpea":    (40, 8,  60, 10, 80, 10, 19, 3.0, 16, 4,  7.3, 0.4, 80,  15),
    "kidneybeans": (20, 6,  60, 10, 20, 6,  18, 2.5, 21, 4,  5.7, 0.4, 105, 20),
    "pigeonpeas":  (20, 6,  60, 10, 20, 6,  27, 3.5, 48, 8,  5.8, 0.5, 150, 25),
    "mothbeans":   (20, 6,  48, 8,  20, 6,  28, 2.5, 53, 8,  6.8, 0.5, 50,  15),
    "mungbean":    (20, 6,  47, 8,  20, 6,  28, 2.0, 85, 5,  6.7, 0.4, 48,  10),
    "blackgram":   (40, 8,  60, 10, 20, 6,  29, 2.5, 65, 6,  7.1, 0.4, 68,  15),
    "lentil":      (18, 5,  68, 10, 19, 5,  24, 3.0, 65, 6,  6.9, 0.4, 46,  12),
    "pomegranate": (18, 5,  18, 5,  40, 8,  21, 3.0, 90, 4,  6.4, 0.4, 108, 20),
    "banana":      (100,15, 82, 10, 50, 10, 27, 2.0, 80, 4,  5.9, 0.4, 105, 20),
    "mango":       (20, 6,  27, 6,  30, 8,  31, 2.5, 50, 8,  5.7, 0.4, 95,  20),
    "grapes":      (18, 5,  130,15, 200,20, 24, 2.5, 82, 4,  6.0, 0.4, 70,  15),
    "watermelon":  (100,15, 17, 5,  50, 10, 25, 2.5, 85, 4,  6.5, 0.4, 45,  12),
    "muskmelon":   (100,15, 17, 5,  50, 10, 28, 2.0, 92, 3,  6.4, 0.4, 24,  8),
    "apple":       (21, 5,  135,15, 200,20, 22, 2.5, 92, 3,  5.9, 0.4, 112, 20),
    "orange":      (19, 5,  16, 5,  10, 4,  22, 3.0, 92, 3,  7.0, 0.4, 110, 20),
    "papaya":      (50, 10, 59, 8,  50, 10, 33, 2.5, 92, 3,  6.7, 0.4, 142, 25),
    "coconut":     (22, 6,  17, 5,  31, 8,  27, 2.0, 95, 2,  5.9, 0.4, 175, 25),
    "cotton":      (118,15, 46, 8,  20, 6,  24, 2.5, 80, 5,  6.9, 0.4, 80,  15),
    "jute":        (78, 12, 47, 8,  40, 8,  25, 2.0, 80, 4,  6.7, 0.4, 175, 25),
    "coffee":      (101,15, 28, 6,  30, 8,  25, 2.5, 58, 8,  6.8, 0.4, 158, 25),
}

VALID_RANGES = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (8, 45),
    "humidity": (10, 100),
    "ph": (3.5, 9.5),
    "rainfall": (15, 300),
}


def sample_crop(name, params, n_samples):
    (n_m, n_s, p_m, p_s, k_m, k_s, t_m, t_s, h_m, h_s,
     ph_m, ph_s, r_m, r_s) = params

    rows = {
        "N": np.random.normal(n_m, n_s, n_samples),
        "P": np.random.normal(p_m, p_s, n_samples),
        "K": np.random.normal(k_m, k_s, n_samples),
        "temperature": np.random.normal(t_m, t_s, n_samples),
        "humidity": np.random.normal(h_m, h_s, n_samples),
        "ph": np.random.normal(ph_m, ph_s, n_samples),
        "rainfall": np.random.normal(r_m, r_s, n_samples),
    }

    df = pd.DataFrame(rows)
    for col, (lo, hi) in VALID_RANGES.items():
        df[col] = df[col].clip(lo, hi)

    df["label"] = name
    return df


def build_dataset(n_per_crop=120):
    frames = [sample_crop(crop, params, n_per_crop)
              for crop, params in CROP_PROFILES.items()]
    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data = data.round({"N": 1, "P": 1, "K": 1, "temperature": 2,
                        "humidity": 2, "ph": 2, "rainfall": 2})
    return data


if __name__ == "__main__":
    df = build_dataset(n_per_crop=120)
    df.to_csv("data/crop_data.csv", index=False)
    print(f"Generated {len(df)} rows for {df['label'].nunique()} crops.")
    print(df.head())
