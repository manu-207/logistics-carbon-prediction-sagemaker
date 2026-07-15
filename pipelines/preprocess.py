"""
preprocess.py — SageMaker Processing Job script
=================================================
Runs as a Processing Job inside SageMaker.
Reads raw CSV from /opt/ml/processing/input/
Outputs train/test CSVs to /opt/ml/processing/output/
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ── Emission factors ──────────────────────────────────────────────────────────
EMISSION_FACTOR_SURFACE = 0.062
EMISSION_FACTOR_AIR = 0.602

# ── Zone mapping ──────────────────────────────────────────────────────────────
PINCODE_ZONE_MAP = {
    1: "delhi_ncr", 2: "uttar_pradesh", 3: "west_india",
    4: "maharashtra", 5: "south_central", 6: "south",
    7: "east", 8: "central_east", 9: "army_post",
}

ZONE_CENTROIDS = {
    "delhi_ncr": (28.6, 77.2), "uttar_pradesh": (26.8, 80.9),
    "west_india": (24.5, 72.5), "maharashtra": (19.7, 75.7),
    "south_central": (15.9, 78.5), "south": (11.0, 78.0),
    "east": (22.5, 87.5), "central_east": (25.0, 85.0),
    "army_post": (28.6, 77.2),
}


def get_zone(pincode):
    """Map pincode to geographic zone."""
    if pd.isna(pincode):
        return "unknown"
    prefix = int(str(int(pincode))[0])
    return PINCODE_ZONE_MAP.get(prefix, "unknown")


def estimate_distance(origin_pin, dest_pin):
    """Estimate road distance using Haversine + road factor."""
    origin_zone = get_zone(origin_pin)
    dest_zone = get_zone(dest_pin)

    if origin_zone == "unknown" or dest_zone == "unknown":
        return 500.0

    lat1, lon1 = ZONE_CENTROIDS.get(origin_zone, (22.0, 78.0))
    lat2, lon2 = ZONE_CENTROIDS.get(dest_zone, (22.0, 78.0))

    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))
    straight_line = R * c
    road_distance = straight_line * 1.3

    if road_distance < 50:
        pin_diff = abs(int(origin_pin) - int(dest_pin))
        road_distance = max(50, pin_diff * 0.05)

    return round(road_distance, 1)


def process():
    """Main preprocessing function for SageMaker Processing Job."""
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"

    # Read raw data
    raw_path = os.path.join(input_dir, "shipments.csv")
    print(f"[preprocess] Reading {raw_path}")
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"[preprocess] Raw shape: {df.shape}")

    # Clean
    df["BOOKING_PINCODE"] = pd.to_numeric(df["BOOKING_PINCODE"], errors="coerce")
    df["DESTINATION_PINCODE"] = pd.to_numeric(df["DESTINATION_PINCODE"], errors="coerce")
    df["MOD"] = df["MOD"].str.upper().str.strip()
    df = df[df["MOD"].isin(["SURFACE", "AIR"])].copy()
    df = df.dropna(subset=["BOOKING_PINCODE", "DESTINATION_PINCODE"]).reset_index(drop=True)

    # Feature engineering
    np.random.seed(42)

    df["distance_km"] = df.apply(
        lambda r: estimate_distance(r["BOOKING_PINCODE"], r["DESTINATION_PINCODE"]), axis=1
    )
    df["log_distance"] = np.log1p(df["distance_km"])
    df["mode_surface"] = (df["MOD"] == "SURFACE").astype(int)
    df["mode_air"] = (df["MOD"] == "AIR").astype(int)

    zone_list = sorted(set(PINCODE_ZONE_MAP.values()) | {"unknown"})
    zone_to_int = {z: i for i, z in enumerate(zone_list)}
    df["booking_zone_id"] = df["BOOKING_PINCODE"].apply(get_zone).map(zone_to_int).fillna(7).astype(int)
    df["dest_zone_id"] = df["DESTINATION_PINCODE"].apply(get_zone).map(zone_to_int).fillna(7).astype(int)
    df["is_same_zone"] = (df["booking_zone_id"] == df["dest_zone_id"]).astype(int)

    df["DLVRY_DT"] = pd.to_datetime(df["DLVRY_DT"], errors="coerce")
    df["hour_of_day"] = df["DLVRY_DT"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["DLVRY_DT"].dt.dayofweek.fillna(2).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["DLVRY_DT"].dt.month.fillna(9).astype(int)

    route_key = df["BOOKING_PINCODE"].astype(str) + "_" + df["DESTINATION_PINCODE"].astype(str)
    df["route_frequency"] = route_key.map(route_key.value_counts()).astype(int)

    base_weight = np.where(
        df["mode_air"] == 1,
        np.random.lognormal(-0.5, 0.8, len(df)),
        np.random.lognormal(1.0, 1.0, len(df)),
    )
    df["weight_tonnes"] = np.clip(base_weight, 0.01, 25.0)
    df["distance_air_interaction"] = df["distance_km"] * df["mode_air"]

    # CO₂ target
    factor = np.where(df["mode_air"] == 1, EMISSION_FACTOR_AIR, EMISSION_FACTOR_SURFACE)
    noise = np.random.uniform(0.85, 1.15, len(df))
    df["co2_kg"] = df["distance_km"] * factor * df["weight_tonnes"] * noise

    # Select features
    feature_cols = [
        "distance_km", "log_distance", "mode_surface", "mode_air",
        "booking_zone_id", "dest_zone_id", "is_same_zone",
        "hour_of_day", "day_of_week", "is_weekend", "month",
        "route_frequency", "weight_tonnes", "distance_air_interaction",
    ]
    target = "co2_kg"

    df_final = df[feature_cols + [target]]

    # Train/test split
    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=42)

    # SageMaker XGBoost expects target as first column, no header
    train_out = pd.concat([train_df[target], train_df[feature_cols]], axis=1)
    test_out = pd.concat([test_df[target], test_df[feature_cols]], axis=1)

    # Save outputs
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    train_out.to_csv(os.path.join(output_dir, "train", "train.csv"), index=False, header=False)
    test_out.to_csv(os.path.join(output_dir, "test", "test.csv"), index=False, header=False)

    # Save feature names for inference
    import json
    meta = {"feature_columns": feature_cols, "target": target}
    with open(os.path.join(output_dir, "train", "feature_meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"[preprocess] Train: {len(train_out)}, Test: {len(test_out)}")
    print(f"[preprocess] Features: {feature_cols}")
    print("[preprocess] Done.")


if __name__ == "__main__":
    process()
