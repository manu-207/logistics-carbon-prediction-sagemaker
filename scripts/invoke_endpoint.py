"""
invoke_endpoint.py — Test the deployed SageMaker endpoint
==========================================================
Sends sample prediction requests to the live endpoint.

Usage:
    python scripts/invoke_endpoint.py
"""

import sys
import boto3

sys.path.insert(0, ".")
from pipelines.config import load_config


def invoke():
    """Send test predictions to the endpoint."""
    config = load_config()
    region = config["aws"]["region"]
    endpoint_name = config["endpoint"]["name"]

    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Feature order (must match training):
    # distance_km, log_distance, mode_surface, mode_air,
    # booking_zone_id, dest_zone_id, is_same_zone,
    # hour_of_day, day_of_week, is_weekend, month,
    # route_frequency, weight_tonnes, distance_air_interaction

    samples = [
        {
            "name": "Surface: Mumbai → Delhi (1400 km, 3t)",
            "csv": "1400.0,7.24,1,0,4,2,0,10,1,0,10,8,3.0,0.0",
        },
        {
            "name": "Air: Chennai → Delhi (2100 km, 0.5t)",
            "csv": "2100.0,7.65,0,1,5,2,0,6,3,0,9,3,0.5,2100.0",
        },
        {
            "name": "Short surface: Same zone (80 km, 1t)",
            "csv": "80.0,4.39,1,0,4,4,1,14,5,1,10,12,1.0,0.0",
        },
    ]

    print(f"Invoking endpoint: {endpoint_name}")
    print("=" * 60)

    for sample in samples:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=sample["csv"],
        )
        result = response["Body"].read().decode().strip()
        co2_kg = float(result)

        print(f"\n{sample['name']}")
        print(f"  → CO₂ Prediction: {co2_kg:.2f} kg")

    print("\n" + "=" * 60)
    print("All predictions successful! ✅")


if __name__ == "__main__":
    invoke()
