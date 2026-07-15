"""
invoke_endpoint.py — Test the deployed SageMaker endpoint
==========================================================
Sends sample prediction requests to the live endpoint.

Usage:
    python scripts/invoke_endpoint.py
"""

import sys
import json
import boto3

sys.path.insert(0, ".")
from pipelines.config import load_config


def invoke():
    """Send test predictions to the endpoint."""
    config = load_config()
    region = config["aws"]["region"]
    endpoint_name = config["endpoint"]["name"]

    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Sample shipments
    samples = [
        {
            "name": "Surface: Mumbai → Delhi (1400 km, 3t)",
            "payload": {
                "distance_km": 1400.0, "log_distance": 7.24,
                "mode_surface": 1, "mode_air": 0,
                "booking_zone_id": 4, "dest_zone_id": 2, "is_same_zone": 0,
                "hour_of_day": 10, "day_of_week": 1, "is_weekend": 0, "month": 10,
                "route_frequency": 8, "weight_tonnes": 3.0,
                "distance_air_interaction": 0.0,
            },
        },
        {
            "name": "Air: Chennai → Delhi (2100 km, 0.5t)",
            "payload": {
                "distance_km": 2100.0, "log_distance": 7.65,
                "mode_surface": 0, "mode_air": 1,
                "booking_zone_id": 5, "dest_zone_id": 2, "is_same_zone": 0,
                "hour_of_day": 6, "day_of_week": 3, "is_weekend": 0, "month": 9,
                "route_frequency": 3, "weight_tonnes": 0.5,
                "distance_air_interaction": 2100.0,
            },
        },
        {
            "name": "Short surface: Same zone (80 km, 1t)",
            "payload": {
                "distance_km": 80.0, "log_distance": 4.39,
                "mode_surface": 1, "mode_air": 0,
                "booking_zone_id": 4, "dest_zone_id": 4, "is_same_zone": 1,
                "hour_of_day": 14, "day_of_week": 5, "is_weekend": 1, "month": 10,
                "route_frequency": 12, "weight_tonnes": 1.0,
                "distance_air_interaction": 0.0,
            },
        },
    ]

    print(f"Invoking endpoint: {endpoint_name}")
    print("=" * 60)

    for sample in samples:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(sample["payload"]),
        )
        result = json.loads(response["Body"].read().decode())
        co2_kg = result[0] if isinstance(result, list) else result

        print(f"\n{sample['name']}")
        print(f"  → CO₂ Prediction: {co2_kg:.2f} kg")

    print("\n" + "=" * 60)
    print("All predictions successful!")


if __name__ == "__main__":
    invoke()
