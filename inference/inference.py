"""
inference.py — Custom inference script for SageMaker Endpoint
==============================================================
Handles model loading, input parsing, prediction, and response formatting.
Deployed with the XGBoost model to the real-time endpoint.
"""

import os
import json
import numpy as np
import xgboost as xgb


FEATURE_COLUMNS = [
    "distance_km", "log_distance", "mode_surface", "mode_air",
    "booking_zone_id", "dest_zone_id", "is_same_zone",
    "hour_of_day", "day_of_week", "is_weekend", "month",
    "route_frequency", "weight_tonnes", "distance_air_interaction",
]


def model_fn(model_dir):
    """Load the XGBoost model from the model directory."""
    model_path = os.path.join(model_dir, "xgboost-model")
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data from request."""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Handle single prediction
        if isinstance(data, dict):
            features = [data.get(col, 0) for col in FEATURE_COLUMNS]
            return xgb.DMatrix(np.array([features]), feature_names=FEATURE_COLUMNS)

        # Handle batch prediction (list of dicts)
        elif isinstance(data, list):
            rows = [[item.get(col, 0) for col in FEATURE_COLUMNS] for item in data]
            return xgb.DMatrix(np.array(rows), feature_names=FEATURE_COLUMNS)

    elif request_content_type == "text/csv":
        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return xgb.DMatrix(df.values)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Run prediction on the input data."""
    predictions = model.predict(input_data)
    return predictions
