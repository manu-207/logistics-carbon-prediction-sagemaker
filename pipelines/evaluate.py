"""
evaluate.py — SageMaker Processing Job for model evaluation
=============================================================
Loads the trained model and test data, computes metrics,
writes evaluation report for pipeline condition step.
"""

import os
import json
import tarfile
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def evaluate():
    """Evaluate model on test set and write metrics report."""
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    output_dir = "/opt/ml/processing/evaluation"

    # Extract model from tar.gz
    model_tar = os.path.join(model_dir, "model.tar.gz")
    print(f"[evaluate] Extracting model from {model_tar}")
    with tarfile.open(model_tar, "r:gz") as tar:
        tar.extractall(path=model_dir)

    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))

    # Load test data (first column is target)
    test_path = os.path.join(test_dir, "test.csv")
    print(f"[evaluate] Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, header=None)

    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    # Predict
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # Compute metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(mean_absolute_percentage_error(y_test, y_pred))

    print(f"[evaluate] MAE:  {mae:.4f} kg")
    print(f"[evaluate] RMSE: {rmse:.4f} kg")
    print(f"[evaluate] R²:   {r2:.4f}")
    print(f"[evaluate] MAPE: {mape:.4%}")

    # Write evaluation report (used by ConditionStep)
    os.makedirs(output_dir, exist_ok=True)

    report = {
        "regression_metrics": {
            "mae": {"value": mae},
            "rmse": {"value": rmse},
            "r2_score": {"value": r2},
            "mape": {"value": mape},
        }
    }

    report_path = os.path.join(output_dir, "evaluation.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[evaluate] Report saved to {report_path}")


if __name__ == "__main__":
    evaluate()
