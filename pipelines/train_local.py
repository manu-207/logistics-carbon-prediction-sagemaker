"""
train_local.py — Local Training Script
========================================
Trains XGBoost model locally (on GitHub runner) using
preprocessed data from S3. Uploads model artifact to S3.

Usage:
    python pipelines/train_local.py
"""

import os
import sys
import json
import tarfile
import tempfile

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config, get_role_arn


def download_s3_file(s3_client, bucket, key, local_path):
    """Download a file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  Downloading s3://{bucket}/{key} -> {local_path}")
    s3_client.download_file(bucket, key, local_path)


def upload_s3_file(s3_client, local_path, bucket, key):
    """Upload a file to S3."""
    print(f"  Uploading {local_path} -> s3://{bucket}/{key}")
    s3_client.upload_file(local_path, bucket, key)


def main():
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    prefix = config["aws"]["s3_prefix"]
    hyperparams = config["model"]["hyperparameters"]
    min_r2 = config["evaluation"]["min_r2_threshold"]

    s3 = boto3.client("s3", region_name=region)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Download preprocessed data from S3 ────────────────────────────────
        print("\n[train_local] Step 1: Downloading preprocessed data from S3...")
        train_path = os.path.join(tmpdir, "train.csv")
        test_path = os.path.join(tmpdir, "test.csv")

        download_s3_file(s3, bucket, f"{prefix}/processed/train/train.csv", train_path)
        download_s3_file(s3, bucket, f"{prefix}/processed/test/test.csv", test_path)

        # ── Load data ─────────────────────────────────────────────────────────
        print("\n[train_local] Step 2: Loading data...")
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)

        y_train = train_df.iloc[:, 0].values
        X_train = train_df.iloc[:, 1:].values
        y_test = test_df.iloc[:, 0].values
        X_test = test_df.iloc[:, 1:].values

        print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"  Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")

        # ── Train XGBoost ─────────────────────────────────────────────────────
        print("\n[train_local] Step 3: Training XGBoost model...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "max_depth": int(hyperparams["max_depth"]),
            "eta": float(hyperparams["eta"]),
            "subsample": float(hyperparams["subsample"]),
            "colsample_bytree": float(hyperparams["colsample_bytree"]),
            "objective": hyperparams["objective"],
            "eval_metric": hyperparams["eval_metric"],
            "alpha": float(hyperparams.get("alpha", 0.1)),
            "lambda": float(hyperparams.get("lambda", 1.0)),
        }
        num_round = int(hyperparams["num_round"])
        early_stopping = int(hyperparams.get("early_stopping_rounds", 15))

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dtest, "validation")],
            early_stopping_rounds=early_stopping,
            verbose_eval=50,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        print("\n[train_local] Step 4: Evaluating model...")
        y_pred = model.predict(dtest)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        print(f"  MAE:  {mae:.4f} kg")
        print(f"  RMSE: {rmse:.4f} kg")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.4%}")

        # ── Write evaluation report ───────────────────────────────────────────
        report = {
            "regression_metrics": {
                "mae": {"value": mae},
                "rmse": {"value": rmse},
                "r2_score": {"value": r2},
                "mape": {"value": mape},
            }
        }
        eval_path = os.path.join(tmpdir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(report, f, indent=2)

        # Upload evaluation report to S3
        upload_s3_file(s3, eval_path, bucket, f"{prefix}/evaluation/evaluation.json")

        # ── Check R² threshold ────────────────────────────────────────────────
        print(f"\n[train_local] Step 5: Checking R² threshold ({r2:.4f} >= {min_r2})...")
        if r2 < min_r2:
            print(f"  ❌ R² ({r2:.4f}) below threshold ({min_r2}). Model NOT registered.")
            sys.exit(1)

        print(f"  ✅ R² threshold met!")

        # ── Package model as tar.gz (SageMaker format) ────────────────────────
        print("\n[train_local] Step 6: Packaging and uploading model...")
        model_file = os.path.join(tmpdir, "xgboost-model")
        model.save_model(model_file)

        model_tar_path = os.path.join(tmpdir, "model.tar.gz")
        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(model_file, arcname="xgboost-model")

        # Upload to S3
        model_s3_key = f"{prefix}/models/model.tar.gz"
        upload_s3_file(s3, model_tar_path, bucket, model_s3_key)
        model_s3_uri = f"s3://{bucket}/{model_s3_key}"
        print(f"  Model artifact: {model_s3_uri}")

        # ── Register model in SageMaker Model Registry ────────────────────────
        print("\n[train_local] Step 7: Registering model in SageMaker Model Registry...")
        sm = boto3.client("sagemaker", region_name=region)

        # Ensure model package group exists
        model_group = config["endpoint"]["model_package_group"]
        try:
            sm.describe_model_package_group(ModelPackageGroupName=model_group)
        except sm.exceptions.ClientError:
            sm.create_model_package_group(
                ModelPackageGroupName=model_group,
                ModelPackageGroupDescription="Carbon emissions prediction models",
            )
            print(f"  Created model package group: {model_group}")

        # Get XGBoost image URI for inference
        import sagemaker
        xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

        # Register
        response = sm.create_model_package(
            ModelPackageGroupName=model_group,
            ModelPackageDescription=f"XGBoost carbon emissions model (R²={r2:.4f})",
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": xgb_image,
                        "ModelDataUrl": model_s3_uri,
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.t2.medium", "ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"],
            },
            ModelApprovalStatus="PendingManualApproval",
            ModelMetrics={
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": f"s3://{bucket}/{prefix}/evaluation/evaluation.json",
                    }
                }
            },
        )

        model_package_arn = response["ModelPackageArn"]
        print(f"  ✅ Model registered: {model_package_arn}")
        print(f"\n{'='*60}")
        print(f"  Pipeline completed successfully!")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Model ARN: {model_package_arn}")
        print(f"{'='*60}")
        print(f"\n  Next steps:")
        print(f"    1. Approve: aws sagemaker update-model-package \\")
        print(f"         --model-package-arn {model_package_arn} \\")
        print(f"         --model-approval-status Approved")
        print(f"    2. Deploy: Trigger the deploy-endpoint workflow")


if __name__ == "__main__":
    main()
