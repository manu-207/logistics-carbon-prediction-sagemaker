"""
cleanup.py — Delete SageMaker resources
=========================================
Deletes endpoint, endpoint config, and model to avoid charges.

Usage:
    python scripts/cleanup.py
"""

import sys
import boto3

sys.path.insert(0, ".")
from pipelines.config import load_config


def cleanup():
    """Delete all SageMaker resources."""
    config = load_config()
    region = config["aws"]["region"]
    endpoint_name = config["endpoint"]["name"]

    sm = boto3.client("sagemaker", region_name=region)

    # Delete endpoint
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Deleted endpoint: {endpoint_name}")
    except sm.exceptions.ClientError:
        print(f"⚠️ Endpoint '{endpoint_name}' not found")

    # Delete endpoint config
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"✅ Deleted endpoint config: {endpoint_name}")
    except sm.exceptions.ClientError:
        print(f"⚠️ Endpoint config not found")

    # Delete pipeline
    pipeline_name = config["pipeline"]["name"]
    try:
        sm.delete_pipeline(PipelineName=pipeline_name)
        print(f"✅ Deleted pipeline: {pipeline_name}")
    except sm.exceptions.ClientError:
        print(f"⚠️ Pipeline '{pipeline_name}' not found")

    print("\n🧹 Cleanup complete. No ongoing charges.")


if __name__ == "__main__":
    cleanup()
