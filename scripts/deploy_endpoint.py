"""
deploy_endpoint.py — Deploy approved model to SageMaker real-time endpoint
============================================================================
Fetches the latest approved model from the Model Registry and deploys it.
If endpoint already exists, updates it (zero-downtime). Otherwise creates new.

Usage:
    python scripts/deploy_endpoint.py
"""

import sys
import boto3
import sagemaker
from sagemaker.model import Model

sys.path.insert(0, ".")
from pipelines.config import load_config, get_role_arn


def deploy():
    """Deploy the latest approved model to a real-time endpoint."""
    config = load_config()
    region = config["aws"]["region"]
    endpoint_name = config["endpoint"]["name"]
    model_package_group = config["endpoint"]["model_package_group"]
    instance_type = config["pipeline"]["instance_type_endpoint"]
    instance_count = config["pipeline"]["instance_count_endpoint"]

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    sm_client = boto3.client("sagemaker", region_name=region)
    role = get_role_arn(config)

    # Get latest approved model package
    print(f"[deploy] Looking for approved models in '{model_package_group}'...")
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )

    packages = response.get("ModelPackageSummaryList", [])
    if not packages:
        print("[deploy] No approved models found. Approve a model first:")
        print(f"  aws sagemaker update-model-package "
              f"--model-package-arn <ARN> "
              f"--model-approval-status Approved")
        return

    model_package_arn = packages[0]["ModelPackageArn"]
    print(f"[deploy] Using model: {model_package_arn}")

    # Create model from package
    model = sagemaker.ModelPackage(
        role=role,
        model_package_arn=model_package_arn,
        sagemaker_session=sess,
    )

    # Check if endpoint already exists
    endpoint_exists = False
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sm_client.exceptions.ClientError:
        pass

    print(f"[deploy] Deploying to endpoint '{endpoint_name}'...")
    print(f"[deploy] Instance: {instance_type} x {instance_count}")

    if endpoint_exists:
        # Update existing endpoint (zero-downtime blue/green deployment)
        print("[deploy] Endpoint exists — updating with new model (zero-downtime)...")
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            update_endpoint=True,
        )
    else:
        # Create new endpoint
        print("[deploy] Creating new endpoint...")
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )

    print(f"[deploy] ✅ Endpoint '{endpoint_name}' is now live!")
    print(f"[deploy] Test with: python scripts/invoke_endpoint.py")


if __name__ == "__main__":
    deploy()
