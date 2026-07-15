"""
config.py — Load pipeline configuration from config.yaml
"""

import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config() -> dict:
    """Load the pipeline configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_s3_uri(config: dict, path: str) -> str:
    """Build S3 URI from config."""
    bucket = config["aws"]["s3_bucket"]
    prefix = config["aws"]["s3_prefix"]
    return f"s3://{bucket}/{prefix}/{path}"


def get_role_arn(config: dict) -> str:
    """Get the SageMaker execution role ARN."""
    import boto3
    iam = boto3.client("iam")
    role_name = config["aws"]["role_name"]
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]
