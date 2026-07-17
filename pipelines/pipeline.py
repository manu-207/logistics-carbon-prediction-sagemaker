"""
pipeline.py — SageMaker Pipeline Definition
=============================================
Defines the full MLOps pipeline:
1. Preprocessing (data prep + feature engineering)
2. Training (XGBoost)
3. Evaluation (metrics computation)
4. Condition (R² > threshold?)
5. Model Registration (to SageMaker Model Registry)

Usage:
    python pipelines/pipeline.py
"""

import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.processing import (
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from config import load_config, get_s3_uri, get_role_arn


def create_pipeline():
    """Create and return the SageMaker Pipeline."""
    config = load_config()
    region = config["aws"]["region"]
    bucket = config["aws"]["s3_bucket"]
    prefix = config["aws"]["s3_prefix"]

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))
    role = get_role_arn(config)

    # ── Pipeline Parameters ───────────────────────────────────────────────────
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{bucket}/{prefix}/raw/shipments.csv",
    )
    min_r2_threshold = ParameterFloat(
        name="MinR2Threshold",
        default_value=config["evaluation"]["min_r2_threshold"],
    )
    instance_type_processing = ParameterString(
        name="ProcessingInstanceType",
        default_value=config["pipeline"]["instance_type_processing"],
    )
    instance_type_training = ParameterString(
        name="TrainingInstanceType",
        default_value=config["pipeline"]["instance_type_training"],
    )

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    sklearn_processor = ScriptProcessor(
        command=["python3"],
        image_uri=sagemaker.image_uris.retrieve("sklearn", region, version="1.2-1"),
        instance_type=instance_type_processing,
        instance_count=1,
        role=role,
        sagemaker_session=sess,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{prefix}/processed/train",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/{prefix}/processed/test",
            ),
        ],
        code="pipelines/preprocess.py",
    )

    # ── Step 2: Training (XGBoost) ────────────────────────────────────────────
    xgb_image = sagemaker.image_uris.retrieve("xgboost", region, version="1.7-1")

    xgb_estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=1,
        instance_type=instance_type_training,
        output_path=f"s3://{bucket}/{prefix}/models",
        sagemaker_session=sess,
        hyperparameters=config["model"]["hyperparameters"],
        use_spot_instances=True,
        max_wait=3600,
        max_run=3600,
    )

    step_train = TrainingStep(
        name="TrainXGBoost",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ── Step 3: Evaluation ────────────────────────────────────────────────────
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{prefix}/evaluation",
            ),
        ],
        code="pipelines/evaluate.py",
        property_files=[evaluation_report],
    )

    # ── Step 4: Condition (R² check) ─────────────────────────────────────────
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=f"s3://{bucket}/{prefix}/evaluation/evaluation.json",
            content_type="application/json",
        )
    )

    step_register = RegisterModel(
        name="RegisterModel",
        estimator=xgb_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.t2.medium"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=config["endpoint"]["model_package_group"],
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    cond_r2 = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name="EvaluateModel",
            property_file=evaluation_report,
            json_path="regression_metrics.r2_score.value",
        ),
        right=min_r2_threshold,
    )

    step_condition = ConditionStep(
        name="CheckR2Score",
        conditions=[cond_r2],
        if_steps=[step_register],
        else_steps=[],
    )

    # ── Create Pipeline ───────────────────────────────────────────────────────
    pipeline = Pipeline(
        name=config["pipeline"]["name"],
        parameters=[
            input_data,
            min_r2_threshold,
            instance_type_processing,
            instance_type_training,
        ],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=sess,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = create_pipeline()

    # Upsert (create or update) the pipeline
    pipeline.upsert(role_arn=get_role_arn(load_config()))
    print(f"Pipeline '{pipeline.name}' created/updated.")

    # Start execution
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print("Monitor at: https://console.aws.amazon.com/sagemaker/home#/pipelines")
