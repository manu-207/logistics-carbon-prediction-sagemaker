# ML Carbon Emissions — SageMaker MLOps Deployment

End-to-end CO₂ emission prediction using AWS SageMaker Pipelines.

## Architecture

```
S3 (Raw Data) → SageMaker Pipeline → Model Registry → SageMaker Endpoint
                    │
                    ├── Step 1: Data Prep (Processing Job)
                    ├── Step 2: Feature Engineering (Processing Job)
                    ├── Step 3: Training (XGBoost Training Job)
                    ├── Step 4: Evaluation (Processing Job)
                    ├── Step 5: Condition Check (R² > threshold?)
                    └── Step 6: Register Model → Deploy Endpoint
```

## Project Structure

```
Ml-carbon-emissions-sagemaker/
├── pipelines/
│   ├── __init__.py
│   ├── pipeline.py              # SageMaker Pipeline definition
│   ├── preprocess.py            # Data prep + feature engineering step
│   ├── evaluate.py              # Model evaluation step
│   └── config.py                # Pipeline configuration
├── inference/
│   ├── inference.py             # Custom inference script for endpoint
│   └── requirements.txt         # Endpoint dependencies
├── notebooks/
│   └── run_pipeline.ipynb       # Notebook to execute pipeline
├── scripts/
│   ├── deploy_endpoint.py       # Deploy model to real-time endpoint
│   ├── invoke_endpoint.py       # Test endpoint with sample data
│   └── cleanup.py               # Delete resources
├── data/
│   └── raw/shipments.csv        # Raw shipment data (upload to S3)
├── config.yaml                  # Central configuration
├── requirements.txt             # Development dependencies
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Upload data to S3
aws s3 cp data/raw/shipments.csv s3://YOUR-BUCKET/carbon-emissions/raw/shipments.csv

# 3. Run the pipeline
python pipelines/pipeline.py

# 4. Deploy endpoint
python scripts/deploy_endpoint.py

# 5. Test prediction
python scripts/invoke_endpoint.py
```

## Prerequisites

- AWS Account with SageMaker permissions
- IAM role with SageMaker, S3, ECR access
- Python 3.10+
- AWS CLI configured (`aws configure`)
