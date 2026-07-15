# ML Carbon Emissions — SageMaker Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                       │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     SageMaker Pipeline                                │  │
│  │                                                                       │  │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │  │
│  │  │  Step 1  │───▶│  Step 2  │───▶│  Step 3  │───▶│   Step 4     │   │  │
│  │  │Preprocess│    │  Train   │    │ Evaluate │    │ Condition    │   │  │
│  │  │(sklearn) │    │(XGBoost) │    │(sklearn) │    │ R² ≥ 0.5?   │   │  │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └──────┬───────┘   │  │
│  │       │                │                │                 │           │  │
│  │       ▼                ▼                ▼                 ▼           │  │
│  │  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │  │
│  │  │S3: train│    │S3: model │    │S3: eval  │    │   Step 5     │   │  │
│  │  │S3: test │    │.tar.gz   │    │.json     │    │  Register    │   │  │
│  │  └─────────┘    └──────────┘    └──────────┘    │   Model      │   │  │
│  │                                                  └──────┬───────┘   │  │
│  └─────────────────────────────────────────────────────────┼───────────┘  │
│                                                             │              │
│                                                             ▼              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    SageMaker Model Registry                           │ │
│  │                                                                       │ │
│  │   Model: carbon-emissions-models                                      │ │
│  │   Status: PendingManualApproval → Approved                            │ │
│  │   Versions: v1, v2, v3...                                             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼ (after approval)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    SageMaker Real-Time Endpoint                        │ │
│  │                                                                       │ │
│  │   Endpoint: carbon-emissions-endpoint                                 │ │
│  │   Instance: ml.m5.large                                               │ │
│  │   Auto-scaling: 1-4 instances                                         │ │
│  │                                                                       │ │
│  │   POST /invocations                                                   │ │
│  │   Input: JSON {features}  →  Output: CO₂ prediction (kg)             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ▲                                              │
│                              │                                              │
│  ┌──────────┐    ┌──────────┴───────────┐    ┌───────────────────────┐    │
│  │   S3     │    │    API Gateway /      │    │    CloudWatch         │    │
│  │  Bucket  │    │    Lambda (optional)  │    │    Monitoring         │    │
│  └──────────┘    └──────────────────────┘    └───────────────────────┘    │
│                              ▲                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Client / App /    │
                    │   Dashboard         │
                    └─────────────────────┘
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
│                                                                     │
│  Raw CSV            Features              Model              Predict │
│  (S3)              (S3)                  (Registry)         (Endpoint)│
│                                                                     │
│  shipments.csv ──▶ train.csv  ──────▶  model.tar.gz ──▶  CO₂ (kg)  │
│                    test.csv             xgboost-model                │
│                                                                     │
│  Columns:          14 Features:         XGBoost:          Response:  │
│  - WAYBL_NUM       - distance_km        - 300 trees       - co2_kg   │
│  - PINCODE         - log_distance       - depth 7         - category │
│  - MOD             - mode_surface/air   - lr 0.08                   │
│  - DLVRY_DT        - zone_ids                                       │
│  - BRANCH          - temporal                                       │
│                    - weight                                          │
│                    - interactions                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Steps Detail

### Step 1: Preprocess (Processing Job)

```
┌─────────────────────────────────────────────────────┐
│  SageMaker Processing Job (sklearn container)       │
│                                                     │
│  Input:  s3://bucket/raw/shipments.csv              │
│                                                     │
│  Actions:                                           │
│  1. Load & clean CSV (pincodes, dates, modes)       │
│  2. Compute distance (Haversine × 1.3 road factor) │
│  3. Engineer 14 features                            │
│  4. Compute CO₂ target (emission factor formula)    │
│  5. Train/test split (80/20)                        │
│  6. Save as headerless CSV (XGBoost format)         │
│                                                     │
│  Output: s3://bucket/processed/train/train.csv      │
│          s3://bucket/processed/test/test.csv        │
└─────────────────────────────────────────────────────┘
```

### Step 2: Train (Training Job)

```
┌─────────────────────────────────────────────────────┐
│  SageMaker Training Job (XGBoost 1.7 container)     │
│                                                     │
│  Input:  s3://bucket/processed/train/train.csv      │
│          s3://bucket/processed/test/test.csv        │
│                                                     │
│  Hyperparameters:                                   │
│    max_depth: 7                                     │
│    eta: 0.08                                        │
│    num_round: 300                                   │
│    early_stopping_rounds: 15                        │
│    objective: reg:squarederror                      │
│    eval_metric: rmse                                │
│                                                     │
│  Output: s3://bucket/models/model.tar.gz            │
│          (contains xgboost-model binary)            │
└─────────────────────────────────────────────────────┘
```

### Step 3: Evaluate (Processing Job)

```
┌─────────────────────────────────────────────────────┐
│  SageMaker Processing Job (sklearn container)       │
│                                                     │
│  Input:  model.tar.gz + test.csv                    │
│                                                     │
│  Computes:                                          │
│    - MAE  (Mean Absolute Error)                     │
│    - RMSE (Root Mean Squared Error)                 │
│    - R²   (Coefficient of Determination)            │
│    - MAPE (Mean Absolute Percentage Error)          │
│                                                     │
│  Output: s3://bucket/evaluation/evaluation.json     │
│          {                                          │
│            "regression_metrics": {                  │
│              "r2_score": {"value": 0.68},           │
│              "rmse": {"value": 214.2}              │
│            }                                        │
│          }                                          │
└─────────────────────────────────────────────────────┘
```

### Step 4: Condition Check

```
┌─────────────────────────────────────────────────────┐
│  IF r2_score ≥ 0.5:                                 │
│    → Proceed to Step 5 (Register Model)             │
│                                                     │
│  ELSE:                                              │
│    → Pipeline stops (model not good enough)         │
│    → Alert team to investigate                      │
└─────────────────────────────────────────────────────┘
```

### Step 5: Register Model

```
┌─────────────────────────────────────────────────────┐
│  SageMaker Model Registry                           │
│                                                     │
│  Group: carbon-emissions-models                     │
│  Version: auto-incremented                          │
│  Status: PendingManualApproval                      │
│                                                     │
│  Contains:                                          │
│  - Model artifact (model.tar.gz)                    │
│  - Evaluation metrics                               │
│  - Instance type recommendations                    │
│  - Content types (text/csv, application/json)       │
│                                                     │
│  Approval: Manual (human reviews metrics first)     │
└─────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PATH                                │
│                                                                 │
│  Client Request                                                 │
│  {                                                              │
│    "distance_km": 1400,                                         │
│    "mode_surface": 1,                                           │
│    "weight_tonnes": 3.0,                                        │
│    ...                                                          │
│  }                                                              │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────┐                                       │
│  │  API Gateway         │  (optional — adds auth, rate limit)   │
│  └──────────┬──────────┘                                       │
│             ▼                                                   │
│  ┌─────────────────────────────────────────────────────┐       │
│  │         SageMaker Real-Time Endpoint                │       │
│  │                                                     │       │
│  │  ┌─────────────┐   ┌─────────────┐                │       │
│  │  │ inference.py │   │  XGBoost    │                │       │
│  │  │             │──▶│  Model      │                │       │
│  │  │ input_fn()  │   │  predict()  │                │       │
│  │  │ predict_fn()│   │             │                │       │
│  │  └─────────────┘   └─────────────┘                │       │
│  │                                                     │       │
│  │  Instance: ml.m5.large                              │       │
│  │  Auto-scaling: 1-4 instances                        │       │
│  │  Latency: ~10-50ms                                  │       │
│  └─────────────────────────────────────────────────────┘       │
│             │                                                   │
│             ▼                                                   │
│  Response: [456.2]  (CO₂ in kg)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## AWS Services Used

| Service | Role | Cost Model |
|---------|------|------------|
| **S3** | Data storage (raw, processed, models) | $0.023/GB/month |
| **SageMaker Processing** | Data prep & evaluation jobs | Per-second billing |
| **SageMaker Training** | Model training | Per-second billing |
| **SageMaker Registry** | Model versioning & approval | Free |
| **SageMaker Endpoint** | Real-time inference | Per-hour (instance running) |
| **CloudWatch** | Logs & monitoring | Minimal |
| **IAM** | Access control | Free |

---

## Cost Estimate (ap-south-1)

| Component | Usage | Monthly Cost |
|-----------|-------|--------------|
| S3 (data + models) | ~100 MB | ~$0.01 |
| Processing Jobs (preprocess + eval) | 2 × 5 min × ml.m5.large | ~$0.06 per run |
| Training Job | 1 × 10 min × ml.m5.xlarge | ~$0.08 per run |
| Endpoint (always-on) | ml.m5.large × 24/7 | ~$140/month |
| Endpoint (auto-scale to 0) | Serverless inference | Pay per request |

**Tip:** Use Serverless Inference for dev/testing to avoid always-on costs.

---

## Security Architecture

```
┌─────────────────────────────────────────────┐
│  IAM Role: SageMakerExecutionRole           │
│                                             │
│  Permissions:                               │
│  - sagemaker:*                              │
│  - s3:GetObject, PutObject (bucket only)    │
│  - ecr:GetDownloadUrlForLayer               │
│  - logs:CreateLogGroup, PutLogEvents        │
│  - kms:Decrypt (if using encrypted S3)      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  S3 Bucket Policy                           │
│                                             │
│  - Only SageMaker role can read/write       │
│  - Versioning enabled                       │
│  - Server-side encryption (SSE-S3)          │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Endpoint Security                          │
│                                             │
│  - VPC deployment (optional)                │
│  - API Gateway + API Key for public access  │
│  - CloudWatch alarms for anomalies          │
└─────────────────────────────────────────────┘
```

---

## Comparison: EC2 vs SageMaker

| Aspect | EC2 (Current) | SageMaker (New) |
|--------|---------------|-----------------|
| **Infra management** | Manual (systemd, venv) | Fully managed |
| **Scaling** | Manual (add instances) | Auto-scaling |
| **Model versioning** | MLflow | SageMaker Model Registry |
| **Feature store** | Feast + Redis | SageMaker Feature Store (optional) |
| **Training** | On EC2 directly | Managed Training Jobs |
| **Endpoint** | FastAPI + Uvicorn | SageMaker Endpoint |
| **Cost (dev)** | ~$30/month (t3.medium) | Pay per use (~$5-10) |
| **Cost (prod)** | ~$100/month (m5.large) | ~$140/month (always-on) |
| **Drift monitoring** | Evidently (self-managed) | SageMaker Model Monitor |
| **CI/CD** | Manual or GitHub Actions | SageMaker Pipelines |

---

*Architecture document for ML Carbon Emissions — SageMaker MLOps deployment.*
