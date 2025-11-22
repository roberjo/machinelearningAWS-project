# Getting Started Guide

## Welcome!

This guide will help you get started with the ML-Powered Product Recommendation System. Whether you're a data scientist, ML engineer, or developer, this guide will walk you through setting up your environment and running your first recommendation model.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Understanding the System](#understanding-the-system)
4. [Local Development](#local-development)
5. [AWS Deployment](#aws-deployment)
6. [Next Steps](#next-steps)

## Prerequisites

### Required Tools

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **AWS CLI**: [Install AWS CLI](https://aws.amazon.com/cli/)
- **Terraform 1.5+**: [Install Terraform](https://www.terraform.io/downloads)
- **Docker**: [Install Docker](https://www.docker.com/get-started) (optional, for local testing)

### AWS Account Setup

1. **Create an AWS Account**: If you don't have one, [sign up here](https://aws.amazon.com/free/)

2. **Create an IAM User** with the following permissions:
   - `AmazonS3FullAccess`
   - `AmazonDynamoDBFullAccess`
   - `AWSLambdaFullAccess`
   - `AmazonSageMakerFullAccess`
   - `AmazonAPIGatewayAdministrator`
   - `IAMFullAccess` (for creating roles)

3. **Generate Access Keys**:
   - Go to IAM → Users → Your User → Security Credentials
   - Click "Create access key"
   - Save the Access Key ID and Secret Access Key

### Knowledge Prerequisites

**Recommended Knowledge**:
- Basic Python programming
- Understanding of machine learning concepts
- Familiarity with AWS services (helpful but not required)
- Basic command-line usage

**Nice to Have**:
- Experience with PyTorch or TensorFlow
- Knowledge of REST APIs
- Terraform basics

## Environment Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/machinelearningAWS-project.git

# Navigate to the project directory
cd machinelearningAWS-project
```

### Step 2: Set Up Python Virtual Environment

**On macOS/Linux**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**On Windows**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Verify installation
python -c "import torch; import sklearn; print('Dependencies installed successfully!')"
```

### Step 4: Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# You'll be prompted for:
# AWS Access Key ID: [Enter your access key]
# AWS Secret Access Key: [Enter your secret key]
# Default region name: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the file with your settings
# On macOS/Linux:
nano .env

# On Windows:
notepad .env
```

Example `.env` file:
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# Environment
ENVIRONMENT=dev

# S3 Buckets
DATA_BUCKET=ml-recommendation-data-dev
MODEL_BUCKET=ml-recommendation-models-dev

# DynamoDB Tables
USER_INTERACTIONS_TABLE=UserInteractions-dev
MODEL_REGISTRY_TABLE=ModelRegistry-dev

# API Configuration
API_KEY=your_dev_api_key_here
```

## Understanding the System

### Architecture Overview

The system consists of three main components:

1. **Data Pipeline**: Processes raw e-commerce data into features
2. **Training Pipeline**: Trains and evaluates recommendation models
3. **Inference API**: Serves real-time recommendations

```
Raw Data → Data Processing → Feature Engineering → Model Training → Deployment → API
```

### Key Concepts

**Collaborative Filtering**: Recommends items based on user-item interaction patterns

**Content-Based Filtering**: Recommends items based on product features

**Hybrid Model**: Combines both approaches for better recommendations

**Cold Start**: Handling new users/products with no interaction history

## Local Development

### Step 1: Generate Sample Data

```bash
# Generate synthetic e-commerce data
python scripts/generate_sample_data.py \
  --num-users 1000 \
  --num-products 500 \
  --num-interactions 10000 \
  --output-dir data/raw

# This creates:
# - data/raw/users.csv
# - data/raw/products.csv
# - data/raw/interactions.csv
```

### Step 2: Process Data

```bash
# Run data preprocessing
python src/data_preparation/data_loader.py \
  --input-dir data/raw \
  --output-dir data/processed

# Run feature engineering
python src/data_preparation/feature_engineering.py \
  --input-dir data/processed \
  --output-dir data/features
```

### Step 3: Train a Model Locally

```bash
# Train the recommendation model
python src/training/train.py \
  --data-path data/features \
  --model-dir models/local \
  --embedding-dim 50 \
  --learning-rate 0.001 \
  --batch-size 256 \
  --num-epochs 10

# This will:
# 1. Load and split the data
# 2. Train the model
# 3. Evaluate on test set
# 4. Save model artifacts to models/local/
```

**Expected Output**:
```
Loading data...
Training model...
Epoch 1/10: Train Loss = 0.4523, Val Loss = 0.3891
Epoch 2/10: Train Loss = 0.3245, Val Loss = 0.3102
...
Evaluating model...
Test Metrics:
  RMSE: 0.82
  MAE: 0.65
  Precision@10: 0.38
  Recall@10: 0.28
  NDCG@10: 0.48

Model saved to models/local/
```

### Step 4: Test Inference Locally

```bash
# Start local inference server
python src/api/lambda_handler.py --local --port 8000

# In another terminal, test the API
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10
  }'
```

### Step 5: Run Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_models.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## AWS Deployment

### Step 1: Initialize Terraform

```bash
# Navigate to the dev environment
cd terraform/environments/dev

# Initialize Terraform
terraform init

# This downloads required providers and modules
```

### Step 2: Review Infrastructure Plan

```bash
# Create an execution plan
terraform plan -out=tfplan

# Review the resources that will be created:
# - S3 buckets for data and models
# - DynamoDB tables for metadata
# - Lambda functions for API
# - SageMaker endpoints for inference
# - API Gateway for public access
# - CloudWatch for monitoring
```

### Step 3: Deploy Infrastructure

```bash
# Apply the Terraform plan
terraform apply tfplan

# Type 'yes' to confirm

# This takes ~10-15 minutes
```

**Expected Output**:
```
Apply complete! Resources: 45 added, 0 changed, 0 destroyed.

Outputs:
api_endpoint = "https://abc123.execute-api.us-east-1.amazonaws.com/dev"
model_bucket = "ml-recommendation-models-dev-abc123"
data_bucket = "ml-recommendation-data-dev-abc123"
```

### Step 4: Upload Data to S3

```bash
# Upload processed data to S3
aws s3 sync data/processed/ s3://ml-recommendation-data-dev-abc123/processed/

# Verify upload
aws s3 ls s3://ml-recommendation-data-dev-abc123/processed/
```

### Step 5: Train Model on SageMaker

```bash
# Trigger SageMaker training job
python scripts/trigger_training.py \
  --environment dev \
  --data-path s3://ml-recommendation-data-dev-abc123/processed/ \
  --output-path s3://ml-recommendation-models-dev-abc123/

# Monitor training job
aws sagemaker describe-training-job \
  --training-job-name recommendation-training-$(date +%Y%m%d)
```

### Step 6: Deploy Model

```bash
# Deploy trained model to SageMaker endpoint
python scripts/deploy_model.py \
  --model-version v1.0.0 \
  --environment dev \
  --deployment-strategy blue-green

# This creates a SageMaker Serverless Inference endpoint
```

### Step 7: Test Deployed API

```bash
# Get API endpoint from Terraform output
API_ENDPOINT=$(terraform output -raw api_endpoint)

# Test the API
curl -X POST ${API_ENDPOINT}/recommend \
  -H "X-API-Key: your_dev_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10,
    "exclude_purchased": true
  }'
```

**Expected Response**:
```json
{
  "recommendations": [
    {
      "product_id": "prod_456",
      "name": "Wireless Headphones",
      "score": 0.95,
      "reason": "Based on your recent purchases"
    },
    ...
  ],
  "model_version": "v1.0.0",
  "inference_time_ms": 45
}
```

## Next Steps

### Explore the Documentation

- **[Architecture](ARCHITECTURE.md)**: Understand the system design
- **[Data Pipeline](DATA_PIPELINE.md)**: Learn about data processing
- **[Model Development](MODEL_DEVELOPMENT.md)**: Deep dive into ML models
- **[API Reference](API_REFERENCE.md)**: Complete API documentation

### Experiment with the Model

1. **Try different hyperparameters**:
   ```bash
   python src/training/hyperparameter_tuning.py \
     --max-jobs 20 \
     --objective-metric "validation:ndcg@10"
   ```

2. **Explore the Jupyter notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

3. **Implement custom features**:
   - Edit `src/data_preparation/feature_engineering.py`
   - Add new user or item features

### Set Up Monitoring

1. **Access CloudWatch Dashboard**:
   - Go to AWS Console → CloudWatch → Dashboards
   - Open "ml-recommendation-dashboard-dev"

2. **Set up alerts**:
   ```bash
   # Configure SNS topic for alerts
   aws sns create-topic --name ml-recommendation-alerts
   
   # Subscribe your email
   aws sns subscribe \
     --topic-arn arn:aws:sns:us-east-1:123456789012:ml-recommendation-alerts \
     --protocol email \
     --notification-endpoint your-email@example.com
   ```

### Deploy to Production

Once you're comfortable with the dev environment:

1. **Review the production checklist** in [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Update production configuration** in `terraform/environments/prod/`
3. **Deploy to production**:
   ```bash
   cd terraform/environments/prod
   terraform init
   terraform plan -out=tfplan
   terraform apply tfplan
   ```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'torch'`
- **Solution**: Ensure virtual environment is activated and dependencies are installed
  ```bash
  source venv/bin/activate  # or venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

**Issue**: `AWS credentials not found`
- **Solution**: Configure AWS CLI
  ```bash
  aws configure
  ```

**Issue**: `Terraform: Error creating S3 bucket: BucketAlreadyExists`
- **Solution**: S3 bucket names must be globally unique. Update bucket names in `terraform/environments/dev/terraform.tfvars`

**Issue**: `SageMaker training job failed`
- **Solution**: Check CloudWatch logs
  ```bash
  aws logs tail /aws/sagemaker/TrainingJobs --follow
  ```

### Getting Help

- **Documentation**: Check the [docs/](../docs/) directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/machinelearningAWS-project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/machinelearningAWS-project/discussions)

## Summary

Congratulations! You've successfully:
- ✅ Set up your development environment
- ✅ Generated and processed sample data
- ✅ Trained a recommendation model locally
- ✅ Deployed infrastructure to AWS
- ✅ Deployed and tested the API

You're now ready to build production-grade ML recommendation systems!

---

**Next**: Explore [Model Development](MODEL_DEVELOPMENT.md) to learn about advanced model architectures and training techniques.
