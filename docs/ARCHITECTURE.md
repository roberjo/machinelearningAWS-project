# System Architecture

This document details the architecture of the ML-powered product recommendation system. The system is built on AWS using serverless services to ensure scalability, cost-efficiency, and ease of maintenance.

## High-Level Architecture

The system follows a standard MLOps lifecycle architecture, separated into three main planes:
1. **Data & Training Plane**: Handles data ingestion, processing, and model training.
2. **Inference Plane**: Serves real-time recommendations to users.
3. **Monitoring & Operations Plane**: Tracks performance, detects drift, and manages automation.

```mermaid
graph TD
    subgraph "Data & Training Plane"
        S3_Data[(S3 Data Lake)] -->|Raw Data| Glue[AWS Glue/Lambda]
        Glue -->|Processed Data| S3_Features[(S3 Features)]
        S3_Features --> SM_Train[SageMaker Training Jobs]
        SM_Train -->|Model Artifacts| S3_Models[(S3 Model Registry)]
        SM_Train -->|Metrics| DDB_Exp[DynamoDB Experiments]
    end

    subgraph "Inference Plane"
        Client[Client App] -->|POST /recommend| APIG[API Gateway]
        APIG --> Lambda_Inf[Inference Lambda]
        Lambda_Inf -->|Load Model| S3_Models
        Lambda_Inf -->|Get User History| DDB_User[DynamoDB User History]
        Lambda_Inf -->|Invoke (Optional)| SM_Endpoint[SageMaker Serverless Endpoint]
    end

    subgraph "Monitoring & Ops Plane"
        Lambda_Inf -->|Logs| CW[CloudWatch Logs]
        CW -->|Metrics| CW_Metrics[CloudWatch Metrics]
        CW_Metrics -->|Alarms| SNS[SNS Alerts]
        StepF[Step Functions] -->|Orchestrate| SM_Train
        GitHub[GitHub Actions] -->|CI/CD| StepF
    end
```

## Component Details

### 1. Data Storage & Management
*   **S3 Buckets**:
    *   `data-lake`: Stores raw user interaction logs, product catalogs, and user metadata.
    *   `processed-data`: Stores cleaned and feature-engineered datasets ready for training.
    *   `model-artifacts`: Stores trained model binaries (`.pth`, `.pkl`) and metadata.
*   **DynamoDB**:
    *   `UserInteractions`: Low-latency access to user purchase history for filtering recommendations.
    *   `ModelRegistry`: Tracks model versions, hyperparameters, and evaluation metrics.
    *   `InferenceLogs`: Stores inference requests and responses for monitoring.

### 2. Model Training Pipeline
The training pipeline is orchestrated by **AWS Step Functions** and runs on **Amazon SageMaker**.
*   **Preprocessing**: Cleans data and generates user/item embeddings.
*   **Training**: Uses PyTorch/Scikit-learn to train the Collaborative Filtering/Hybrid model.
*   **Evaluation**: Calculates offline metrics (RMSE, Precision@K) and compares against the current champion model.
*   **Registration**: If the new model performs better, it is registered in the Model Registry.

### 3. Inference Service
The inference service is designed for low latency (<100ms).
*   **API Gateway**: Provides a secure HTTP endpoint for the client.
*   **Lambda Function**:
    *   Validates requests.
    *   Fetches user context.
    *   Loads lightweight models directly or invokes SageMaker Serverless Inference for complex models.
    *   Post-processes scores (ranking, filtering purchased items).
    *   Returns JSON response.

### 4. Infrastructure as Code (IaC)
All infrastructure is defined in **Terraform** modules located in `terraform/modules/`:
*   `sagemaker`: Training jobs, endpoints, and registry.
*   `lambda-inference`: API handlers and layers.
*   `api-gateway`: HTTP API configuration.
*   `ml-pipeline`: Step Functions state machines.
*   `monitoring`: CloudWatch dashboards and alarms.

## Security & Compliance
*   **IAM Roles**: Least-privilege access policies for all services.
*   **Encryption**: S3 buckets enabled with SSE-S3; DynamoDB with KMS.
*   **VPC**: Lambda functions run within a VPC for secure access to data stores (optional based on requirements).
