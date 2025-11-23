"""
Shared test fixtures for unit and integration tests.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any


@pytest.fixture
def sample_users() -> pd.DataFrame:
    """Generate sample user data for testing."""
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(100)],
        'age': np.random.randint(18, 70, 100),
        'gender': np.random.choice(['M', 'F', 'Other'], 100),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 100),
        'join_date': [
            datetime.now() - timedelta(days=np.random.randint(1, 365))
            for _ in range(100)
        ]
    })


@pytest.fixture
def sample_products() -> pd.DataFrame:
    """Generate sample product data for testing."""
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    return pd.DataFrame({
        'product_id': [f'prod_{i}' for i in range(50)],
        'name': [f'Product {i}' for i in range(50)],
        'category': np.random.choice(categories, 50),
        'price': np.random.uniform(10, 500, 50).round(2),
        'brand': [f'Brand_{np.random.randint(1, 10)}' for _ in range(50)],
        'rating': np.random.uniform(3, 5, 50).round(1),
        'num_reviews': np.random.randint(0, 1000, 50)
    })


@pytest.fixture
def sample_interactions(sample_users, sample_products) -> pd.DataFrame:
    """Generate sample user-product interactions for testing."""
    n_interactions = 500
    user_ids = np.random.choice(sample_users['user_id'], n_interactions)
    product_ids = np.random.choice(sample_products['product_id'], n_interactions)
    
    return pd.DataFrame({
        'interaction_id': [f'int_{i}' for i in range(n_interactions)],
        'user_id': user_ids,
        'product_id': product_ids,
        'interaction_type': np.random.choice(
            ['view', 'add_to_cart', 'purchase'], 
            n_interactions, 
            p=[0.7, 0.2, 0.1]
        ),
        'timestamp': [
            datetime.now() - timedelta(hours=np.random.randint(1, 720))
            for _ in range(n_interactions)
        ],
        'rating': np.random.choice([None, 1, 2, 3, 4, 5], n_interactions, p=[0.7, 0.02, 0.03, 0.05, 0.1, 0.1])
    })


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration for testing."""
    return {
        'embedding_dim': 32,
        'hidden_layers': [64, 32, 16],
        'learning_rate': 0.001,
        'batch_size': 128,
        'num_epochs': 5,
        'dropout_rate': 0.2,
        'patience': 3
    }


@pytest.fixture
def sample_user_item_matrix() -> torch.Tensor:
    """Generate sample user-item interaction matrix."""
    num_users = 100
    num_items = 50
    # Sparse matrix with ~10% density
    matrix = torch.zeros(num_users, num_items)
    num_interactions = int(num_users * num_items * 0.1)
    
    for _ in range(num_interactions):
        user_idx = np.random.randint(0, num_users)
        item_idx = np.random.randint(0, num_items)
        matrix[user_idx, item_idx] = np.random.uniform(1, 5)
    
    return matrix


@pytest.fixture
def sample_api_request() -> Dict[str, Any]:
    """Sample API request payload for testing."""
    return {
        'user_id': 'user_123',
        'num_recommendations': 10,
        'exclude_purchased': True,
        'context': {
            'page': 'homepage',
            'device': 'mobile',
            'session_id': 'sess_abc123'
        },
        'filters': {
            'categories': ['Electronics', 'Home'],
            'price_min': 20.0,
            'price_max': 500.0
        }
    }


@pytest.fixture
def sample_api_response() -> Dict[str, Any]:
    """Sample API response for testing."""
    return {
        'recommendations': [
            {
                'product_id': 'prod_456',
                'name': 'Wireless Headphones',
                'score': 0.95,
                'reason': 'Based on your recent purchases',
                'category': 'Electronics',
                'price': 99.99
            },
            {
                'product_id': 'prod_789',
                'name': 'Smart Speaker',
                'score': 0.89,
                'reason': 'Customers who bought items you liked also bought this',
                'category': 'Electronics',
                'price': 79.99
            }
        ],
        'model_version': 'v1.0.0',
        'inference_time_ms': 45,
        'request_id': 'req_xyz789',
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_training_data() -> Dict[str, torch.Tensor]:
    """Sample training data tensors for testing."""
    batch_size = 32
    return {
        'user_ids': torch.randint(0, 100, (batch_size,)),
        'item_ids': torch.randint(0, 50, (batch_size,)),
        'ratings': torch.rand(batch_size) * 4 + 1  # Ratings between 1-5
    }


@pytest.fixture
def sample_model_metrics() -> Dict[str, float]:
    """Sample model evaluation metrics for testing."""
    return {
        'rmse': 0.82,
        'mae': 0.65,
        'precision@5': 0.42,
        'precision@10': 0.38,
        'recall@5': 0.15,
        'recall@10': 0.28,
        'ndcg@5': 0.51,
        'ndcg@10': 0.48,
        'map': 0.35,
        'coverage': 0.73,
        'diversity': 0.68
    }


@pytest.fixture
def mock_s3_client(monkeypatch):
    """Mock boto3 S3 client for testing."""
    class MockS3Client:
        def __init__(self):
            self.objects = {}
        
        def put_object(self, Bucket, Key, Body):
            self.objects[f"{Bucket}/{Key}"] = Body
            return {'ETag': 'mock-etag'}
        
        def get_object(self, Bucket, Key):
            if f"{Bucket}/{Key}" in self.objects:
                return {'Body': self.objects[f"{Bucket}/{Key}"]}
            raise Exception('NoSuchKey')
        
        def download_file(self, Bucket, Key, Filename):
            # Mock download
            pass
        
        def upload_file(self, Filename, Bucket, Key):
            # Mock upload
            pass
    
    return MockS3Client()


@pytest.fixture
def mock_dynamodb_table(monkeypatch):
    """Mock boto3 DynamoDB table for testing."""
    class MockDynamoDBTable:
        def __init__(self):
            self.items = {}
        
        def put_item(self, Item):
            key = Item.get('user_id') or Item.get('model_id')
            self.items[key] = Item
            return {}
        
        def get_item(self, Key):
            key_value = list(Key.values())[0]
            if key_value in self.items:
                return {'Item': self.items[key_value]}
            return {}
        
        def query(self, **kwargs):
            # Simple mock query
            return {'Items': list(self.items.values())[:10]}
        
        def scan(self, **kwargs):
            limit = kwargs.get('Limit', 10)
            return {'Items': list(self.items.values())[:limit]}
    
    return MockDynamoDBTable()


@pytest.fixture
def mock_sagemaker_client(monkeypatch):
    """Mock boto3 SageMaker client for testing."""
    class MockSageMakerClient:
        def create_training_job(self, **kwargs):
            return {'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:123456789012:training-job/test-job'}
        
        def describe_training_job(self, TrainingJobName):
            return {
                'TrainingJobStatus': 'Completed',
                'ModelArtifacts': {
                    'S3ModelArtifacts': 's3://bucket/model.tar.gz'
                }
            }
        
        def create_model(self, **kwargs):
            return {'ModelArn': 'arn:aws:sagemaker:us-east-1:123456789012:model/test-model'}
        
        def create_endpoint_config(self, **kwargs):
            return {'EndpointConfigArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/test-config'}
        
        def create_endpoint(self, **kwargs):
            return {'EndpointArn': 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint'}
        
        def invoke_endpoint(self, EndpointName, Body, ContentType):
            return {'Body': b'{"score": 0.95}'}
    
    return MockSageMakerClient()


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model artifacts."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return str(model_dir)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return str(data_dir)
