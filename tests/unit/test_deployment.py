"""
Unit tests for deployment and utility modules.
"""
import pytest
import json
import hashlib
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestDeploymentStrategies:
    """Test suite for deployment strategies."""
    
    def test_blue_green_deployment_routing(self):
        """Test blue-green deployment traffic routing."""
        # Initially all traffic to blue
        blue_traffic = 100
        green_traffic = 0
        
        assert blue_traffic == 100
        assert green_traffic == 0
        
        # After validation, switch to green
        blue_traffic = 0
        green_traffic = 100
        
        assert blue_traffic == 0
        assert green_traffic == 100
    
    def test_canary_deployment_gradual_rollout(self):
        """Test canary deployment gradual rollout."""
        rollout_stages = [
            {'champion': 90, 'challenger': 10},
            {'champion': 50, 'challenger': 50},
            {'champion': 0, 'challenger': 100}
        ]
        
        for stage in rollout_stages:
            assert stage['champion'] + stage['challenger'] == 100
        
        # Verify gradual increase
        assert rollout_stages[0]['challenger'] < rollout_stages[1]['challenger']
        assert rollout_stages[1]['challenger'] < rollout_stages[2]['challenger']
    
    def test_traffic_split_routing(self):
        """Test traffic split routing logic."""
        user_id = 'user_12345'
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        
        challenger_percentage = 10
        
        if hash_value < challenger_percentage:
            model_version = 'challenger'
        else:
            model_version = 'champion'
        
        # With 10% split, most users should get champion
        assert model_version in ['champion', 'challenger']
    
    def test_rollback_trigger_conditions(self):
        """Test rollback trigger conditions."""
        metrics = {
            'error_rate': 6.0,  # Above 5% threshold
            'latency_p95': 600,  # Above 500ms threshold
            'availability': 99.0  # Below 99.5% threshold
        }
        
        thresholds = {
            'error_rate': 5.0,
            'latency_p95': 500,
            'availability': 99.5
        }
        
        should_rollback = (
            metrics['error_rate'] > thresholds['error_rate'] or
            metrics['latency_p95'] > thresholds['latency_p95'] or
            metrics['availability'] < thresholds['availability']
        )
        
        assert should_rollback is True


class TestModelVersioning:
    """Test suite for model versioning."""
    
    def test_semantic_versioning(self):
        """Test semantic versioning format."""
        version = 'v2.3.1'
        
        # Parse version
        parts = version.lstrip('v').split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        assert major == 2
        assert minor == 3
        assert patch == 1
    
    def test_version_comparison(self):
        """Test version comparison logic."""
        def parse_version(version_str):
            parts = version_str.lstrip('v').split('.')
            return tuple(int(p) for p in parts)
        
        v1 = parse_version('v2.3.1')
        v2 = parse_version('v2.4.0')
        v3 = parse_version('v1.9.9')
        
        assert v2 > v1
        assert v1 > v3
    
    def test_model_registry_entry(self):
        """Test model registry entry structure."""
        registry_entry = {
            'model_id': 'rec-model-v2.3.1',
            'version': 'v2.3.1',
            'created_at': datetime.now().isoformat(),
            'status': 'candidate',
            's3_path': 's3://models/v2.3.1/',
            'metrics': {
                'ndcg@10': 0.48,
                'precision@10': 0.38
            },
            'hyperparameters': {
                'embedding_dim': 50,
                'learning_rate': 0.001
            }
        }
        
        assert 'model_id' in registry_entry
        assert 'version' in registry_entry
        assert 'metrics' in registry_entry
        assert 'status' in registry_entry


class TestConfigurationManagement:
    """Test suite for configuration management."""
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration."""
        configs = {
            'dev': {
                'api_throttle_rate': 100,
                'lambda_memory': 512,
                'log_level': 'DEBUG'
            },
            'prod': {
                'api_throttle_rate': 10000,
                'lambda_memory': 1024,
                'log_level': 'INFO'
            }
        }
        
        assert configs['dev']['api_throttle_rate'] < configs['prod']['api_throttle_rate']
        assert configs['dev']['log_level'] == 'DEBUG'
        assert configs['prod']['log_level'] == 'INFO'
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = {
            'embedding_dim': 50,
            'learning_rate': 0.001,
            'batch_size': 256
        }
        
        # Validate ranges
        assert 16 <= config['embedding_dim'] <= 256
        assert 0.0001 <= config['learning_rate'] <= 0.1
        assert config['batch_size'] in [64, 128, 256, 512]
    
    def test_config_override(self):
        """Test configuration override mechanism."""
        default_config = {'timeout': 30, 'retries': 3}
        user_config = {'timeout': 60}
        
        final_config = {**default_config, **user_config}
        
        assert final_config['timeout'] == 60  # Overridden
        assert final_config['retries'] == 3  # Default


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_s3_path_parsing(self):
        """Test S3 path parsing."""
        s3_path = 's3://my-bucket/path/to/file.txt'
        
        # Parse bucket and key
        parts = s3_path.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        assert bucket == 'my-bucket'
        assert key == 'path/to/file.txt'
    
    def test_timestamp_formatting(self):
        """Test timestamp formatting."""
        now = datetime.now()
        
        # ISO format
        iso_format = now.isoformat()
        assert 'T' in iso_format
        
        # Custom format
        custom_format = now.strftime('%Y-%m-%d %H:%M:%S')
        assert len(custom_format) == 19
    
    def test_json_serialization(self):
        """Test JSON serialization with datetime."""
        data = {
            'user_id': 'user_123',
            'timestamp': datetime.now(),
            'score': 0.95
        }
        
        # Convert datetime to string
        data['timestamp'] = data['timestamp'].isoformat()
        
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        
        # Deserialize
        loaded_data = json.loads(json_str)
        assert loaded_data['user_id'] == 'user_123'
    
    def test_hash_generation(self):
        """Test hash generation for user IDs."""
        user_id = 'user_12345'
        
        # MD5 hash
        hash_md5 = hashlib.md5(user_id.encode()).hexdigest()
        assert len(hash_md5) == 32
        
        # SHA256 hash
        hash_sha256 = hashlib.sha256(user_id.encode()).hexdigest()
        assert len(hash_sha256) == 64
    
    def test_retry_logic(self):
        """Test retry logic with exponential backoff."""
        max_retries = 3
        base_delay = 1
        
        delays = []
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)
            delays.append(delay)
        
        assert delays == [1, 2, 4]
    
    def test_batch_processing(self):
        """Test batch processing logic."""
        items = list(range(100))
        batch_size = 25
        
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        assert len(batches) == 4
        assert all(len(batch) == batch_size for batch in batches)


class TestDataTransformations:
    """Test suite for data transformation utilities."""
    
    def test_normalize_scores(self):
        """Test score normalization."""
        scores = np.array([10, 20, 30, 40, 50])
        
        # Min-max normalization
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(scores)
    
    def test_standardize_features(self):
        """Test feature standardization (z-score)."""
        features = np.array([10, 20, 30, 40, 50])
        
        # Z-score standardization
        mean = features.mean()
        std = features.std()
        standardized = (features - mean) / std
        
        assert abs(standardized.mean()) < 1e-10  # Mean ~0
        assert abs(standardized.std() - 1.0) < 1e-10  # Std ~1
    
    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        categories = ['A', 'B', 'C', 'A', 'B']
        unique_categories = sorted(set(categories))
        
        # Create one-hot encoding
        one_hot = []
        for cat in categories:
            encoding = [1 if cat == uc else 0 for uc in unique_categories]
            one_hot.append(encoding)
        
        assert len(one_hot) == len(categories)
        assert all(sum(encoding) == 1 for encoding in one_hot)  # Exactly one 1 per row
    
    def test_label_encoding(self):
        """Test label encoding."""
        categories = ['low', 'medium', 'high', 'low', 'high']
        label_map = {'low': 0, 'medium': 1, 'high': 2}
        
        encoded = [label_map[cat] for cat in categories]
        
        assert encoded == [0, 1, 2, 0, 2]


class TestCaching:
    """Test suite for caching mechanisms."""
    
    def test_in_memory_cache(self):
        """Test in-memory cache."""
        cache = {}
        
        # Cache miss
        key = 'user_123'
        if key not in cache:
            cache[key] = {'data': 'computed_value'}
        
        # Cache hit
        cached_value = cache.get(key)
        
        assert cached_value is not None
        assert cached_value['data'] == 'computed_value'
    
    def test_cache_expiration(self):
        """Test cache expiration logic."""
        cache = {
            'key1': {'value': 'data1', 'expires_at': datetime.now() + timedelta(minutes=5)},
            'key2': {'value': 'data2', 'expires_at': datetime.now() - timedelta(minutes=5)}
        }
        
        # Check expiration
        now = datetime.now()
        valid_keys = [k for k, v in cache.items() if v['expires_at'] > now]
        
        assert 'key1' in valid_keys
        assert 'key2' not in valid_keys
    
    def test_cache_size_limit(self):
        """Test cache size limiting (LRU)."""
        max_size = 3
        cache = {}
        access_order = []
        
        # Add items
        for i in range(5):
            key = f'key_{i}'
            cache[key] = f'value_{i}'
            access_order.append(key)
            
            # Evict oldest if over limit
            if len(cache) > max_size:
                oldest_key = access_order.pop(0)
                del cache[oldest_key]
        
        assert len(cache) == max_size
        assert 'key_0' not in cache  # Evicted
        assert 'key_4' in cache  # Most recent


class TestSecurityUtilities:
    """Test suite for security utilities."""
    
    def test_api_key_hashing(self):
        """Test API key hashing."""
        api_key = 'my_secret_api_key'
        
        # Hash the key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        assert len(key_hash) == 64
        assert key_hash != api_key
    
    def test_api_key_validation(self):
        """Test API key validation."""
        stored_hash = hashlib.sha256('correct_key'.encode()).hexdigest()
        
        # Correct key
        provided_key = 'correct_key'
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        assert provided_hash == stored_hash
        
        # Incorrect key
        wrong_key = 'wrong_key'
        wrong_hash = hashlib.sha256(wrong_key.encode()).hexdigest()
        assert wrong_hash != stored_hash
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        user_input = "<script>alert('xss')</script>"
        
        # Simple sanitization (remove HTML tags)
        import re
        sanitized = re.sub(r'<[^>]+>', '', user_input)
        
        assert '<script>' not in sanitized
        assert sanitized == "alert('xss')"
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention (parameterized queries)."""
        # Bad: String concatenation
        user_input = "'; DROP TABLE users; --"
        
        # Good: Parameterized query (simulated)
        query_template = "SELECT * FROM users WHERE user_id = ?"
        parameters = (user_input,)
        
        # In real implementation, parameters would be escaped
        assert query_template.count('?') == len(parameters)


import numpy as np

# Add numpy import at the top of the file if not already there
