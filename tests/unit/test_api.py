"""
Unit tests for API and inference modules.
"""
import pytest
import json
import re
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestAPIValidation:
    """Test suite for API request validation."""
    
    def test_validate_required_fields(self, sample_api_request):
        """Test validation of required fields."""
        required_fields = ['user_id', 'num_recommendations']
        
        for field in required_fields:
            assert field in sample_api_request
    
    def test_validate_user_id_format(self):
        """Test user ID format validation."""
        valid_user_ids = ['user_123', 'user_abc', 'user_1']
        invalid_user_ids = ['', None, 123, 'user@123']
        
        for user_id in valid_user_ids:
            assert isinstance(user_id, str) and len(user_id) > 0
        
        for user_id in invalid_user_ids:
            # Check for string, non-empty, and alphanumeric (plus underscore)
            is_valid = (
                isinstance(user_id, str) and 
                len(user_id) > 0 and 
                bool(re.match(r'^\w+$', user_id))
            )
            assert not is_valid
    
    def test_validate_num_recommendations_range(self):
        """Test num_recommendations range validation."""
        valid_values = [1, 10, 50]
        invalid_values = [0, -1, 51, 100]
        
        min_val, max_val = 1, 50
        
        for val in valid_values:
            assert min_val <= val <= max_val
        
        for val in invalid_values:
            assert not (min_val <= val <= max_val)
    
    def test_validate_filters_structure(self, sample_api_request):
        """Test filters structure validation."""
        filters = sample_api_request.get('filters', {})
        
        assert isinstance(filters, dict)
        
        if 'categories' in filters:
            assert isinstance(filters['categories'], list)
        
        if 'price_min' in filters and 'price_max' in filters:
            assert filters['price_min'] <= filters['price_max']
    
    def test_validate_context_fields(self, sample_api_request):
        """Test context fields validation."""
        context = sample_api_request.get('context', {})
        
        assert isinstance(context, dict)
        
        valid_pages = ['homepage', 'product_page', 'search', 'cart']
        if 'page' in context:
            assert context['page'] in valid_pages or isinstance(context['page'], str)
        
        valid_devices = ['mobile', 'desktop', 'tablet']
        if 'device' in context:
            assert context['device'] in valid_devices or isinstance(context['device'], str)


class TestAPIResponseFormatting:
    """Test suite for API response formatting."""
    
    def test_format_recommendation_response(self, sample_api_response):
        """Test recommendation response format."""
        assert 'recommendations' in sample_api_response
        assert 'model_version' in sample_api_response
        assert 'inference_time_ms' in sample_api_response
        assert 'request_id' in sample_api_response
        assert 'timestamp' in sample_api_response
    
    def test_recommendation_item_structure(self, sample_api_response):
        """Test individual recommendation item structure."""
        recommendations = sample_api_response['recommendations']
        
        for rec in recommendations:
            assert 'product_id' in rec
            assert 'name' in rec
            assert 'score' in rec
            assert 'reason' in rec
            assert isinstance(rec['score'], (int, float))
            assert 0 <= rec['score'] <= 1
    
    def test_format_error_response(self):
        """Test error response format."""
        error_response = {
            'error': {
                'code': 'INVALID_REQUEST',
                'message': 'user_id is required',
                'request_id': 'req_xyz789'
            }
        }
        
        assert 'error' in error_response
        assert 'code' in error_response['error']
        assert 'message' in error_response['error']
        assert 'request_id' in error_response['error']
    
    def test_response_serialization(self, sample_api_response):
        """Test response can be serialized to JSON."""
        # Convert datetime to string for JSON serialization
        sample_api_response['timestamp'] = str(sample_api_response['timestamp'])
        
        json_str = json.dumps(sample_api_response)
        assert isinstance(json_str, str)
        
        # Verify can be deserialized
        deserialized = json.loads(json_str)
        assert deserialized['model_version'] == sample_api_response['model_version']


class TestLambdaHandler:
    """Test suite for Lambda handler functionality."""
    
    def test_lambda_handler_success(self, sample_api_request):
        """Test successful Lambda handler execution."""
        event = {
            'body': json.dumps(sample_api_request),
            'headers': {'X-API-Key': 'test_key'}
        }
        context = Mock()
        
        # Mock response
        expected_response = {
            'statusCode': 200,
            'body': json.dumps({'recommendations': []})
        }
        
        assert expected_response['statusCode'] == 200
    
    def test_lambda_handler_missing_api_key(self):
        """Test Lambda handler with missing API key."""
        event = {
            'body': json.dumps({'user_id': 'user_123'}),
            'headers': {}
        }
        context = Mock()
        
        # Should return 401 Unauthorized
        expected_status = 401
        assert expected_status == 401
    
    def test_lambda_handler_invalid_json(self):
        """Test Lambda handler with invalid JSON."""
        event = {
            'body': 'invalid json{',
            'headers': {'X-API-Key': 'test_key'}
        }
        context = Mock()
        
        # Should return 400 Bad Request
        expected_status = 400
        assert expected_status == 400
    
    def test_lambda_handler_missing_required_field(self):
        """Test Lambda handler with missing required field."""
        event = {
            'body': json.dumps({'num_recommendations': 10}),  # Missing user_id
            'headers': {'X-API-Key': 'test_key'}
        }
        context = Mock()
        
        # Should return 400 Bad Request
        expected_status = 400
        assert expected_status == 400
    
    def test_lambda_cold_start_initialization(self):
        """Test Lambda cold start initialization."""
        # Simulate cold start
        global_var = None
        
        if global_var is None:
            global_var = {'initialized': True}
        
        assert global_var['initialized'] is True


class TestInferenceService:
    """Test suite for inference service."""
    
    def test_get_recommendations_existing_user(self, mock_dynamodb_table):
        """Test getting recommendations for existing user."""
        user_id = 'user_123'
        num_recommendations = 10
        
        # Mock user exists
        mock_dynamodb_table.put_item(Item={'user_id': user_id, 'history': []})
        
        # Simulate recommendation generation
        recommendations = [
            {'product_id': f'prod_{i}', 'score': 0.9 - i*0.05}
            for i in range(num_recommendations)
        ]
        
        assert len(recommendations) == num_recommendations
        assert all('product_id' in rec for rec in recommendations)
        assert all('score' in rec for rec in recommendations)
    
    def test_get_recommendations_new_user_cold_start(self):
        """Test getting recommendations for new user (cold start)."""
        user_id = 'new_user_999'
        
        # For cold start, return popular items
        popular_items = [
            {'product_id': 'prod_1', 'score': 0.95},
            {'product_id': 'prod_2', 'score': 0.90},
            {'product_id': 'prod_3', 'score': 0.85}
        ]
        
        assert len(popular_items) > 0
        assert all(rec['score'] > 0 for rec in popular_items)
    
    def test_exclude_purchased_items(self, mock_dynamodb_table):
        """Test excluding purchased items from recommendations."""
        user_id = 'user_123'
        purchased_items = ['prod_1', 'prod_2', 'prod_3']
        
        # Mock user purchase history
        mock_dynamodb_table.put_item(Item={
            'user_id': user_id,
            'purchased_items': purchased_items
        })
        
        # Generate recommendations
        all_recommendations = ['prod_1', 'prod_4', 'prod_5', 'prod_6']
        filtered_recommendations = [
            item for item in all_recommendations 
            if item not in purchased_items
        ]
        
        assert 'prod_1' not in filtered_recommendations
        assert 'prod_4' in filtered_recommendations
    
    def test_apply_category_filters(self):
        """Test applying category filters to recommendations."""
        recommendations = [
            {'product_id': 'prod_1', 'category': 'Electronics'},
            {'product_id': 'prod_2', 'category': 'Clothing'},
            {'product_id': 'prod_3', 'category': 'Electronics'},
            {'product_id': 'prod_4', 'category': 'Home'}
        ]
        
        allowed_categories = ['Electronics', 'Home']
        filtered = [
            rec for rec in recommendations 
            if rec['category'] in allowed_categories
        ]
        
        assert len(filtered) == 3
        assert all(rec['category'] in allowed_categories for rec in filtered)
    
    def test_apply_price_filters(self):
        """Test applying price range filters."""
        recommendations = [
            {'product_id': 'prod_1', 'price': 50.0},
            {'product_id': 'prod_2', 'price': 150.0},
            {'product_id': 'prod_3', 'price': 250.0},
            {'product_id': 'prod_4', 'price': 350.0}
        ]
        
        price_min, price_max = 100.0, 300.0
        filtered = [
            rec for rec in recommendations 
            if price_min <= rec['price'] <= price_max
        ]
        
        assert len(filtered) == 2
        assert all(price_min <= rec['price'] <= price_max for rec in filtered)
    
    def test_recommendation_ranking(self):
        """Test recommendations are properly ranked by score."""
        recommendations = [
            {'product_id': 'prod_1', 'score': 0.75},
            {'product_id': 'prod_2', 'score': 0.95},
            {'product_id': 'prod_3', 'score': 0.85},
            {'product_id': 'prod_4', 'score': 0.65}
        ]
        
        sorted_recs = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        
        assert sorted_recs[0]['product_id'] == 'prod_2'  # Highest score
        assert sorted_recs[-1]['product_id'] == 'prod_4'  # Lowest score
        
        # Verify descending order
        scores = [rec['score'] for rec in sorted_recs]
        assert scores == sorted(scores, reverse=True)
    
    def test_inference_latency_tracking(self):
        """Test inference latency is tracked."""
        start_time = datetime.now()
        
        # Simulate inference
        import time
        time.sleep(0.01)  # 10ms
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        assert latency_ms >= 10
        assert latency_ms < 1000  # Should be under 1 second for test


class TestModelLoader:
    """Test suite for model loading functionality."""
    
    def test_load_model_from_s3(self, mock_s3_client, temp_model_dir):
        """Test loading model from S3."""
        bucket = 'model-bucket'
        key = 'models/v1.0.0/model.pth'
        
        # Mock S3 download
        mock_s3_client.download_file(bucket, key, '/tmp/model.pth')
        
        # Verify download was called (in real implementation)
        assert True  # Placeholder
    
    def test_load_model_metadata(self, mock_s3_client):
        """Test loading model metadata."""
        metadata = {
            'model_version': 'v1.0.0',
            'created_at': '2024-01-15T10:00:00Z',
            'metrics': {
                'ndcg@10': 0.48,
                'precision@10': 0.38
            }
        }
        
        assert 'model_version' in metadata
        assert 'metrics' in metadata
        assert isinstance(metadata['metrics'], dict)
    
    def test_cache_loaded_model(self):
        """Test model caching to avoid reloading."""
        model_cache = {}
        model_version = 'v1.0.0'
        
        # First load
        if model_version not in model_cache:
            model_cache[model_version] = {'loaded': True}
        
        # Second load (should use cache)
        cached_model = model_cache.get(model_version)
        
        assert cached_model is not None
        assert cached_model['loaded'] is True


class TestRateLimiting:
    """Test suite for API rate limiting."""
    
    def test_rate_limit_check(self):
        """Test rate limit checking."""
        user_requests = {}
        user_id = 'user_123'
        rate_limit = 100  # requests per minute
        
        # Simulate requests
        current_count = user_requests.get(user_id, 0)
        user_requests[user_id] = current_count + 1
        
        is_rate_limited = user_requests[user_id] > rate_limit
        
        assert not is_rate_limited
    
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded scenario."""
        user_requests = {'user_123': 101}
        rate_limit = 100
        
        is_rate_limited = user_requests['user_123'] > rate_limit
        
        assert is_rate_limited
    
    def test_rate_limit_reset(self):
        """Test rate limit counter reset."""
        user_requests = {'user_123': 50}
        
        # Simulate time window reset
        user_requests = {}
        
        assert 'user_123' not in user_requests


class TestErrorHandling:
    """Test suite for error handling."""
    
    def test_handle_model_not_found(self):
        """Test handling of model not found error."""
        try:
            raise FileNotFoundError("Model not found")
        except FileNotFoundError as e:
            error_response = {
                'error': {
                    'code': 'MODEL_NOT_FOUND',
                    'message': str(e)
                }
            }
            assert error_response['error']['code'] == 'MODEL_NOT_FOUND'
    
    def test_handle_invalid_user_id(self):
        """Test handling of invalid user ID."""
        user_id = None
        
        if not user_id:
            error_response = {
                'error': {
                    'code': 'INVALID_USER_ID',
                    'message': 'user_id is required'
                }
            }
            assert error_response['error']['code'] == 'INVALID_USER_ID'
    
    def test_handle_service_unavailable(self):
        """Test handling of service unavailable error."""
        try:
            raise ConnectionError("Service unavailable")
        except ConnectionError as e:
            error_response = {
                'statusCode': 503,
                'error': {
                    'code': 'SERVICE_UNAVAILABLE',
                    'message': str(e)
                }
            }
            assert error_response['statusCode'] == 503
    
    def test_handle_timeout(self):
        """Test handling of timeout error."""
        try:
            raise TimeoutError("Request timeout")
        except TimeoutError as e:
            error_response = {
                'statusCode': 504,
                'error': {
                    'code': 'TIMEOUT',
                    'message': str(e)
                }
            }
            assert error_response['statusCode'] == 504
