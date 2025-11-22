"""
Unit tests for data preparation modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestDataValidation:
    """Test suite for data validation functionality."""
    
    def test_validate_schema_valid_data(self, sample_interactions):
        """Test schema validation with valid data."""
        required_columns = ['user_id', 'product_id', 'interaction_type', 'timestamp']
        assert all(col in sample_interactions.columns for col in required_columns)
    
    def test_validate_schema_missing_columns(self):
        """Test schema validation with missing columns."""
        df = pd.DataFrame({'user_id': ['user_1'], 'product_id': ['prod_1']})
        required_columns = ['user_id', 'product_id', 'timestamp']
        missing = set(required_columns) - set(df.columns)
        assert 'timestamp' in missing
    
    def test_validate_data_types(self, sample_interactions):
        """Test data type validation."""
        assert sample_interactions['user_id'].dtype == object
        assert sample_interactions['product_id'].dtype == object
        assert pd.api.types.is_datetime64_any_dtype(sample_interactions['timestamp'])
    
    def test_validate_no_duplicates(self, sample_interactions):
        """Test duplicate detection."""
        duplicates = sample_interactions.duplicated(subset=['interaction_id'])
        assert duplicates.sum() == 0
    
    def test_validate_null_values(self, sample_interactions):
        """Test null value detection in required fields."""
        required_fields = ['user_id', 'product_id', 'interaction_type']
        for field in required_fields:
            assert sample_interactions[field].isnull().sum() == 0
    
    def test_validate_value_ranges(self, sample_products):
        """Test value range validation."""
        assert (sample_products['price'] > 0).all()
        assert (sample_products['rating'] >= 1).all() and (sample_products['rating'] <= 5).all()
        assert (sample_products['num_reviews'] >= 0).all()


class TestDataCleaning:
    """Test suite for data cleaning functionality."""
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        df = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_2'],
            'product_id': ['prod_1', 'prod_1', 'prod_2'],
            'timestamp': [datetime.now()] * 3
        })
        df_clean = df.drop_duplicates(subset=['user_id', 'product_id'])
        assert len(df_clean) == 2
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = pd.DataFrame({
            'user_id': ['user_1', 'user_2', None],
            'rating': [5.0, None, 4.0]
        })
        # Drop rows with missing user_id
        df_clean = df.dropna(subset=['user_id'])
        assert len(df_clean) == 2
        
        # Fill missing ratings with mean
        df['rating'] = df['rating'].fillna(df['rating'].mean())
        assert df['rating'].isnull().sum() == 0
    
    def test_normalize_text_fields(self):
        """Test text normalization."""
        df = pd.DataFrame({
            'category': ['  Electronics  ', 'CLOTHING', 'home']
        })
        df['category'] = df['category'].str.strip().str.title()
        assert df['category'].tolist() == ['Electronics', 'Clothing', 'Home']
    
    def test_filter_inactive_products(self, sample_products):
        """Test filtering inactive products."""
        sample_products['is_active'] = np.random.choice([True, False], len(sample_products))
        active_products = sample_products[sample_products['is_active']]
        assert all(active_products['is_active'])


class TestFeatureEngineering:
    """Test suite for feature engineering functionality."""
    
    def test_user_aggregation_features(self, sample_interactions):
        """Test user-level aggregation features."""
        user_features = sample_interactions.groupby('user_id').agg({
            'interaction_id': 'count',
            'timestamp': 'max'
        }).reset_index()
        user_features.columns = ['user_id', 'num_interactions', 'last_interaction']
        
        assert 'num_interactions' in user_features.columns
        assert 'last_interaction' in user_features.columns
        assert (user_features['num_interactions'] > 0).all()
    
    def test_item_aggregation_features(self, sample_interactions):
        """Test item-level aggregation features."""
        item_features = sample_interactions.groupby('product_id').agg({
            'interaction_id': 'count',
            'user_id': 'nunique'
        }).reset_index()
        item_features.columns = ['product_id', 'num_interactions', 'num_unique_users']
        
        assert 'num_interactions' in item_features.columns
        assert 'num_unique_users' in item_features.columns
    
    def test_recency_features(self, sample_interactions):
        """Test recency feature calculation."""
        now = datetime.now()
        sample_interactions['recency_days'] = (
            now - sample_interactions['timestamp']
        ).dt.total_seconds() / 86400
        
        assert 'recency_days' in sample_interactions.columns
        assert (sample_interactions['recency_days'] >= 0).all()
    
    def test_frequency_features(self, sample_interactions):
        """Test frequency feature calculation."""
        user_freq = sample_interactions.groupby('user_id').size().reset_index(name='frequency')
        assert 'frequency' in user_freq.columns
        assert (user_freq['frequency'] > 0).all()
    
    def test_category_encoding(self, sample_products):
        """Test categorical feature encoding."""
        # One-hot encoding
        category_dummies = pd.get_dummies(sample_products['category'], prefix='category')
        assert category_dummies.shape[1] > 0
        assert category_dummies.sum(axis=1).eq(1).all()  # Each row has exactly one 1
    
    def test_price_percentile(self, sample_products):
        """Test price percentile calculation."""
        sample_products['price_percentile'] = sample_products.groupby('category')['price'].rank(pct=True)
        assert 'price_percentile' in sample_products.columns
        assert (sample_products['price_percentile'] >= 0).all()
        assert (sample_products['price_percentile'] <= 1).all()


class TestDatasetSplitter:
    """Test suite for dataset splitting functionality."""
    
    def test_temporal_split(self, sample_interactions):
        """Test temporal train/val/test split."""
        sample_interactions = sample_interactions.sort_values('timestamp')
        n = len(sample_interactions)
        
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        
        train_data = sample_interactions[:train_end]
        val_data = sample_interactions[train_end:val_end]
        test_data = sample_interactions[val_end:]
        
        assert len(train_data) + len(val_data) + len(test_data) == n
        assert train_data['timestamp'].max() <= val_data['timestamp'].min()
        assert val_data['timestamp'].max() <= test_data['timestamp'].min()
    
    def test_split_ratios(self, sample_interactions):
        """Test split ratio correctness."""
        n = len(sample_interactions)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        
        train_size = train_end
        val_size = val_end - train_end
        test_size = n - val_end
        
        assert abs(train_size / n - 0.7) < 0.01
        assert abs(val_size / n - 0.1) < 0.01
        assert abs(test_size / n - 0.2) < 0.01
    
    def test_no_data_leakage(self, sample_interactions):
        """Test that there's no data leakage between splits."""
        sample_interactions = sample_interactions.sort_values('timestamp')
        n = len(sample_interactions)
        
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        
        train_data = sample_interactions[:train_end]
        test_data = sample_interactions[val_end:]
        
        # Check no overlapping interaction IDs
        train_ids = set(train_data['interaction_id'])
        test_ids = set(test_data['interaction_id'])
        assert len(train_ids & test_ids) == 0


class TestDataLoader:
    """Test suite for data loading functionality."""
    
    def test_load_csv_data(self, temp_data_dir, sample_users):
        """Test loading CSV data."""
        file_path = os.path.join(temp_data_dir, 'users.csv')
        sample_users.to_csv(file_path, index=False)
        
        loaded_data = pd.read_csv(file_path)
        assert len(loaded_data) == len(sample_users)
        assert list(loaded_data.columns) == list(sample_users.columns)
    
    def test_load_parquet_data(self, temp_data_dir, sample_products):
        """Test loading Parquet data."""
        file_path = os.path.join(temp_data_dir, 'products.parquet')
        sample_products.to_parquet(file_path, index=False)
        
        loaded_data = pd.read_parquet(file_path)
        assert len(loaded_data) == len(sample_products)
    
    def test_handle_missing_file(self, temp_data_dir):
        """Test handling of missing file."""
        file_path = os.path.join(temp_data_dir, 'nonexistent.csv')
        with pytest.raises(FileNotFoundError):
            pd.read_csv(file_path)
    
    def test_parse_dates(self, temp_data_dir, sample_interactions):
        """Test date parsing during load."""
        file_path = os.path.join(temp_data_dir, 'interactions.csv')
        sample_interactions.to_csv(file_path, index=False)
        
        loaded_data = pd.read_csv(file_path, parse_dates=['timestamp'])
        assert pd.api.types.is_datetime64_any_dtype(loaded_data['timestamp'])
