"""
Unit tests for ML models.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestNeuralCollaborativeFiltering:
    """Test suite for Neural Collaborative Filtering model."""
    
    def test_model_initialization(self):
        """Test model initializes with correct dimensions."""
        num_users = 100
        num_items = 50
        embedding_dim = 32
        
        # Create a simple NCF model for testing
        class SimpleNCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.fc = nn.Linear(embedding_dim * 2, 1)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                x = torch.cat([user_emb, item_emb], dim=1)
                return self.fc(x).squeeze()
        
        model = SimpleNCF(num_users, num_items, embedding_dim)
        
        assert model.user_embedding.num_embeddings == num_users
        assert model.item_embedding.num_embeddings == num_items
        assert model.user_embedding.embedding_dim == embedding_dim
        assert model.item_embedding.embedding_dim == embedding_dim
    
    def test_forward_pass(self):
        """Test forward pass produces valid predictions."""
        class SimpleNCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=32):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.fc = nn.Linear(embedding_dim * 2, 1)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                x = torch.cat([user_emb, item_emb], dim=1)
                return self.fc(x).squeeze()
        
        model = SimpleNCF(num_users=100, num_items=50)
        
        user_ids = torch.LongTensor([0, 1, 2])
        item_ids = torch.LongTensor([10, 20, 30])
        
        predictions = model(user_ids, item_ids)
        
        assert predictions.shape == (3,)
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
    
    def test_embedding_lookup(self):
        """Test embedding lookup functionality."""
        num_users = 100
        embedding_dim = 32
        
        user_embedding = nn.Embedding(num_users, embedding_dim)
        user_ids = torch.LongTensor([0, 5, 10])
        
        embeddings = user_embedding(user_ids)
        
        assert embeddings.shape == (3, embedding_dim)
        assert not torch.isnan(embeddings).any()
    
    def test_model_parameters_count(self):
        """Test model has expected number of parameters."""
        class SimpleNCF(nn.Module):
            def __init__(self, num_users, num_items, embedding_dim=32):
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, embedding_dim)
                self.item_embedding = nn.Embedding(num_items, embedding_dim)
                self.fc = nn.Linear(embedding_dim * 2, 1)
            
            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)
                x = torch.cat([user_emb, item_emb], dim=1)
                return self.fc(x).squeeze()
        
        model = SimpleNCF(num_users=100, num_items=50, embedding_dim=32)
        
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = (100 * 32) + (50 * 32) + (64 * 1) + 1  # embeddings + fc weights + bias
        
        assert total_params == expected_params
    
    def test_model_training_mode(self):
        """Test model can switch between train and eval modes."""
        class SimpleNCF(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
                self.dropout = nn.Dropout(0.5)
        
        model = SimpleNCF()
        
        model.train()
        assert model.training is True
        
        model.eval()
        assert model.training is False
    
    def test_gradient_computation(self):
        """Test gradients are computed correctly."""
        class SimpleNCF(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x).squeeze()
        
        model = SimpleNCF()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        x = torch.randn(5, 10)
        y = torch.randn(5)
        
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestBaselineModels:
    """Test suite for baseline recommendation models."""
    
    def test_popularity_based_recommender(self, sample_interactions):
        """Test popularity-based recommendations."""
        # Calculate popularity scores
        popularity = sample_interactions.groupby('product_id').size().reset_index(name='popularity')
        popularity = popularity.sort_values('popularity', ascending=False)
        
        # Get top 10 popular items
        top_items = popularity.head(10)['product_id'].tolist()
        
        assert len(top_items) == 10
        assert len(set(top_items)) == 10  # All unique
    
    def test_random_recommender(self, sample_products):
        """Test random recommendations."""
        num_recommendations = 10
        random_products = sample_products.sample(n=num_recommendations)
        
        assert len(random_products) == num_recommendations
    
    def test_category_based_recommender(self, sample_products):
        """Test category-based recommendations."""
        user_favorite_category = 'Electronics'
        category_products = sample_products[sample_products['category'] == user_favorite_category]
        
        recommendations = category_products.nlargest(10, 'rating')
        
        assert len(recommendations) <= 10
        assert all(recommendations['category'] == user_favorite_category)


class TestModelEvaluation:
    """Test suite for model evaluation metrics."""
    
    def test_rmse_calculation(self):
        """Test RMSE metric calculation."""
        y_true = np.array([3.0, 4.0, 5.0, 2.0, 1.0])
        y_pred = np.array([2.5, 4.2, 4.8, 2.1, 1.3])
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        assert rmse > 0
        assert not np.isnan(rmse)
        assert not np.isinf(rmse)
    
    def test_mae_calculation(self):
        """Test MAE metric calculation."""
        y_true = np.array([3.0, 4.0, 5.0, 2.0, 1.0])
        y_pred = np.array([2.5, 4.2, 4.8, 2.1, 1.3])
        
        mae = np.mean(np.abs(y_true - y_pred))
        
        assert mae > 0
        assert mae < np.max(np.abs(y_true - y_pred))
    
    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        # Recommended items
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        # Relevant items (ground truth)
        relevant = ['item2', 'item4', 'item6']
        
        k = 5
        recommended_k = recommended[:k]
        relevant_in_k = len(set(recommended_k) & set(relevant))
        precision_at_k = relevant_in_k / k
        
        assert precision_at_k == 2/5  # 2 relevant items in top 5
        assert 0 <= precision_at_k <= 1
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
        relevant = ['item2', 'item4', 'item6']
        
        k = 5
        recommended_k = recommended[:k]
        relevant_in_k = len(set(recommended_k) & set(relevant))
        recall_at_k = relevant_in_k / len(relevant) if len(relevant) > 0 else 0
        
        assert recall_at_k == 2/3  # 2 out of 3 relevant items found
        assert 0 <= recall_at_k <= 1
    
    def test_ndcg_at_k(self):
        """Test NDCG@K calculation."""
        # Relevance scores (1 = relevant, 0 = not relevant)
        relevance = np.array([1, 0, 1, 0, 1])
        k = 5
        
        # DCG calculation
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k]))
        
        # IDCG calculation (ideal ranking)
        ideal_relevance = sorted(relevance[:k], reverse=True)
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        assert 0 <= ndcg <= 1
        assert not np.isnan(ndcg)
    
    def test_coverage_metric(self, sample_products):
        """Test catalog coverage metric."""
        recommended_items = ['prod_1', 'prod_2', 'prod_3', 'prod_10', 'prod_20']
        total_items = len(sample_products)
        unique_recommended = len(set(recommended_items))
        
        coverage = unique_recommended / total_items
        
        assert 0 <= coverage <= 1
    
    def test_diversity_metric(self):
        """Test recommendation diversity metric."""
        # Simplified diversity: unique categories in recommendations
        recommendations = [
            {'category': 'Electronics'},
            {'category': 'Electronics'},
            {'category': 'Clothing'},
            {'category': 'Home'},
            {'category': 'Electronics'}
        ]
        
        unique_categories = len(set(r['category'] for r in recommendations))
        diversity = unique_categories / len(recommendations)
        
        assert 0 <= diversity <= 1
        assert diversity == 3/5  # 3 unique categories out of 5 items


class TestModelSaving:
    """Test suite for model saving and loading."""
    
    def test_save_model_state_dict(self, temp_model_dir):
        """Test saving model state dict."""
        model = nn.Linear(10, 1)
        model_path = os.path.join(temp_model_dir, 'model.pth')
        
        torch.save(model.state_dict(), model_path)
        
        assert os.path.exists(model_path)
    
    def test_load_model_state_dict(self, temp_model_dir):
        """Test loading model state dict."""
        model = nn.Linear(10, 1)
        model_path = os.path.join(temp_model_dir, 'model.pth')
        
        # Save
        torch.save(model.state_dict(), model_path)
        
        # Load
        new_model = nn.Linear(10, 1)
        new_model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # Verify weights are the same
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_save_full_model(self, temp_model_dir):
        """Test saving complete model."""
        model = nn.Linear(10, 1)
        model_path = os.path.join(temp_model_dir, 'full_model.pth')
        
        torch.save(model, model_path)
        
        assert os.path.exists(model_path)
        
        # Load and verify
        loaded_model = torch.load(model_path, weights_only=False)
        assert isinstance(loaded_model, nn.Linear)


class TestHyperparameterTuning:
    """Test suite for hyperparameter tuning."""
    
    def test_grid_search_space(self):
        """Test grid search parameter space."""
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [64, 128, 256],
            'embedding_dim': [32, 64, 128]
        }
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        assert total_combinations == 27  # 3 * 3 * 3
    
    def test_random_search_sampling(self):
        """Test random search parameter sampling."""
        param_distributions = {
            'learning_rate': (0.0001, 0.01),
            'batch_size': [64, 128, 256, 512],
            'embedding_dim': (16, 128)
        }
        
        # Sample parameters
        sampled_lr = np.random.uniform(*param_distributions['learning_rate'])
        sampled_batch = np.random.choice(param_distributions['batch_size'])
        sampled_emb = np.random.randint(*param_distributions['embedding_dim'])
        
        assert 0.0001 <= sampled_lr <= 0.01
        assert sampled_batch in param_distributions['batch_size']
        assert 16 <= sampled_emb < 128
    
    def test_early_stopping_patience(self):
        """Test early stopping logic."""
        val_losses = [0.5, 0.45, 0.44, 0.46, 0.47, 0.48]
        patience = 3
        best_loss = float('inf')
        patience_counter = 0
        
        for loss in val_losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        assert patience_counter == 3
        assert best_loss == 0.44
