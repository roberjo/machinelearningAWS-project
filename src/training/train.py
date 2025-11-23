import argparse
import json

import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model"""

    def __init__(
        self, num_users, num_items, embedding_dim=50, hidden_layers=[128, 64, 32]
    ):
        super().__init__()

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.mlp(x).squeeze()


class RecommenderTrainer:
    """Training pipeline for recommendation model"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.user_id_map = {}
        self.item_id_map = {}

    def load_data(self):
        """Load and preprocess training data"""
        print("Loading data...")

        # Load from S3 or local
        interactions = pd.read_csv(f"{self.config['data_path']}/interactions.csv")
        users = pd.read_csv(f"{self.config['data_path']}/users.csv")
        products = pd.read_csv(f"{self.config['data_path']}/products.csv")

        # Filter to only purchases/ratings
        interactions = interactions[
            interactions["interaction_type"].isin(["purchase", "rating"])
        ].copy()

        # Create implicit feedback (1 for purchase, rating/5 for explicit)
        interactions["feedback"] = interactions.apply(
            lambda x: 1.0 if x["interaction_type"] == "purchase" else x["rating"] / 5.0,
            axis=1,
        )

        # Create user and item ID mappings
        unique_users = interactions["user_id"].unique()
        unique_items = interactions["product_id"].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        interactions["user_idx"] = interactions["user_id"].map(self.user_id_map)
        interactions["item_idx"] = interactions["product_id"].map(self.item_id_map)

        return interactions, users, products

    def prepare_features(self, interactions, users, products):
        """Feature engineering"""
        print("Engineering features...")

        # User features
        user_stats = (
            interactions.groupby("user_id")
            .agg({"feedback": ["count", "mean"], "timestamp": "max"})
            .reset_index()
        )
        user_stats.columns = [
            "user_id",
            "num_interactions",
            "avg_rating",
            "last_interaction",
        ]

        # Item features
        item_stats = (
            interactions.groupby("product_id")
            .agg({"feedback": ["count", "mean"]})
            .reset_index()
        )
        item_stats.columns = ["product_id", "num_interactions", "avg_rating"]

        # Merge features
        interactions = interactions.merge(
            user_stats, on="user_id", how="left", suffixes=("", "_user")
        )
        interactions = interactions.merge(
            item_stats, on="product_id", how="left", suffixes=("", "_item")
        )

        return interactions

    def split_data(self, interactions):
        """Temporal train/test split"""
        print("Splitting data...")

        # Sort by timestamp
        interactions = interactions.sort_values("timestamp")

        # Use last 20% for test, 10% for validation
        n = len(interactions)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        train_data = interactions[:train_end]
        val_data = interactions[train_end:val_end]
        test_data = interactions[val_end:]

        return train_data, val_data, test_data

    def train_model(self, train_data, val_data):
        """Train the model"""
        print("Training model...")

        num_users = len(self.user_id_map)
        num_items = len(self.item_id_map)

        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.config.get("embedding_dim", 50),
            hidden_layers=self.config.get("hidden_layers", [128, 64, 32]),
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.get("learning_rate", 0.001)
        )

        # Prepare data loaders
        train_loader = self._create_dataloader(
            train_data, batch_size=self.config.get("batch_size", 256)
        )
        val_loader = self._create_dataloader(
            val_data, batch_size=self.config.get("batch_size", 256)
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.get("num_epochs", 50)):
            # Train
            self.model.train()
            train_loss = 0
            for batch_users, batch_items, batch_feedback in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_feedback)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_users, batch_items, batch_feedback in val_loader:
                    predictions = self.model(batch_users, batch_items)
                    loss = criterion(predictions, batch_feedback)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(
                f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
                if patience_counter >= self.config.get("patience", 5):
                    print("Early stopping triggered")
                    break

        return self.model

    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        print("Evaluating model...")

        self.model.eval()
        test_loader = self._create_dataloader(
            test_data, batch_size=self.config.get("batch_size", 256)
        )

        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_users, batch_items, batch_feedback in test_loader:
                preds = self.model(batch_users, batch_items)
                predictions.extend(preds.numpy())
                actuals.extend(batch_feedback.numpy())

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        # Calculate ranking metrics (Precision@K, Recall@K, NDCG)
        precision_at_10 = self._calculate_precision_at_k(test_data, k=10)
        recall_at_10 = self._calculate_recall_at_k(test_data, k=10)
        ndcg_at_10 = self._calculate_ndcg_at_k(test_data, k=10)

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "precision@10": float(precision_at_10),
            "recall@10": float(recall_at_10),
            "ndcg@10": float(ndcg_at_10),
            "timestamp": datetime.now().isoformat(),
        }

        print(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def save_model(self, model_path):
        """Save model artifacts"""
        print(f"Saving model to {model_path}")

        # Save model weights
        torch.save(self.model.state_dict(), f"{model_path}/model.pth")

        # Save ID mappings
        joblib.dump(self.user_id_map, f"{model_path}/user_id_map.pkl")
        joblib.dump(self.item_id_map, f"{model_path}/item_id_map.pkl")

        # Save config
        with open(f"{model_path}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def _create_dataloader(self, data, batch_size):
        """Create PyTorch DataLoader"""
        from torch.utils.data import TensorDataset, DataLoader

        users = torch.LongTensor(data["user_idx"].values)
        items = torch.LongTensor(data["item_idx"].values)
        feedback = torch.FloatTensor(data["feedback"].values)

        dataset = TensorDataset(users, items, feedback)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def _calculate_precision_at_k(self, test_data, k=10):
        """Calculate Precision@K"""
        # Get top K recommendations for each user
        # Compare with actual purchases in test set
        # Implementation details omitted for brevity
        return 0.0  # Placeholder

    def _calculate_recall_at_k(self, test_data, k=10):
        """Calculate Recall@K"""
        return 0.0  # Placeholder

    def _calculate_ndcg_at_k(self, test_data, k=10):
        """Calculate NDCG@K"""
        return 0.0  # Placeholder

    def _save_checkpoint(self):
        """Save training checkpoint"""
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/opt/ml/input/data/training")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output")
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--hidden-layers", type=str, default="128,64,32")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)

    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "embedding_dim": args.embedding_dim,
        "hidden_layers": [int(x) for x in args.hidden_layers.split(",")],
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
    }

    # Train
    trainer = RecommenderTrainer(config)
    interactions, users, products = trainer.load_data()
    interactions = trainer.prepare_features(interactions, users, products)
    train_data, val_data, test_data = trainer.split_data(interactions)

    trainer.train_model(train_data, val_data)
    metrics = trainer.evaluate_model(test_data)

    # Save
    trainer.save_model(args.model_dir)

    # Save metrics
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
