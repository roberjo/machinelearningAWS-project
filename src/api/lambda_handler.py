"""
Lambda handler for recommendation API.
"""

import json
import os
from datetime import datetime

import boto3
import joblib
import numpy as np


class RecommendationService:
    """Recommendation inference service"""

    # pylint: disable=too-few-public-methods

    def __init__(self):
        self.s3 = boto3.client("s3")
        self.sagemaker_runtime = boto3.client("sagemaker-runtime")
        self.dynamodb = boto3.resource("dynamodb")

        # Load model artifacts
        self.model = None
        self.user_id_map = None
        self.item_id_map = None
        self.reverse_item_map = None

        self._load_model()

    def _load_model(self):
        """Load model from S3"""
        model_bucket = os.environ.get("MODEL_BUCKET")
        model_version = os.environ.get("MODEL_VERSION", "latest")

        if not model_bucket:
            print("MODEL_BUCKET env var not set, skipping model load")
            return

        # Download model artifacts
        try:
            self.s3.download_file(
                model_bucket, f"{model_version}/user_id_map.pkl", "/tmp/user_id_map.pkl"
            )
            self.s3.download_file(
                model_bucket, f"{model_version}/item_id_map.pkl", "/tmp/item_id_map.pkl"
            )

            self.user_id_map = joblib.load("/tmp/user_id_map.pkl")
            self.item_id_map = joblib.load("/tmp/item_id_map.pkl")
            self.reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error loading model: {e}")

    def get_recommendations(
        self, user_id, num_recommendations=10, exclude_purchased=True, context=None
    ):
        """Get personalized recommendations"""
        start_time = datetime.now()

        # Check if user exists
        if self.user_id_map is None or user_id not in self.user_id_map:
            # Cold start: return popular items
            return self._get_popular_items(num_recommendations)

        user_idx = self.user_id_map[user_id]

        # Get user's purchase history if excluding purchased items
        purchased_items = set()
        if exclude_purchased:
            purchased_items = self._get_user_purchases(user_id)

        # Generate scores for all items
        scores = []
        for item_id, item_idx in self.item_id_map.items():
            if item_id not in purchased_items:
                # Call SageMaker endpoint or use loaded model
                score = self._predict_score(user_idx, item_idx)
                scores.append((item_id, score))

        # Sort by score and get top K
        scores.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = scores[:num_recommendations]

        # Format response
        recommendations = []
        for item_id, score in top_recommendations:
            recommendations.append(
                {
                    "product_id": item_id,
                    "score": float(score),
                    "reason": self._generate_reason(user_id, item_id, context),
                }
            )

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        # Log inference for monitoring
        self._log_inference(user_id, recommendations, inference_time)

        return {
            "recommendations": recommendations,
            "model_version": os.environ.get("MODEL_VERSION", "latest"),
            "inference_time_ms": inference_time,
        }

    def _predict_score(self, user_idx, item_idx):
        """Predict score for user-item pair"""
        # Option 1: Call SageMaker endpoint
        endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT")
        if endpoint_name:
            payload = json.dumps({"user_idx": user_idx, "item_idx": item_idx})
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="application/json", Body=payload
            )
            result = json.loads(response["Body"].read().decode())
            return result["score"]

        # Option 2: Use locally loaded model (for small models)
        # This requires loading the full model in Lambda
        return 0.5  # Placeholder

    def _get_user_purchases(self, user_id):
        """Get user's purchase history"""
        # Query DynamoDB or cache
        try:
            table = self.dynamodb.Table(os.environ["INTERACTIONS_TABLE"])
            response = table.query(
                KeyConditionExpression="user_id = :uid AND interaction_type = :type",
                ExpressionAttributeValues={":uid": user_id, ":type": "purchase"},
            )
            return {item["product_id"] for item in response["Items"]}
        except Exception:  # pylint: disable=broad-exception-caught
            return set()

    def _get_popular_items(self, num_items):
        """Get popular items for cold start users"""
        try:
            table = self.dynamodb.Table(os.environ["POPULAR_ITEMS_TABLE"])
            response = table.scan(Limit=num_items)

            recommendations = []
            for item in response["Items"]:
                recommendations.append(
                    {
                        "product_id": item["product_id"],
                        "score": float(item["popularity_score"]),
                        "reason": "Popular item",
                    }
                )

            return {
                "recommendations": recommendations,
                "model_version": "baseline",
                "inference_time_ms": 0,
            }
        except Exception:  # pylint: disable=broad-exception-caught
            return {
                "recommendations": [],
                "model_version": "baseline",
                "inference_time_ms": 0,
            }

    def _generate_reason(
        self, user_id, item_id, context
    ):  # pylint: disable=unused-argument
        """Generate explanation for recommendation"""
        reasons = [
            "Based on your recent purchases",
            "Customers who bought items you liked also bought this",
            "Popular in your category",
            "Trending now",
            "Based on your browsing history",
        ]
        return np.random.choice(reasons)

    def _log_inference(self, user_id, recommendations, inference_time):
        """Log inference for monitoring"""
        try:
            table = self.dynamodb.Table(os.environ["INFERENCE_LOG_TABLE"])
            table.put_item(
                Item={
                    "log_id": f"{user_id}_{datetime.now().isoformat()}",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "num_recommendations": len(recommendations),
                    "inference_time_ms": inference_time,
                    "model_version": os.environ.get("MODEL_VERSION", "latest"),
                }
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Failed to log inference: {e}")


# Lambda handler
recommendation_service = None  # pylint: disable=invalid-name


def lambda_handler(event, context):  # pylint: disable=unused-argument
    """Main Lambda handler for recommendation API"""
    global recommendation_service  # pylint: disable=global-statement

    # Initialize service (cached across invocations)
    if recommendation_service is None:
        recommendation_service = RecommendationService()

    try:
        # Parse request
        body = (
            json.loads(event["body"])
            if isinstance(event.get("body"), str)
            else event.get("body", {})
        )

        user_id = body.get("user_id")
        if not user_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "user_id is required"}),
            }

        num_recommendations = body.get("num_recommendations", 10)
        exclude_purchased = body.get("exclude_purchased", True)
        context_data = body.get("context", {})

        # Get recommendations
        result = recommendation_service.get_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations,
            exclude_purchased=exclude_purchased,
            context=context_data,
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result),
        }

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
        }
