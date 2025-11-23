import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import json
import os


class DriftDetector:
    """Detect data and concept drift in ML models"""

    def __init__(self):
        self.s3 = boto3.client("s3")
        self.cloudwatch = boto3.client("cloudwatch")
        self.sns = boto3.client("sns")

    def detect_feature_drift(self, current_data, reference_data, threshold=0.05):
        """Detect drift in feature distributions using KS test"""
        drift_detected = {}

        for column in current_data.columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    reference_data[column].dropna(), current_data[column].dropna()
                )

                drift_detected[column] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift": p_value < threshold,
                }

        return drift_detected

    def detect_prediction_drift(
        self, current_predictions, reference_predictions, threshold=0.1
    ):
        """Detect drift in prediction distributions"""
        # Compare distribution of predictions
        statistic, p_value = stats.ks_2samp(reference_predictions, current_predictions)

        # Compare mean and variance
        mean_diff = abs(np.mean(current_predictions) - np.mean(reference_predictions))
        var_ratio = np.var(current_predictions) / np.var(reference_predictions)

        drift_metrics = {
            "ks_statistic": float(statistic),
            "ks_p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "variance_ratio": float(var_ratio),
            "drift_detected": p_value < threshold or mean_diff > 0.1 or var_ratio > 2.0,
        }

        return drift_metrics

    def detect_performance_degradation(self, window_days=7):
        """Monitor model performance metrics over time"""
        # Query CloudWatch for recent metrics
        end_time = datetime.now()
        start_time = end_time - timedelta(days=window_days)

        # Get prediction accuracy, latency, error rate
        metrics = self._get_cloudwatch_metrics(
            namespace="ML/Recommendations",
            metric_names=["Accuracy", "Latency", "ErrorRate"],
            start_time=start_time,
            end_time=end_time,
        )

        # Check for degradation
        degradation = {
            "accuracy_drop": False,
            "latency_increase": False,
            "error_rate_increase": False,
        }

        # Compare to baseline
        if "Accuracy" in metrics and len(metrics["Accuracy"]) > 24:
            current_accuracy = np.mean(metrics["Accuracy"][-24:])  # Last 24 hours
            baseline_accuracy = np.mean(metrics["Accuracy"][:-24])
            if current_accuracy < baseline_accuracy * 0.95:  # 5% drop
                degradation["accuracy_drop"] = True

        return degradation

    def check_data_quality(self, data):
        """Check for data quality issues"""
        issues = []

        # Check for missing values
        missing_pct = data.isnull().sum() / len(data)
        for col, pct in missing_pct.items():
            if pct > 0.1:  # More than 10% missing
                issues.append(f"High missing rate in {col}: {pct:.2%}")

        # Check for outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((data[col] < q1 - 3 * iqr) | (data[col] > q3 + 3 * iqr)).sum()
            if outliers / len(data) > 0.05:  # More than 5% outliers
                issues.append(f"High outlier rate in {col}: {outliers/len(data):.2%}")

        return issues

    def send_alert(self, alert_type, details):
        """Send alert via SNS"""
        topic_arn = os.environ.get("ALERT_TOPIC_ARN")
        if not topic_arn:
            print(f"ALERT_TOPIC_ARN not set. Alert: {alert_type}")
            return

        message = f"""
        Alert Type: {alert_type}
        Timestamp: {datetime.now().isoformat()}

        Details:
        {json.dumps(details, indent=2)}
        """

        self.sns.publish(
            TopicArn=topic_arn, Subject=f"ML Model Alert: {alert_type}", Message=message
        )

    def _get_cloudwatch_metrics(self, namespace, metric_names, start_time, end_time):
        """Retrieve metrics from CloudWatch"""
        metrics = {}

        for metric_name in metric_names:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=["Average"],
            )

            metrics[metric_name] = [
                point["Average"]
                for point in sorted(
                    response["Datapoints"], key=lambda x: x["Timestamp"]
                )
            ]

        return metrics


def load_recent_data(days=7):
    # Placeholder for loading recent data
    return pd.DataFrame()


def load_reference_data():
    # Placeholder for loading reference data
    return pd.DataFrame()


# Lambda handler for scheduled monitoring
def lambda_handler(event, context):
    """Scheduled monitoring job"""
    detector = DriftDetector()

    # Load current and reference data
    current_data = load_recent_data(days=7)
    reference_data = load_reference_data()

    if current_data.empty or reference_data.empty:
        print("No data available for monitoring")
        return {"statusCode": 200, "body": json.dumps({"message": "No data available"})}

    # Detect feature drift
    feature_drift = detector.detect_feature_drift(current_data, reference_data)
    if any(v["drift"] for v in feature_drift.values()):
        detector.send_alert("Feature Drift Detected", feature_drift)

    # Detect performance degradation
    performance = detector.detect_performance_degradation()
    if any(performance.values()):
        detector.send_alert("Performance Degradation", performance)

    # Check data quality
    quality_issues = detector.check_data_quality(current_data)
    if quality_issues:
        detector.send_alert("Data Quality Issues", {"issues": quality_issues})

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "feature_drift": feature_drift,
                "performance": performance,
                "quality_issues": quality_issues,
            }
        ),
    }
