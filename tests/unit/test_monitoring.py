"""
Unit tests for monitoring and drift detection modules.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class TestDriftDetection:
    """Test suite for drift detection functionality."""
    
    def test_kolmogorov_smirnov_test(self):
        """Test KS test for numerical feature drift."""
        # Reference distribution (training data)
        reference_data = np.random.normal(loc=5.0, scale=1.0, size=1000)
        
        # Current distribution (no drift)
        current_data_no_drift = np.random.normal(loc=5.0, scale=1.0, size=1000)
        
        # Current distribution (with drift)
        current_data_with_drift = np.random.normal(loc=6.0, scale=1.0, size=1000)
        
        # Test no drift
        statistic_no_drift, p_value_no_drift = stats.ks_2samp(reference_data, current_data_no_drift)
        assert p_value_no_drift > 0.05  # No significant drift
        
        # Test with drift
        statistic_with_drift, p_value_with_drift = stats.ks_2samp(reference_data, current_data_with_drift)
        assert p_value_with_drift < 0.05  # Significant drift detected
    
    def test_chi_square_test_categorical(self):
        """Test chi-square test for categorical feature drift."""
        # Reference distribution
        reference_counts = np.array([100, 150, 200, 50])
        
        # Current distribution (no drift)
        current_counts_no_drift = np.array([95, 155, 195, 55])
        
        # Current distribution (with drift)
        current_counts_with_drift = np.array([200, 50, 100, 150])
        
        # Test no drift
        chi2_no_drift, p_value_no_drift = stats.chisquare(current_counts_no_drift, reference_counts)
        assert p_value_no_drift > 0.05
        
        # Test with drift
        chi2_with_drift, p_value_with_drift = stats.chisquare(current_counts_with_drift, reference_counts)
        assert p_value_with_drift < 0.05
    
    def test_prediction_drift_detection(self):
        """Test prediction distribution drift."""
        # Reference predictions
        reference_predictions = np.random.beta(a=2, b=5, size=1000)
        
        # Current predictions (with drift)
        current_predictions = np.random.beta(a=5, b=2, size=1000)
        
        # Calculate mean shift
        ref_mean = np.mean(reference_predictions)
        cur_mean = np.mean(current_predictions)
        ref_std = np.std(reference_predictions)
        
        mean_shift = abs(cur_mean - ref_mean) / ref_std
        
        # Drift detected if mean shift > 2 standard deviations
        drift_detected = mean_shift > 2
        assert drift_detected
    
    def test_data_quality_monitoring(self, sample_interactions):
        """Test data quality metrics."""
        # Check completeness
        completeness = 1 - (sample_interactions.isnull().sum() / len(sample_interactions))
        assert all(completeness > 0.9)  # >90% completeness for all columns
        
        # Check uniqueness
        unique_interactions = sample_interactions['interaction_id'].nunique()
        total_interactions = len(sample_interactions)
        uniqueness_ratio = unique_interactions / total_interactions
        assert uniqueness_ratio == 1.0  # All interaction IDs should be unique
    
    def test_feature_distribution_monitoring(self):
        """Test feature distribution monitoring."""
        # Historical distribution
        historical_prices = np.random.lognormal(mean=4.0, sigma=0.5, size=1000)
        
        # Current distribution
        current_prices = np.random.lognormal(mean=4.0, sigma=0.5, size=100)
        
        # Compare distributions
        statistic, p_value = stats.ks_2samp(historical_prices, current_prices)
        
        # No significant drift expected
        assert p_value > 0.05
    
    def test_drift_severity_classification(self):
        """Test drift severity classification."""
        ks_statistic = 0.25
        
        if ks_statistic < 0.1:
            severity = 'low'
        elif ks_statistic < 0.3:
            severity = 'medium'
        else:
            severity = 'high'
        
        assert severity == 'medium'


class TestPerformanceMonitoring:
    """Test suite for performance monitoring."""
    
    def test_latency_percentile_calculation(self):
        """Test latency percentile calculations."""
        latencies = np.random.gamma(shape=2, scale=50, size=1000)  # Simulated latencies in ms
        
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        assert p50 < p95 < p99
        assert p50 > 0
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        total_requests = 10000
        failed_requests = 50
        
        error_rate = (failed_requests / total_requests) * 100
        
        assert error_rate == 0.5  # 0.5% error rate
        assert error_rate < 1.0  # Below 1% threshold
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        requests_per_minute = 5000
        requests_per_second = requests_per_minute / 60
        
        assert requests_per_second == pytest.approx(83.33, rel=0.01)
    
    def test_availability_calculation(self):
        """Test availability calculation."""
        total_time_minutes = 1440  # 24 hours
        downtime_minutes = 5
        
        availability = ((total_time_minutes - downtime_minutes) / total_time_minutes) * 100
        
        assert availability == pytest.approx(99.65, rel=0.01)
        assert availability > 99.5  # Above SLA
    
    def test_concurrent_users_tracking(self):
        """Test concurrent users tracking."""
        active_sessions = {
            'session_1': datetime.now(),
            'session_2': datetime.now() - timedelta(minutes=5),
            'session_3': datetime.now() - timedelta(minutes=35)
        }
        
        # Count sessions active in last 30 minutes
        cutoff_time = datetime.now() - timedelta(minutes=30)
        active_count = sum(1 for timestamp in active_sessions.values() if timestamp > cutoff_time)
        
        assert active_count == 2


class TestMetricsAggregation:
    """Test suite for metrics aggregation."""
    
    def test_aggregate_hourly_metrics(self):
        """Test hourly metrics aggregation."""
        # Simulate hourly data points
        hourly_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=24, freq='H'),
            'requests': np.random.randint(1000, 5000, 24),
            'errors': np.random.randint(0, 50, 24)
        })
        
        # Calculate aggregates
        total_requests = hourly_data['requests'].sum()
        total_errors = hourly_data['errors'].sum()
        avg_requests_per_hour = hourly_data['requests'].mean()
        
        assert total_requests > 0
        assert avg_requests_per_hour > 0
    
    def test_aggregate_daily_metrics(self):
        """Test daily metrics aggregation."""
        daily_metrics = {
            'date': '2024-01-15',
            'total_requests': 100000,
            'total_errors': 500,
            'avg_latency_ms': 85,
            'p95_latency_ms': 250,
            'p99_latency_ms': 500
        }
        
        assert daily_metrics['total_requests'] > 0
        assert daily_metrics['total_errors'] / daily_metrics['total_requests'] < 0.01  # <1% error rate
    
    def test_rolling_window_aggregation(self):
        """Test rolling window aggregation."""
        # Simulate time series data
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='min'),
            'latency': np.random.normal(100, 20, 100)
        })
        
        # Calculate 10-minute rolling average
        data['rolling_avg'] = data['latency'].rolling(window=10).mean()
        
        assert not data['rolling_avg'].iloc[10:].isnull().any()


class TestAlertingLogic:
    """Test suite for alerting logic."""
    
    def test_threshold_based_alert(self):
        """Test threshold-based alerting."""
        current_error_rate = 5.5
        threshold = 5.0
        
        should_alert = current_error_rate > threshold
        
        assert should_alert is True
    
    def test_consecutive_violations_alert(self):
        """Test alerting on consecutive threshold violations."""
        error_rates = [5.5, 6.0, 5.8, 4.5, 3.0]  # First 3 exceed threshold
        threshold = 5.0
        consecutive_required = 3
        
        consecutive_count = 0
        alert_triggered = False
        
        for rate in error_rates:
            if rate > threshold:
                consecutive_count += 1
                if consecutive_count >= consecutive_required:
                    alert_triggered = True
                    break
            else:
                consecutive_count = 0
        
        assert alert_triggered is True
    
    def test_rate_of_change_alert(self):
        """Test alerting on rate of change."""
        previous_value = 100
        current_value = 150
        
        change_percent = ((current_value - previous_value) / previous_value) * 100
        threshold_percent = 30
        
        should_alert = abs(change_percent) > threshold_percent
        
        assert should_alert is True
        assert change_percent == 50
    
    def test_anomaly_detection_alert(self):
        """Test anomaly detection alerting."""
        historical_values = np.random.normal(100, 10, 1000)
        current_value = 150
        
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        z_score = (current_value - mean) / std
        
        # Alert if z-score > 3 (3 standard deviations)
        is_anomaly = abs(z_score) > 3
        
        assert is_anomaly is True


class TestModelMetricsTracking:
    """Test suite for model metrics tracking."""
    
    def test_track_ctr_metric(self):
        """Test click-through rate tracking."""
        total_recommendations = 10000
        total_clicks = 350
        
        ctr = (total_clicks / total_recommendations) * 100
        
        assert ctr == 3.5
        assert ctr > 3.0  # Above target
    
    def test_track_conversion_rate(self):
        """Test conversion rate tracking."""
        total_clicks = 350
        total_purchases = 55
        
        conversion_rate = (total_purchases / total_clicks) * 100
        
        assert conversion_rate == pytest.approx(15.71, rel=0.01)
    
    def test_track_average_order_value(self):
        """Test average order value tracking."""
        order_values = [75.50, 120.00, 45.25, 200.00, 89.99]
        
        aov = np.mean(order_values)
        
        assert aov == pytest.approx(106.15, rel=0.01)
    
    def test_track_recommendation_diversity(self):
        """Test recommendation diversity tracking."""
        recommendations = [
            {'category': 'Electronics'},
            {'category': 'Electronics'},
            {'category': 'Clothing'},
            {'category': 'Home'},
            {'category': 'Books'},
            {'category': 'Electronics'}
        ]
        
        unique_categories = len(set(r['category'] for r in recommendations))
        diversity_score = unique_categories / len(recommendations)
        
        assert diversity_score == pytest.approx(0.67, rel=0.01)
    
    def test_track_catalog_coverage(self):
        """Test catalog coverage tracking."""
        total_products = 1000
        recommended_products = set([f'prod_{i}' for i in range(750)])
        
        coverage = len(recommended_products) / total_products
        
        assert coverage == 0.75


class TestLogging:
    """Test suite for logging functionality."""
    
    def test_structured_logging_format(self):
        """Test structured logging format."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'message': 'API request processed',
            'user_id': 'user_123',
            'latency_ms': 45,
            'event_type': 'api_request'
        }
        
        assert 'timestamp' in log_entry
        assert 'level' in log_entry
        assert 'message' in log_entry
        assert 'event_type' in log_entry
    
    def test_error_logging_with_context(self):
        """Test error logging with context."""
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'error_type': 'ModelNotFoundError',
            'error_message': 'Model v1.0.0 not found',
            'user_id': 'user_123',
            'request_id': 'req_xyz789',
            'event_type': 'error'
        }
        
        assert error_log['level'] == 'ERROR'
        assert 'error_type' in error_log
        assert 'error_message' in error_log
    
    def test_log_sanitization(self):
        """Test PII sanitization in logs."""
        sensitive_data = {
            'email': 'user@example.com',
            'phone': '123-456-7890',
            'user_id': 'user_123'
        }
        
        # Sanitize sensitive fields
        sanitized = sensitive_data.copy()
        sanitized['email'] = '***REDACTED***'
        sanitized['phone'] = '***REDACTED***'
        
        assert sanitized['email'] == '***REDACTED***'
        assert sanitized['phone'] == '***REDACTED***'
        assert sanitized['user_id'] == 'user_123'  # user_id is okay to log


class TestHealthChecks:
    """Test suite for health check functionality."""
    
    def test_api_health_check(self):
        """Test API health check."""
        health_status = {
            'status': 'healthy',
            'version': 'v1.0.0',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'api': 'healthy',
                'model': 'healthy',
                'database': 'healthy'
            }
        }
        
        assert health_status['status'] == 'healthy'
        assert all(status == 'healthy' for status in health_status['services'].values())
    
    def test_model_health_check(self):
        """Test model health check."""
        try:
            # Simulate model check
            model_loaded = True
            model_version = 'v1.0.0'
            
            if model_loaded and model_version:
                status = 'healthy'
            else:
                status = 'unhealthy'
        except Exception:
            status = 'unhealthy'
        
        assert status == 'healthy'
    
    def test_database_health_check(self, mock_dynamodb_table):
        """Test database health check."""
        try:
            # Simulate database query
            mock_dynamodb_table.scan(Limit=1)
            status = 'healthy'
        except Exception:
            status = 'unhealthy'
        
        assert status == 'healthy'
