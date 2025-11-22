# API Reference Documentation

## Overview
The Recommendation API provides personalized product recommendations for e-commerce platforms. This document describes all available endpoints, request/response formats, authentication, error handling, and usage examples.

## Base URL
```
Production:  https://api.recommendation.example.com/v1
Staging:     https://api-staging.recommendation.example.com/v1
Development: https://api-dev.recommendation.example.com/v1
```

## Authentication

### API Key Authentication
All requests must include an API key in the request header:

```http
X-API-Key: your_api_key_here
```

**Obtaining an API Key**:
Contact the API administrator or use the self-service portal at `https://portal.recommendation.example.com`

### Rate Limiting
- **Standard Tier**: 1,000 requests/minute
- **Premium Tier**: 10,000 requests/minute
- **Enterprise Tier**: Custom limits

**Rate Limit Headers**:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1640000000
```

## Endpoints

### 1. Get Recommendations

Get personalized product recommendations for a user.

**Endpoint**: `POST /recommend`

**Request Body**:
```json
{
  "user_id": "string (required)",
  "num_recommendations": "integer (optional, default: 10, max: 50)",
  "exclude_purchased": "boolean (optional, default: true)",
  "context": {
    "page": "string (optional)",
    "device": "string (optional)",
    "session_id": "string (optional)"
  },
  "filters": {
    "categories": ["string"],
    "price_min": "number",
    "price_max": "number",
    "brands": ["string"]
  }
}
```

**Request Example**:
```json
{
  "user_id": "user_12345",
  "num_recommendations": 10,
  "exclude_purchased": true,
  "context": {
    "page": "homepage",
    "device": "mobile",
    "session_id": "sess_abc123"
  },
  "filters": {
    "categories": ["Electronics", "Home"],
    "price_min": 20.00,
    "price_max": 500.00
  }
}
```

**Response** (200 OK):
```json
{
  "recommendations": [
    {
      "product_id": "prod_456",
      "name": "Wireless Headphones",
      "score": 0.95,
      "reason": "Based on your recent purchases",
      "category": "Electronics",
      "price": 99.99,
      "image_url": "https://cdn.example.com/products/prod_456.jpg",
      "metadata": {
        "brand": "TechBrand",
        "rating": 4.5,
        "num_reviews": 1247
      }
    },
    {
      "product_id": "prod_789",
      "name": "Smart Speaker",
      "score": 0.89,
      "reason": "Customers who bought items you liked also bought this",
      "category": "Electronics",
      "price": 79.99,
      "image_url": "https://cdn.example.com/products/prod_789.jpg",
      "metadata": {
        "brand": "SmartHome",
        "rating": 4.3,
        "num_reviews": 892
      }
    }
  ],
  "model_version": "v2.3.1",
  "inference_time_ms": 45,
  "request_id": "req_xyz789",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

**Error Responses**:

*400 Bad Request*:
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "user_id is required",
    "request_id": "req_xyz789"
  }
}
```

*404 Not Found*:
```json
{
  "error": {
    "code": "USER_NOT_FOUND",
    "message": "User user_12345 not found",
    "request_id": "req_xyz789"
  }
}
```

*429 Too Many Requests*:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 60 seconds",
    "retry_after": 60,
    "request_id": "req_xyz789"
  }
}
```

*500 Internal Server Error*:
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An internal error occurred. Please try again later",
    "request_id": "req_xyz789"
  }
}
```

---

### 2. Submit Feedback

Submit user feedback on recommendations (clicks, purchases, ratings).

**Endpoint**: `POST /feedback`

**Request Body**:
```json
{
  "user_id": "string (required)",
  "product_id": "string (required)",
  "feedback_type": "string (required: 'click', 'purchase', 'rating', 'dismiss')",
  "rating": "number (optional, 1-5, required if feedback_type='rating')",
  "context": {
    "recommendation_id": "string (optional)",
    "position": "integer (optional)",
    "session_id": "string (optional)"
  }
}
```

**Request Example**:
```json
{
  "user_id": "user_12345",
  "product_id": "prod_456",
  "feedback_type": "purchase",
  "context": {
    "recommendation_id": "rec_abc123",
    "position": 1,
    "session_id": "sess_abc123"
  }
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "message": "Feedback recorded",
  "request_id": "req_xyz789"
}
```

---

### 3. Get Similar Products

Get products similar to a given product.

**Endpoint**: `GET /similar/{product_id}`

**Path Parameters**:
- `product_id` (string, required): Product identifier

**Query Parameters**:
- `num_recommendations` (integer, optional, default: 10, max: 50)
- `include_metadata` (boolean, optional, default: true)

**Request Example**:
```http
GET /similar/prod_456?num_recommendations=5&include_metadata=true
```

**Response** (200 OK):
```json
{
  "product_id": "prod_456",
  "similar_products": [
    {
      "product_id": "prod_789",
      "name": "Premium Wireless Headphones",
      "similarity_score": 0.92,
      "category": "Electronics",
      "price": 149.99
    },
    {
      "product_id": "prod_101",
      "name": "Noise-Cancelling Headphones",
      "similarity_score": 0.88,
      "category": "Electronics",
      "price": 199.99
    }
  ],
  "model_version": "v2.3.1",
  "request_id": "req_xyz789"
}
```

---

### 4. Get Trending Products

Get currently trending products globally or by category.

**Endpoint**: `GET /trending`

**Query Parameters**:
- `category` (string, optional): Filter by category
- `time_window` (string, optional, default: "24h", options: "1h", "24h", "7d")
- `num_products` (integer, optional, default: 20, max: 100)

**Request Example**:
```http
GET /trending?category=Electronics&time_window=24h&num_products=10
```

**Response** (200 OK):
```json
{
  "trending_products": [
    {
      "product_id": "prod_456",
      "name": "Wireless Headphones",
      "trending_score": 0.95,
      "category": "Electronics",
      "price": 99.99,
      "view_count_24h": 15420,
      "purchase_count_24h": 342
    }
  ],
  "category": "Electronics",
  "time_window": "24h",
  "request_id": "req_xyz789"
}
```

---

### 5. Health Check

Check API health status.

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "v2.3.1",
  "timestamp": "2024-01-15T14:30:00Z",
  "services": {
    "api": "healthy",
    "model": "healthy",
    "database": "healthy"
  }
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "unhealthy",
  "version": "v2.3.1",
  "timestamp": "2024-01-15T14:30:00Z",
  "services": {
    "api": "healthy",
    "model": "unhealthy",
    "database": "healthy"
  }
}
```

---

### 6. Get Model Metrics

Get current model performance metrics (requires admin API key).

**Endpoint**: `GET /metrics`

**Headers**:
```http
X-API-Key: admin_api_key_here
```

**Response** (200 OK):
```json
{
  "model_version": "v2.3.1",
  "deployed_at": "2024-01-15T10:00:00Z",
  "metrics": {
    "online": {
      "ctr": 3.2,
      "conversion_rate": 1.8,
      "avg_order_value": 82.50,
      "requests_24h": 1500000
    },
    "offline": {
      "ndcg@10": 0.48,
      "precision@10": 0.38,
      "recall@10": 0.28
    },
    "performance": {
      "avg_latency_ms": 45,
      "p95_latency_ms": 120,
      "p99_latency_ms": 250,
      "error_rate": 0.05
    }
  },
  "traffic_split": {
    "champion": {
      "version": "v2.3.1",
      "percentage": 90
    },
    "challenger": {
      "version": "v2.4.0",
      "percentage": 10
    }
  }
}
```

---

## Request/Response Formats

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`

### Character Encoding
All requests and responses use UTF-8 encoding.

### Date/Time Format
All timestamps use ISO 8601 format: `YYYY-MM-DDTHH:MM:SSZ` (UTC)

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `MISSING_PARAMETER` | 400 | Required parameter missing |
| `INVALID_PARAMETER` | 400 | Parameter value invalid |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `USER_NOT_FOUND` | 404 | User ID not found |
| `PRODUCT_NOT_FOUND` | 404 | Product ID not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Usage Examples

### Python
```python
import requests

API_URL = "https://api.recommendation.example.com/v1"
API_KEY = "your_api_key_here"

def get_recommendations(user_id, num_recommendations=10):
    """Get personalized recommendations"""
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "user_id": user_id,
        "num_recommendations": num_recommendations,
        "exclude_purchased": True,
        "context": {
            "page": "homepage",
            "device": "web"
        }
    }
    
    response = requests.post(
        f"{API_URL}/recommend",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Usage
recommendations = get_recommendations("user_12345", num_recommendations=10)
if recommendations:
    for rec in recommendations['recommendations']:
        print(f"{rec['name']}: {rec['score']:.2f}")
```

### JavaScript (Node.js)
```javascript
const axios = require('axios');

const API_URL = 'https://api.recommendation.example.com/v1';
const API_KEY = 'your_api_key_here';

async function getRecommendations(userId, numRecommendations = 10) {
  try {
    const response = await axios.post(
      `${API_URL}/recommend`,
      {
        user_id: userId,
        num_recommendations: numRecommendations,
        exclude_purchased: true,
        context: {
          page: 'homepage',
          device: 'web'
        }
      },
      {
        headers: {
          'X-API-Key': API_KEY,
          'Content-Type': 'application/json'
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.status);
    console.error(error.response?.data);
    return null;
  }
}

// Usage
getRecommendations('user_12345', 10)
  .then(data => {
    if (data) {
      data.recommendations.forEach(rec => {
        console.log(`${rec.name}: ${rec.score.toFixed(2)}`);
      });
    }
  });
```

### cURL
```bash
curl -X POST https://api.recommendation.example.com/v1/recommend \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "num_recommendations": 10,
    "exclude_purchased": true,
    "context": {
      "page": "homepage",
      "device": "web"
    }
  }'
```

---

## Best Practices

### 1. Caching
Cache recommendations on the client side for 5-10 minutes to reduce API calls:
```python
from functools import lru_cache
from datetime import datetime, timedelta

class RecommendationCache:
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, user_id):
        if user_id in self.cache:
            data, timestamp = self.cache[user_id]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
        return None
    
    def set(self, user_id, data):
        self.cache[user_id] = (data, datetime.now())
```

### 2. Error Handling
Always implement retry logic with exponential backoff:
```python
import time

def get_recommendations_with_retry(user_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            return get_recommendations(user_id)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
```

### 3. Batch Requests
For multiple users, batch requests when possible:
```python
def get_batch_recommendations(user_ids):
    """Get recommendations for multiple users"""
    # Note: Batch endpoint not yet available, use concurrent requests
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(get_recommendations, user_ids)
    
    return list(results)
```

### 4. Context Enrichment
Always include context for better recommendations:
```python
payload = {
    "user_id": user_id,
    "num_recommendations": 10,
    "context": {
        "page": "product_page",  # Current page
        "device": "mobile",       # Device type
        "session_id": session_id, # Session tracking
        "referrer": "email",      # Traffic source
        "time_of_day": "evening"  # Time context
    }
}
```

### 5. Feedback Loop
Always submit feedback to improve recommendations:
```python
def track_recommendation_click(user_id, product_id, recommendation_id):
    """Track when user clicks on a recommendation"""
    submit_feedback(
        user_id=user_id,
        product_id=product_id,
        feedback_type="click",
        context={"recommendation_id": recommendation_id}
    )

def track_purchase(user_id, product_id, recommendation_id):
    """Track when user purchases a recommended product"""
    submit_feedback(
        user_id=user_id,
        product_id=product_id,
        feedback_type="purchase",
        context={"recommendation_id": recommendation_id}
    )
```

---

## Webhooks (Coming Soon)

Subscribe to events for real-time updates:
- `model.deployed`: New model version deployed
- `model.performance.degraded`: Model performance below threshold
- `user.recommendations.ready`: Batch recommendations completed

---

## Changelog

### v2.3.1 (2024-01-15)
- Improved recommendation diversity
- Added support for price filters
- Reduced p95 latency by 20%

### v2.2.0 (2023-12-01)
- Added trending products endpoint
- Implemented A/B testing framework
- Enhanced cold-start handling

### v2.1.0 (2023-10-15)
- Added similar products endpoint
- Improved error messages
- Added request_id to all responses

---

## Support

**Documentation**: https://docs.recommendation.example.com  
**Status Page**: https://status.recommendation.example.com  
**Support Email**: support@recommendation.example.com  
**Slack Community**: https://slack.recommendation.example.com
