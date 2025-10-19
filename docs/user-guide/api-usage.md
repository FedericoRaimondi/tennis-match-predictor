# API Usage

The FastAPI backend provides RESTful endpoints for match prediction and data retrieval.

## API Endpoints

### Health Check

**GET** `/`

Returns API status and metadata.

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "healthy",
  "service": "Tennis Match Predictor API",
  "version": "0.1.0"
}
```

### Predict Winner

**POST** `/predict_winner`

Predicts the winner between two players.

Request:
```bash
curl -X POST "http://localhost:8000/predict_winner" \
  -H "Content-Type: application/json" \
  -d '{
    "player1": "Novak Djokovic",
    "player2": "Rafael Nadal",
    "tournament": "Wimbledon"
  }'
```

Response:
```json
{
  "player1": "Novak Djokovic",
  "player2": "Rafael Nadal",
  "player1_win_probability": 0.62,
  "player2_win_probability": 0.38,
  "predicted_winner": "Novak Djokovic",
  "confidence": 0.62
}
```

### Get Latest Matches

**GET** `/latest_matches`

Retrieves recent match data.

Query Parameters:
- `player` (optional): Filter by player name
- `limit` (optional): Number of matches to return (default: 5)

```bash
curl "http://localhost:8000/latest_matches?player=Roger%20Federer&limit=10"
```

Response:
```json
{
  "matches": [
    {
      "date": "2023-07-15",
      "player1": "Roger Federer",
      "player2": "Rafael Nadal",
      "winner": "Rafael Nadal",
      "tournament": "Wimbledon",
      "score": "6-4, 6-3, 7-6"
    }
  ],
  "count": 10
}
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Python Client Example

```python
import requests

# Base URL
base_url = "http://localhost:8000"

# Predict winner
response = requests.post(
    f"{base_url}/predict_winner",
    json={
        "player1": "Novak Djokovic",
        "player2": "Rafael Nadal",
        "tournament": "Australian Open"
    }
)
result = response.json()
print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.2%}")

# Get latest matches
response = requests.get(
    f"{base_url}/latest_matches",
    params={"player": "Novak Djokovic", "limit": 5}
)
matches = response.json()
print(f"Found {matches['count']} matches")
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding:

- API key authentication
- OAuth2 integration
- Rate limiting

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

Example error response:
```json
{
  "detail": "Player not found in database"
}
```

## Rate Limiting

In production, implement rate limiting to prevent abuse:

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/predict_winner", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def predict_winner(...):
    ...
```
