# Quick Start

This guide will help you get started with the Tennis Match Predictor quickly.

## Prerequisites

- Python 3.13+
- uv package manager
- Git

## Installation

```bash
# Clone the repository
git clone https://github.com/FedericoRaimondi/tennis-match-predictor.git
cd tennis-match-predictor

# Install dependencies with uv
uv sync
```

## Running the Application

### Option 1: Local Development

```bash
# Run the API
uv run uvicorn match_predictor.api.main:app --reload

# In another terminal, run the Streamlit app
uv run streamlit run src/match_predictor/app/main.py
```

### Option 2: Docker

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

## Accessing the Services

Once running, you can access:

- **Streamlit App**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=match_predictor --cov-report=html

# Run specific test file
uv run pytest tests/test_api.py
```

## Next Steps

- Check out the [Configuration Guide](configuration.md) to customize the application
- Learn about the [Streamlit App](../user-guide/streamlit-app.md) features
- Explore the [API Usage](../user-guide/api-usage.md) documentation
- Understand the [ML Pipeline](../ml-pipeline/training.md)
