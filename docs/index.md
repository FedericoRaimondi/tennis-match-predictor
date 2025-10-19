# Tennis Match Predictor

Welcome to the Tennis Match Predictor documentation!

## Overview

The Tennis Match Predictor is a complete end-to-end machine learning system for predicting tennis match outcomes. It includes:

- **Machine Learning Model**: XGBoost-based classifier trained on historical ATP match data
- **FastAPI Service**: RESTful API for serving predictions
- **Streamlit App**: Interactive web interface for users
- **ML Pipeline**: Automated training, hyperparameter tuning, and monitoring
- **Docker Support**: Containerized deployment for easy scaling

## Features

### 🎾 Match Prediction
Predict the winner of tennis matches based on:
- Player historical performance
- Head-to-head records
- Tournament and surface statistics
- ELO ratings
- Rolling statistics (last 3, 5, 10 matches)

### 📊 Interactive Dashboard
- Select players and tournaments
- View win probabilities
- Explore head-to-head records
- Analyze recent match statistics

### 🚀 Production-Ready
- FastAPI backend with automatic documentation
- Docker containerization
- MLflow model tracking
- Evidently monitoring for drift detection
- Automated CI/CD pipelines (GitHub Actions)

## Quick Start

### Prerequisites
- Python 3.13+
- uv package manager
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/FedericoRaimondi/tennis-match-predictor.git
cd tennis-match-predictor

# Install dependencies with uv
uv sync

# Run the API
uv run uvicorn match_predictor.api.main:app --reload

# Run the Streamlit app (in another terminal)
uv run streamlit run src/match_predictor/app/main.py
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Access the services:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
```

## Architecture

```
┌─────────────────┐
│  Streamlit App  │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI API   │
│    (Backend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│  (XGBoost)      │
└─────────────────┘
```

## Project Structure

```
.
├── src/
│   └── match_predictor/
│       ├── api/                 # FastAPI service
│       ├── app/                 # Streamlit app
│       ├── ml_pipeline/         # Training & monitoring
│       ├── data/                # Data loading utilities
│       ├── model/               # Model classes
│       └── utils/               # Helper functions
│
├── config/
│   ├── data_config.py           # Data configuration
│   └── model_config.py          # Model configuration
│
├── tests/                       # Test suite
├── data/                        # Inference data
├── models/                      # Champion model
├── docs/                        # Documentation
├── Dockerfile                   # Docker setup
├── docker-compose.yml           # Multi-service setup
└── pyproject.toml               # Project dependencies
```

## Data Source

This project uses data from [Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/).

!!! note "License"
    The tennis data is provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [API Usage](user-guide/api-usage.md)
- [ML Pipeline Overview](ml-pipeline/training.md)
