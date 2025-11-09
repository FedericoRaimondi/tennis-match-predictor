# Multi-stage Dockerfile for Tennis Match Predictor

# Base stage with Python and uv
FROM python:3.13-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Copy source code (needed for package installation)
COPY src/ ./src/

RUN uv sync --frozen --no-dev

# Copy remaining files
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

# API stage
FROM base AS api

EXPOSE 8000

# Run the FastAPI application
CMD ["uv", "run", "uvicorn", "match_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit stage
FROM base AS streamlit

EXPOSE 8501

# Run the Streamlit application
CMD ["uv", "run", "streamlit", "run", "src/match_predictor/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage with dev dependencies
FROM base AS dev

# Install dev dependencies
RUN uv sync --frozen

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Default to bash for development
CMD ["/bin/bash"]
