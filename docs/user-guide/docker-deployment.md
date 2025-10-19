# Docker Deployment

Deploy the Tennis Match Predictor using Docker for a consistent, production-ready environment.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+

## Quick Start

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

The Docker setup includes three services:

### 1. API Service

- **Port**: 8000
- **Image**: FastAPI with Uvicorn
- **Health Check**: http://localhost:8000/

### 2. Streamlit App

- **Port**: 8501
- **Image**: Streamlit server
- **Access**: http://localhost:8501

### 3. Development Environment

- **Purpose**: Development and testing
- **Includes**: All dev dependencies

## Docker Compose Configuration

The `docker-compose.yml` file defines the services:

```yaml
services:
  api:
    build:
      context: .
      target: api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  streamlit:
    build:
      context: .
      target: streamlit
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
```

## Dockerfile

The multi-stage Dockerfile optimizes image size and build time:

### Stage 1: Base

- Python 3.13
- uv package manager
- System dependencies

### Stage 2: API

- FastAPI application
- Uvicorn server
- Minimal runtime dependencies

### Stage 3: Streamlit

- Streamlit application
- Connects to API service

## Building Images

Build specific services:

```bash
# Build API only
docker-compose build api

# Build Streamlit only
docker-compose build streamlit

# Build all
docker-compose build
```

## Environment Variables

Configure services using environment variables:

```bash
# Create .env file
cat > .env << EOF
GITHUB_REPO=JeffSackmann/tennis_atp
MIN_ACCURACY=0.60
MLFLOW_TRACKING_URI=http://mlflow:5000
EOF

# Docker Compose will load .env automatically
docker-compose up
```

## Volume Mounts

Persist data using volumes:

```yaml
volumes:
  - ./models:/app/models  # Model files
  - ./data:/app/data      # Data files
  - ./mlruns:/app/mlruns  # MLflow artifacts
```

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml tennis-predictor

# Check services
docker service ls

# Remove stack
docker stack rm tennis-predictor
```

### Using Kubernetes

Convert docker-compose to Kubernetes manifests:

```bash
# Install kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-linux-amd64 -o kompose
chmod +x kompose

# Convert
./kompose convert

# Deploy
kubectl apply -f .
```

## Health Checks

Services include health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose logs api
docker-compose logs streamlit
```

### Port Conflicts

Change ports in docker-compose.yml:
```yaml
ports:
  - "8080:8000"  # API on port 8080
  - "8502:8501"  # Streamlit on port 8502
```

### Performance Issues

Increase container resources:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

## CI/CD Integration

Automated builds in GitHub Actions:

```yaml
- name: Build Docker image
  run: docker build -t tennis-predictor:latest .

- name: Push to registry
  run: |
    docker tag tennis-predictor:latest ghcr.io/user/tennis-predictor:latest
    docker push ghcr.io/user/tennis-predictor:latest
```
