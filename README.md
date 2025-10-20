# 🎾 Tennis Match Predictor

[![CI](https://img.shields.io/github/actions/workflow/status/FedericoRaimondi/tennis-match-predictor/ci.yml?branch=main&label=CI)](https://github.com/FedericoRaimondi/tennis-match-predictor/actions)
[![codecov](https://codecov.io/gh/FedericoRaimondi/tennis-match-predictor/graph/badge.svg?token=ZgiMd9iSyu)](https://codecov.io/gh/FedericoRaimondi/tennis-match-predictor)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/github/actions/workflow/status/FedericoRaimondi/tennis-match-predictor/docs.yml?branch=main&label=Documentation)](https://github.com/FedericoRaimondi/tennis-match-predictor/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete end-to-end machine learning system for predicting tennis match outcomes, featuring a FastAPI backend, Streamlit frontend, and automated MLOps pipelines.

<div align="center">
    <img src="https://images.unsplash.com/photo-1531315396756-905d68d21b56?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Tennis Match" width="600">
</div>

## ✨ Features

### 🎯 Match Prediction
- **ML-Powered**: XGBoost classifier trained on historical ATP match data
- **Rich Features**: Player stats, ELO ratings, head-to-head records, surface preferences
- **Real-time API**: FastAPI service for instant predictions
- **Interactive UI**: Streamlit app with visualizations and insights

### 📊 MLOps Pipeline
- **Workflow Orchestration**: Prefect flows for training and monitoring
- **Automated Training**: Hyperparameter tuning with Optuna
- **Model Monitoring**: Data drift detection with Evidently
- **Version Control**: MLflow for experiment tracking
- **Champion Model**: Automatic model promotion based on performance

### 🚀 Production Ready
- **Docker Support**: Multi-stage builds for API and Streamlit
- **CI/CD**: GitHub Actions for testing and deployment
- **Configuration Management**: Pydantic-based configs
- **Documentation**: MkDocs with auto-generated API reference

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                   (Streamlit Dashboard)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         REST API                                │
│                      (FastAPI Service)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌───────────────────────┐   ┌───────────────────────┐
│   ML Model            │   │   Data Processing     │
│   (XGBoost)           │   │ (Feature Engineering) │
└───────────────────────┘   └───────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                             │
│                    (Orchestrated by Prefect)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Training │→ │  Tuning  │→ │Evaluation│→ │Deployment│           │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │   Drift  │→ │Monitoring│→ │ Alerting │                         │
│  │Detection │  │          │  │          │                         │
│  └──────────┘  └──────────┘  └──────────┘                         │
└───────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
.
├── src/
│   └── match_predictor/
│       ├── api/                 # FastAPI service
│       │   ├── __init__.py
│       │   └── main.py         # API endpoints
│       ├── app/                 # Streamlit app
│       │   ├── __init__.py
│       │   └── main.py         # UI components
│       ├── ml_pipeline/         # Training & monitoring
│       │   ├── feature_engineering.py
│       │   ├── training.py
│       │   ├── hyperparameter_tuning.py
│       │   ├── monitoring.py
│       │   └── flows.py        # Prefect workflows
│       ├── data/                # Data loading utilities
│       │   └── data_loader.py
│       ├── model/               # Model classes
│       │   ├── base_model.py
│       │   └── estimator_model.py
│       ├── utils/               # Helper functions
│       │   ├── gh_utils.py
│       │   └── stats_helpers.py
│       └── config.py            # Pydantic configuration classes
│
├── config/
│   ├── data_config.yaml         # Data configuration (YAML)
│   └── model_config.yaml        # Model configuration (YAML)
│
├── tests/                       # Test suite (pytest)
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_flows.py
│   ├── test_gh_utils.py
│   └── test_stats_helpers.py
│
├── data/                        # Inference data only
│   ├── matches_results.pkl
│   └── tournament_info.pkl
│
├── models/                      # Champion model storage
│   └── .gitkeep
│
├── notebook/                    # Jupyter notebooks for exploration
│   ├── tennis_eda.ipynb        # Exploratory data analysis
│   ├── data_engineering.ipynb  # Feature engineering experiments
│   └── ml_experimenting.ipynb  # Model experimentation
│
├── docs/                        # MkDocs documentation
│   ├── index.md
│   └── ...
│
├── .github/
│   └── workflows/               # CI/CD pipelines (paused)
│       ├── ci.yml              # Tests and linting
│       ├── cd.yml              # Deployment
│       ├── training.yml        # Model training
│       ├── monitoring.yml      # Model monitoring
│       └── docs.yml            # Documentation deployment
│
├── Dockerfile                   # Multi-stage Docker setup
├── docker-compose.yml           # Multi-service orchestration
├── mkdocs.yml                   # Documentation config
├── pyproject.toml               # Project dependencies (uv)
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/FedericoRaimondi/tennis-match-predictor.git
cd tennis-match-predictor
```

2. **Install dependencies**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

3. **Run the application**

#### Option A: Local Development
```bash
# Terminal 1 - Start the API
uv run uvicorn match_predictor.api.main:app --reload --port 8000

# Terminal 2 - Start the Streamlit app
uv run streamlit run src/match_predictor/app/main.py
```

#### Option B: Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

4. **Access the application**
- 🌐 Streamlit App: http://localhost:8501
- 📡 API Documentation: http://localhost:8000/docs
- 🔍 API Health: http://localhost:8000/

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/match_predictor --cov-report=html

# Run specific test file
uv run pytest tests/test_api.py -v
```

### Code Quality

```bash
# Lint code
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Type checking (if mypy is added)
uv run mypy src/
```

## 📚 Documentation

Full documentation is available via MkDocs:

```bash
# Serve documentation locally
uv run mkdocs serve

# Build static documentation
uv run mkdocs build

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

Visit the [documentation site](https://FedericoRaimondi.github.io/tennis-match-predictor/) for:
- Installation guide
- API reference
- ML pipeline details
- Configuration options
- Contributing guidelines

## 🔧 Configuration

The project uses Pydantic for configuration management with YAML files. Pydantic classes are in `src/match_predictor/config.py`, and configuration values are stored in YAML files in the `config/` directory.

### Data Configuration (`config/data_config.yaml`)
```yaml
source:
  github_repo: "JeffSackmann/tennis_atp"
  selected_year: 1991
  tourney_levels: ["G", "F", "M", "A"]

features:
  rolling_windows: [3, 5, 10]
  elo_k_factor: 32.0
  elo_initial_rating: 1500.0
```

### Model Configuration (`config/model_config.yaml`)
```yaml
model_name: atp_match_predictor

estimator:
  module: xgboost
  class_name: XGBClassifier
  params:
    max_depth: 3
    learning_rate: 0.01
    n_estimators: 1000

training:
  min_accuracy_threshold: 0.60
  test_size: 0.2
```

### Loading Configuration in Python
```python
from match_predictor.config import DataConfig, ModelConfig

# Load from YAML files
data_config = DataConfig.from_yaml("config/data_config.yaml")
model_config = ModelConfig.from_yaml("config/model_config.yaml")

# Or use defaults
data_config = DataConfig()
model_config = ModelConfig()
```

## 🤖 ML Pipeline

### Training Pipeline

The training pipeline includes:
1. **Feature Engineering**: Create derived features from raw match data
2. **Hyperparameter Tuning**: Optimize model parameters using Optuna
3. **Model Training**: Train XGBoost classifier
4. **Model Evaluation**: Validate on test set
5. **Model Deployment**: Promote to champion if accuracy > threshold

Run using Prefect flow (recommended):
```bash
# Using the entrypoint script
uv run train-model

# Or run the flow directly
uv run python -c "
from match_predictor.ml_pipeline.flows import run_training_flow
run_training_flow()
"
```

Or run manually without Prefect:
```bash
uv run python -c "
from match_predictor.ml_pipeline.training import ModelTrainer
from match_predictor.config import ModelConfig

trainer = ModelTrainer(ModelConfig.from_yaml('config/model_config.yaml'))
metrics = trainer.train(df, tune_hyperparameters=True)
print(metrics)
"
```

### Monitoring Pipeline

The monitoring pipeline detects:
- **Data Drift**: Statistical changes in feature distributions
- **Model Performance**: Accuracy degradation over time
- **Data Quality**: Missing values, outliers, anomalies

Run using Prefect flow (recommended):
```bash
# Using the entrypoint script
uv run monitor-model

# Or run the flow directly
uv run python -c "
from match_predictor.ml_pipeline.flows import run_monitoring_flow
run_monitoring_flow()
"
```

Automatically triggers retraining when:
- Drift share > 30%
- Accuracy < 60%
- Manual trigger via GitHub Actions

## 🔄 CI/CD Workflows

All GitHub Actions workflows are **paused by default** and can be enabled by uncommenting the trigger sections:

### CI Workflow (`.github/workflows/ci.yml`)
- Runs on: Pull requests to `main`
- Tests: Unit and integration tests
- Linting: Ruff code quality checks
- Coverage: Code coverage reports

### CD Workflow (`.github/workflows/cd.yml`)
- Runs on: Push to `main` or version tags
- Deploys API to Google Cloud Run
- Deploys Streamlit to Streamlit Community Cloud
- Pushes Docker images to registry

### Training Workflow (`.github/workflows/training.yml`)
- Runs on: Monthly schedule or manual trigger
- Checks for new data
- Trains model with hyperparameter tuning
- Promotes to champion if accuracy improves

### Monitoring Workflow (`.github/workflows/monitoring.yml`)
- Runs on: Monthly schedule or manual trigger
- Detects data and model drift
- Generates monitoring reports
- Triggers retraining if needed

### Documentation Workflow (`.github/workflows/docs.yml`)
- Runs on: Push to `main` (docs changes) or manual trigger
- Builds documentation with MkDocs
- Deploys to GitHub Pages
- Auto-generates API reference

## 🐋 Docker

### Build Individual Services

```bash
# Build API
docker build --target api -t tennis-predictor-api .

# Build Streamlit app
docker build --target streamlit -t tennis-predictor-streamlit .

# Build dev environment
docker build --target dev -t tennis-predictor-dev .
```

### Docker Compose Services

```yaml
services:
  api:       # FastAPI service (port 8000)
  streamlit: # Streamlit app (port 8501)
  dev:       # Development environment
```

## 📊 API Endpoints

### Prediction
```bash
curl -X POST "http://localhost:8000/predict_winner" \
  -H "Content-Type: application/json" \
  -d '{
    "player1": "Novak Djokovic",
    "player2": "Rafael Nadal",
    "tournament": "Wimbledon"
  }'
```

### Latest Matches
```bash
curl "http://localhost:8000/latest_matches?player=Roger Federer&limit=5"
```

## 🧪 Testing

The project includes comprehensive tests:
- **Unit Tests**: Individual component testing
- **Integration Tests**: API and pipeline testing  
- **Feature Tests**: Feature engineering validation
- **Config Tests**: Configuration validation
- **Flow Tests**: Prefect workflow testing

**Test Coverage**: 181 tests passing with 79% code coverage across all modules.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/match_predictor --cov-report=html --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_api.py -v

# View coverage report
open htmlcov/index.html  # Opens HTML coverage report in browser
```

### Coverage by Module

| Module | Coverage | Status |
|--------|----------|--------|
| training.py | 100% | ✅ Perfect |
| hyperparameter_tuning.py | 100% | ✅ Perfect |
| gh_utils.py | 100% | ✅ Perfect |
| data_loader.py | 100% | ✅ Perfect |
| stats_helpers.py | 100% | ✅ Perfect |
| config.py | 98% | ✅ Excellent |
| feature_engineering.py | 95% | ✅ Good |
| base_model.py | 83% | ⚠️ Good |
| estimator_model.py | 78% | ⚠️ Needs work |
| monitoring.py | 66% | ⚠️ Needs tests |
| flows.py | 46% | ⚠️ Needs tests |
| **Overall** | **79%** | ⚠️ Target: ≥90% |

## 📝 Data Source

This project uses data from [Jeff Sackmann / Tennis Abstract](http://www.tennisabstract.com/).

> ⚠️ **License**: The tennis data is provided under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Please respect the terms of use.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Jeff Sackmann](https://github.com/JeffSackmann) for the comprehensive tennis data
- [Tennis Abstract](http://www.tennisabstract.com/) for statistical insights
- All contributors and maintainers

## 📬 Contact

Federico Raimondi Cominesi - [@FedericoRaimondi](https://github.com/FedericoRaimondi)

Project Link: [https://github.com/FedericoRaimondi/tennis-match-predictor](https://github.com/FedericoRaimondi/tennis-match-predictor)

---

**Made with ❤️ and 🎾**
