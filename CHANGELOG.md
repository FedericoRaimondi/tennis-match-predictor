# Changelog

All notable changes to the Tennis Match Predictor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete end-to-end ML prototype with Streamlit, FastAPI, Docker, and MLOps
- FastAPI service under `src/match_predictor/api/` with prediction and match data endpoints
- Streamlit app under `src/match_predictor/app/` with interactive UI for predictions
- ML pipeline modules:
  - Feature engineering with derived statistics
  - Hyperparameter tuning using Optuna
  - Model training with MLflow tracking
  - Data and model drift monitoring with Evidently
- Pydantic-based configuration management:
  - `config/data_config.py` for data source and feature configuration
  - `config/model_config.py` for model and training configuration
- Docker support:
  - Multi-stage Dockerfile for API, Streamlit, and development
  - `docker-compose.yml` for orchestrating all services
- GitHub Actions workflows (paused by default):
  - CI workflow for tests and linting
  - CD workflow for deployment to Cloud Run and Streamlit Cloud
  - Training workflow for automated model retraining
  - Monitoring workflow for drift detection and alerting
- Comprehensive test suite (49+ tests):
  - API endpoint tests
  - Configuration validation tests
  - Feature engineering tests
  - Data loader tests
  - Utilities tests
- MkDocs documentation:
  - Project overview and architecture
  - Installation and quick start guides
  - API reference with auto-generated docs
  - ML pipeline documentation
- Utility scripts:
  - `scripts/run_api.sh` for starting the API
  - `scripts/run_streamlit.sh` for starting the Streamlit app
  - `scripts/run_tests.sh` for running tests with coverage
  - `scripts/build_docs.sh` for building/serving documentation
- Additional dependencies:
  - streamlit for interactive UI
  - pydantic and pydantic-settings for configuration
  - evidently for drift monitoring
  - optuna for hyperparameter tuning
  - plotly for visualizations
  - httpx for async HTTP requests
  - mkdocs and mkdocs-material for documentation
- `.dockerignore` for optimized Docker builds
- Comprehensive README with badges, architecture diagrams, and usage examples
- CONTRIBUTING.md with development guidelines

### Changed
- Renamed package from `tennis_match_predictor` to `match_predictor` as per requirements
- Renamed `test/` directory to `tests/` following standard conventions
- Updated project name in `pyproject.toml` to `match-predictor`
- Updated all import statements to use new package name
- Enhanced `.gitignore` to exclude monitoring reports and temporary model files

### Fixed
- API response type hint for `/info` endpoint to handle boolean `model_loaded` field

## [0.1.0] - Initial Release

### Added
- Initial project structure with data loading utilities
- Base model classes (BaseModel, EstimatorModel)
- Data loader for ATP match data from GitHub
- Utility functions for GitHub API and statistics helpers
- Initial test suite for data loading and utilities
- Basic configuration with `pyproject.toml` and `uv` package manager
- XGBoost model configuration in YAML format

[Unreleased]: https://github.com/FedericoRaimondi/tennis-match-predictor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FedericoRaimondi/tennis-match-predictor/releases/tag/v0.1.0
