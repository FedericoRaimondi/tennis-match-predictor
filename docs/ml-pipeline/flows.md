# Prefect Flows

The Tennis Match Predictor uses [Prefect](https://www.prefect.io/) to orchestrate ML pipeline workflows. Flows automate the complete lifecycle from data loading to model training, evaluation, and deployment.

## Overview

The project includes two main flows:

1. **Training Flow** - Loads data, trains models, and promotes champions
2. **Monitoring Flow** - Detects data drift and triggers retraining when needed

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Training Flow                          │
├─────────────────────────────────────────────────────────┤
│  1. Load Data (load_data_task)                          │
│     ↓                                                    │
│  2. Save Data (save_data_task)                          │
│     ↓                                                    │
│  3. Prepare ML Data (prepare_ml_data_task)              │
│     ↓                                                    │
│  4. Train Model (train_model_task)                      │
│     ↓                                                    │
│  5. Evaluate & Promote (evaluate_and_promote_task)      │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Monitoring Flow                         │
├─────────────────────────────────────────────────────────┤
│  1. Check New Data (check_new_data_task)                │
│     ↓                                                    │
│  2. Detect Drift (detect_drift_task)                    │
│     ↓                                                    │
│  3. Evaluate Model (evaluate_model_task)                │
│     ↓                                                    │
│  4. Trigger Training (if needed)                        │
└─────────────────────────────────────────────────────────┘
```

## Training Flow

The training flow orchestrates the complete model training pipeline.

### Tasks

#### 1. Load Data Task

```python
@task(name="load_data", retries=2, retry_delay_seconds=60)
def load_data_task(data_config: DataConfig) -> pd.DataFrame:
    """Load match data from source."""
```

- Loads ATP match data from GitHub
- Applies basic cleaning and filtering
- Includes retry logic for network failures

#### 2. Save Data Task

```python
@task(name="save_data")
def save_data_task(matches: pd.DataFrame, data_config: DataConfig) -> None:
    """Save processed data for inference."""
```

- Saves match results to CSV
- Extracts and saves tournament information
- Saves latest player statistics for API inference

#### 3. Prepare ML Data Task

```python
@task(name="prepare_ml_data")
def prepare_ml_data_task(data_config: DataConfig, matches: pd.DataFrame) -> pd.DataFrame:
    """Prepare ML dataset from matches."""
```

- Creates player matchups with random assignment
- Adds rolling statistics and ELO ratings
- Generates target labels (winner)

#### 4. Train Model Task

```python
@task(name="train_model")
def train_model_task(
    ml_data: pd.DataFrame, 
    model_config: ModelConfig, 
    tune_hyperparameters: bool = False
) -> dict:
    """Train the model."""
```

- Trains XGBoost classifier
- Optionally performs hyperparameter tuning with Optuna
- Logs metrics to MLflow
- Saves model to disk

#### 5. Evaluate and Promote Task

```python
@task(name="evaluate_and_promote")
def evaluate_and_promote_task(metrics: dict, model_config: ModelConfig) -> bool:
    """Evaluate if the new model should be promoted to champion."""
```

- Compares test accuracy against threshold
- Promotes model to champion if accuracy exceeds threshold
- Copies model file to champion location

### Running the Training Flow

#### Via Python

```python
from match_predictor.ml_pipeline.flows import training_flow

# Run with default configs
result = training_flow(tune_hyperparameters=False)

# Run with custom configs
result = training_flow(
    tune_hyperparameters=True,
    data_config_path="config/data_config.yaml",
    model_config_path="config/model_config.yaml"
)
```

#### Via Command Line

```bash
# Using the entrypoint script
uv run train-model

# Or run directly with Python
uv run python -c "
from match_predictor.ml_pipeline.flows import run_training_flow
run_training_flow()
"
```

#### Via Prefect CLI

```bash
# Deploy the flow
prefect deploy --name training-pipeline

# Run the deployment
prefect deployment run 'Training Pipeline/training-pipeline'
```

## Monitoring Flow

The monitoring flow checks for data drift and model performance degradation.

### Tasks

#### 1. Check New Data Task

```python
@task(name="check_new_data")
def check_new_data_task(data_config: DataConfig) -> bool:
    """Check if new data is available."""
```

- Compares current data count with saved data
- Returns `True` if new matches are available

#### 2. Detect Drift Task

```python
@task(name="detect_drift")
def detect_drift_task(data_config: DataConfig) -> dict:
    """Detect data drift using Evidently."""
```

- Uses Evidently AI to detect distribution shifts
- Compares reference data with current data
- Returns drift metrics and visualizations

#### 3. Evaluate Model Task

```python
@task(name="evaluate_model")
def evaluate_model_task(data_config: DataConfig, model_config: ModelConfig) -> dict:
    """Evaluate current model performance."""
```

- Tests champion model on recent data
- Calculates accuracy metrics
- Returns performance scores

### Running the Monitoring Flow

#### Via Python

```python
from match_predictor.ml_pipeline.flows import monitoring_flow

# Run monitoring
result = monitoring_flow(
    data_config_path="config/data_config.yaml",
    model_config_path="config/model_config.yaml"
)
```

#### Via Command Line

```bash
# Using the entrypoint script
uv run monitor-model

# Or run directly
uv run python -c "
from match_predictor.ml_pipeline.flows import run_monitoring_flow
run_monitoring_flow()
"
```

## Configuration

Flows use YAML configuration files for flexibility.

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

inference_data_path: "data/"
matches_results_file: "matches_results.csv"
tournament_info_file: "tournament_info.csv"
player_stats_file: "player_stats_latest.csv"
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
  test_size: 0.2
  min_accuracy_threshold: 0.60

champion_model_path: "models/"
```

## Task Retries and Error Handling

### Retry Logic

Tasks with network dependencies include automatic retries:

```python
@task(name="load_data", retries=2, retry_delay_seconds=60)
def load_data_task(data_config: DataConfig) -> pd.DataFrame:
    # Will retry up to 2 times with 60 second delay
    ...
```

### Error Notifications

Failed tasks log errors with context:

```python
try:
    matches = loader.load_matches()
except Exception as e:
    logger.error(f"Failed to load matches: {e}")
    raise
```

## Scheduling

### GitHub Actions Integration

Flows can be triggered via GitHub Actions workflows:

```yaml
# .github/workflows/training.yml
name: Monthly Training
on:
  schedule:
    - cron: '0 0 1 * *'  # First day of each month
  workflow_dispatch:  # Manual trigger

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run training
        run: uv run train-model
```

### Prefect Cloud Scheduling

Deploy flows to Prefect Cloud for advanced scheduling:

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=training_flow,
    name="monthly-training",
    schedule=CronSchedule(cron="0 0 1 * *")
)
deployment.apply()
```

## Monitoring and Observability

### MLflow Integration

Training tasks log to MLflow automatically:

- Model parameters
- Training metrics
- Model artifacts
- Feature importance

### Logging

All tasks use structured logging:

```python
logger.info(f"Loaded {len(matches)} matches")
logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
```

### Prefect UI

View flow runs in the Prefect UI:

```bash
prefect server start
# Visit http://localhost:4200
```

## Best Practices

### 1. Use Configuration Files

Keep data and model configs separate from code:

```python
# Good
training_flow(data_config_path="config/data_config.yaml")

# Avoid
training_flow(data_config=DataConfig(source=...))
```

### 2. Handle Task Failures Gracefully

Implement proper error handling:

```python
@task
def my_task():
    try:
        # Task logic
        ...
    except SpecificError as e:
        logger.error(f"Known error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 3. Use Retries for Network Operations

Add retries to tasks that depend on external services:

```python
@task(retries=3, retry_delay_seconds=30)
def fetch_external_data():
    ...
```

### 4. Log Progress

Include informative logging at each step:

```python
logger.info(f"Starting task with {len(data)} samples")
result = process(data)
logger.info(f"Task complete. Processed {len(result)} items")
```

## Troubleshooting

### Common Issues

#### Task Timeout

```python
# Increase task timeout
@task(timeout_seconds=3600)  # 1 hour
def long_running_task():
    ...
```

#### Memory Issues

```python
# Process data in chunks
for chunk in pd.read_csv(file, chunksize=10000):
    process_chunk(chunk)
```

#### Failed Retries

Check logs for the root cause:

```bash
prefect deployment run 'Training Pipeline/training' --watch
```

## See Also

- [Training Documentation](training.md)
- [Monitoring Documentation](monitoring.md)
- [Hyperparameter Tuning](hyperparameter-tuning.md)
- [Prefect Documentation](https://docs.prefect.io/)
