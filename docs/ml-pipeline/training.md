# Model Training

Train machine learning models using the Prefect-orchestrated training pipeline.

## Quick Start

```bash
# Run training pipeline
uv run train-model

# Run with hyperparameter tuning
uv run train-model --tune
```

## Training Pipeline

The training pipeline is implemented as a Prefect flow with the following tasks:

1. **Load Data**: Fetch match data from source
2. **Save Data**: Persist data for inference
3. **Prepare ML Data**: Transform to ML-ready format
4. **Train Model**: Train XGBoost classifier
5. **Evaluate and Promote**: Compare with champion model

### Pipeline Flow

```python
from match_predictor.ml_pipeline.flows import training_flow

# Run the flow
result = training_flow(
    tune_hyperparameters=True,
    data_config_path="config/data_config.yaml",
    model_config_path="config/model_config.yaml"
)

print(f"Model promoted: {result['promoted']}")
print(f"Test accuracy: {result['metrics']['test_accuracy']}")
```

## Model Trainer

The `ModelTrainer` class handles model training:

```python
from match_predictor.ml_pipeline.training import ModelTrainer
from match_predictor.config import ModelConfig

config = ModelConfig.from_yaml("config/model_config.yaml")
trainer = ModelTrainer(config)

# Train model
metrics = trainer.train(
    ml_data,
    tune_hyperparameters=False,
    log_to_mlflow=True
)
```

## Configuration

Configure training in `config/model_config.yaml`:

```yaml
model_name: atp_match_predictor

estimator:
  module: xgboost
  class_name: XGBClassifier
  params:
    objective: binary:logistic
    eval_metric: logloss
    n_estimators: 1000
    max_depth: 3
    learning_rate: 0.01

training:
  test_size: 0.2
  validation_size: 0.2
  random_state: 42
  min_accuracy_threshold: 0.60

mlflow:
  experiment_name: "tennis-match-predictor"
  tracking_uri: "file:./mlruns"
```

## Data Splitting

Data is split into three sets:

```python
# Training: 64% (80% of 80%)
# Validation: 16% (20% of 80%)
# Test: 20%

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

## Model Selection

The default model is XGBoost, but you can configure other estimators:

```yaml
estimator:
  module: sklearn.ensemble
  class_name: RandomForestClassifier
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

## Champion Model Promotion

Models are promoted based on performance:

```python
def should_promote(new_accuracy, champion_accuracy, threshold=0.60):
    if champion_accuracy is None:
        return new_accuracy >= threshold
    return new_accuracy > champion_accuracy
```

Promotion criteria:
- New model accuracy ≥ 60%
- New model better than current champion
- Validation metrics stable

## MLflow Integration

Track experiments with MLflow:

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(model_config.estimator.params)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

View experiments:
```bash
uv run mlflow ui
```

## Model Saving

Models are saved with metadata:

```python
import pickle

model_data = {
    "model": trained_model,
    "feature_names": feature_names,
    "metrics": {
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall
    },
    "config": model_config.dict(),
    "timestamp": datetime.now()
}

with open("models/champion_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
```

## Evaluation Metrics

Models are evaluated on multiple metrics:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve

## Best Practices

1. **Use Cross-Validation**: Validate hyperparameters
2. **Monitor Overfitting**: Compare train vs. validation metrics
3. **Log Everything**: Use MLflow for experiment tracking
4. **Version Models**: Tag models with version numbers
5. **Test Thoroughly**: Evaluate on held-out test set
6. **Retrain Regularly**: Keep model fresh with new data
