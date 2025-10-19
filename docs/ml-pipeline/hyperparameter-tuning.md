# Hyperparameter Tuning

Optimize model performance using Optuna-based Bayesian optimization.

## Quick Start

```bash
# Run training with hyperparameter tuning
uv run train-model --tune
```

## Overview

Hyperparameter tuning uses Optuna to find the best model parameters through Bayesian optimization with cross-validation.

## Configuration

Configure tuning in `config/model_config.yaml`:

```yaml
hyperparameter_tuning:
  n_trials: 50
  cv_folds: 5
  random_state: 42
  timeout: 3600  # 1 hour
```

## Hyperparameter Search Space

For XGBoost:

```python
def suggest_hyperparameters(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
```

## Tuning Process

The `HyperparameterTuner` class manages optimization:

```python
from match_predictor.ml_pipeline.hyperparameter_tuning import HyperparameterTuner
from match_predictor.config import ModelConfig

config = ModelConfig.from_yaml("config/model_config.yaml")
tuner = HyperparameterTuner(config)

# Tune hyperparameters
best_params = tuner.tune(X_train, y_train)

print("Best parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
```

## Optimization Objective

Maximize cross-validation accuracy:

```python
def objective(trial):
    # Suggest hyperparameters
    params = suggest_hyperparameters(trial)
    
    # Create model
    model = XGBClassifier(**params)
    
    # Cross-validation
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='accuracy'
    )
    
    # Return mean accuracy
    return scores.mean()
```

## Running Tuning

### Command Line

```bash
# With default settings
uv run train-model --tune

# View progress in Optuna dashboard
uv run optuna-dashboard sqlite:///optuna_study.db
```

### Python API

```python
from match_predictor.ml_pipeline.flows import training_flow

result = training_flow(
    tune_hyperparameters=True,
    data_config_path="config/data_config.yaml",
    model_config_path="config/model_config.yaml"
)
```

## Monitoring Progress

Optuna provides real-time optimization progress:

```
[I 2024-01-15 10:30:45,123] Trial 0 finished with value: 0.6234 and parameters: {'max_depth': 5, ...}
[I 2024-01-15 10:31:12,456] Trial 1 finished with value: 0.6412 and parameters: {'max_depth': 3, ...}
Best value: 0.6412, Best params: {'max_depth': 3, 'learning_rate': 0.05, ...}
```

## Visualization

Visualize optimization results:

```python
import optuna

# Load study
study = optuna.load_study(
    study_name="tennis-predictor-tuning",
    storage="sqlite:///optuna_study.db"
)

# Plot optimization history
fig1 = optuna.visualization.plot_optimization_history(study)
fig1.show()

# Plot parameter importances
fig2 = optuna.visualization.plot_param_importances(study)
fig2.show()

# Plot parallel coordinate plot
fig3 = optuna.visualization.plot_parallel_coordinate(study)
fig3.show()
```

## Advanced Tuning

### Pruning

Enable early stopping for poor trials:

```python
import optuna.pruners as pruners

study = optuna.create_study(
    direction="maximize",
    pruner=pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)
```

### Multi-Objective Optimization

Optimize multiple metrics:

```python
def multi_objective(trial):
    params = suggest_hyperparameters(trial)
    model = XGBClassifier(**params)
    
    # Cross-validation
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=5, scoring='precision').mean()
    
    return accuracy, precision

study = optuna.create_study(directions=["maximize", "maximize"])
study.optimize(multi_objective, n_trials=100)
```

### Custom Sampler

Use different sampling strategies:

```python
import optuna.samplers as samplers

# TPE sampler (default)
sampler = samplers.TPESampler(seed=42)

# Grid search
sampler = samplers.GridSampler({
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1]
})

# Random search
sampler = samplers.RandomSampler(seed=42)

study = optuna.create_study(sampler=sampler)
```

## Best Practices

1. **Start Small**: Begin with fewer trials for quick feedback
2. **Use Cross-Validation**: Ensure generalization
3. **Set Timeouts**: Prevent excessively long tuning
4. **Monitor Overfitting**: Check train vs. validation metrics
5. **Save Studies**: Persist results for analysis
6. **Parallelize**: Run trials in parallel when possible

## Troubleshooting

### Memory Issues

Reduce cross-validation folds or dataset size:

```yaml
hyperparameter_tuning:
  cv_folds: 3  # Reduce from 5
```

### Slow Tuning

- Reduce `n_trials`
- Increase `timeout`
- Use pruning to stop poor trials early
- Reduce dataset size for initial tuning

### Poor Results

- Expand search space
- Increase number of trials
- Check data quality
- Verify feature engineering
