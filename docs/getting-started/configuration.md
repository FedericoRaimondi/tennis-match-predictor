# Configuration

The Tennis Match Predictor uses Pydantic for configuration management with YAML files.

## Configuration Files

Configuration is split between:

- **Pydantic Classes**: `src/match_predictor/config.py` - Type-safe configuration models
- **YAML Files**: `config/` - Human-readable configuration values

### Data Configuration

Located at `config/data_config.yaml`:

```yaml
source:
  github_repo: "JeffSackmann/tennis_atp"
  selected_year: 1991
  tourney_levels:
    - "G"  # Grand Slam
    - "F"  # Tour Finals
    - "M"  # Masters
    - "A"  # ATP Tour

features:
  rolling_windows: [3, 5, 10]
  elo_k_factor: 32.0
  elo_initial_rating: 1500.0
```

### Model Configuration

Located at `config/model_config.yaml`:

```yaml
model_name: atp_match_predictor

estimator:
  module: xgboost
  class_name: XGBClassifier
  params:
    objective: binary:logistic
    eval_metric: logloss
    max_depth: 3
    learning_rate: 0.01
    n_estimators: 1000

training:
  test_size: 0.2
  validation_size: 0.2
  min_accuracy_threshold: 0.60
```

## Loading Configuration in Python

```python
from match_predictor.config import DataConfig, ModelConfig

# Load from YAML files
data_config = DataConfig.from_yaml("config/data_config.yaml")
model_config = ModelConfig.from_yaml("config/model_config.yaml")

# Or use defaults
data_config = DataConfig()
model_config = ModelConfig()

# Access configuration values
print(data_config.source.github_repo)
print(model_config.training.min_accuracy_threshold)
```

## Environment Variables

You can override configuration values using environment variables:

```bash
export GITHUB_REPO="custom/repo"
export MIN_ACCURACY=0.65
```

## Configuration Validation

Pydantic automatically validates configuration:

```python
from match_predictor.config import ModelConfig

# This will raise a validation error
config = ModelConfig(
    training={"min_accuracy_threshold": 1.5}  # Invalid: > 1.0
)
```

## Advanced Configuration

For more advanced configuration options, see the API reference:

- [DataConfig API Reference](../api-reference/data.md)
- [ModelConfig API Reference](../api-reference/model.md)
