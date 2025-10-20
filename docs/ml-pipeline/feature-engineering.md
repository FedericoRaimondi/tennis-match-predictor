# Feature Engineering

Feature engineering transforms raw match data into meaningful features for machine learning.

## Overview

The `FeatureEngineer` class handles all feature transformations:

```python
from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.engineer_features(ml_data)
```

## Feature Types

### 1. Rolling Statistics

Calculate rolling averages and sums over recent matches:

```python
rolling_windows = [3, 5, 10]  # Last 3, 5, and 10 matches

stats_columns = [
    "p_ace", "p_df", "p_svpt", 
    "p_1stIn", "p_1stWon", "p_2ndWon"
]
```

Features generated:
- `p_ace_mean_3`: Average aces in last 3 matches
- `p_df_sum_5`: Total double faults in last 5 matches
- etc.

### 2. ELO Ratings

Dynamic player ratings that update after each match:

```python
elo_k_factor = 32.0          # Weight of each match result
elo_initial_rating = 1500.0  # Starting rating for new players
```

Features:
- `player_elo`: Current player ELO rating
- `opponent_elo`: Current opponent ELO rating
- `elo_diff`: Difference between player and opponent

### 3. Match Statistics

Direct match-level features:

- Service statistics (aces, double faults, first serve %)
- Return statistics
- Break points saved/faced
- Match duration

### 4. Temporal Features

Time-based features:

- `days_since_last_match`: Rest days
- `matches_in_last_month`: Recent activity level
- `tournament_week`: Week of the tournament

## Feature Engineering Pipeline

```python
from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer
from match_predictor.config import DataConfig

# Initialize
config = DataConfig.from_yaml("config/data_config.yaml")
engineer = FeatureEngineer(config.features)

# Engineer features
ml_data = load_match_data()  # Your data loading function
features = engineer.engineer_features(ml_data)

# Prepare for training
X, y = engineer.prepare_features_for_training(features)
```

## Configuration

Configure feature engineering in `config/data_config.yaml`:

```yaml
features:
  rolling_windows: [3, 5, 10]
  stats_columns_mean:
    - p_ace
    - p_df
    - p_svpt
  stats_columns_sum:
    - minutes
    - results
  elo_k_factor: 32.0
  elo_initial_rating: 1500.0
```

## Feature Selection

Get a list of generated features:

```python
feature_names = engineer.get_feature_names()
print(f"Total features: {len(feature_names)}")
```

## Missing Value Handling

Missing values are handled automatically:

- Forward fill for time-series features
- Median imputation for statistical features
- Zero-fill for counting features

## Custom Features

Add custom features by extending the `FeatureEngineer` class:

```python
class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, df):
        # Add head-to-head win rate
        df['h2h_win_rate'] = self.calculate_h2h_rate(df)
        
        # Add surface-specific statistics
        df['clay_win_rate'] = self.calculate_surface_rate(df, 'clay')
        
        return df
    
    def engineer_features(self, df):
        df = super().engineer_features(df)
        df = self.add_custom_features(df)
        return df
```

## Feature Importance

After training, analyze feature importance:

```python
from match_predictor.ml_pipeline.training import ModelTrainer

trainer = ModelTrainer(model_config)
trainer.train(ml_data)

# Get feature importance
importance = trainer.get_feature_importance()
top_features = importance.head(10)
print(top_features)
```

## Best Practices

1. **Avoid Data Leakage**: Don't use future information
2. **Handle Missing Data**: Ensure robust imputation
3. **Scale Features**: Normalize when necessary
4. **Test Features**: Validate feature distributions
5. **Monitor Features**: Track feature drift over time
