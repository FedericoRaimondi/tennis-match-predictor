# Model Monitoring

Monitor model performance and detect data drift using Evidently.

## Quick Start

```bash
# Run monitoring pipeline
uv run monitor-model
```

## Overview

The monitoring pipeline detects:
- **Data Drift**: Changes in feature distributions
- **Model Performance**: Accuracy degradation
- **Data Quality**: Missing values, outliers

## Monitoring Pipeline

Implemented as a Prefect flow:

```python
from match_predictor.ml_pipeline.flows import monitoring_flow

result = monitoring_flow(
    data_config_path="config/data_config.yaml",
    model_config_path="config/model_config.yaml"
)

if result["should_retrain"]:
    print("⚠️ Retraining recommended")
```

## Data Drift Detection

The `ModelMonitor` class uses Evidently to detect drift:

```python
from match_predictor.ml_pipeline.monitoring import ModelMonitor

monitor = ModelMonitor()

# Set reference data (baseline)
monitor.set_reference_data(reference_df, target_column="winner")

# Set current data (new data)
monitor.set_current_data(current_df)

# Check for drift
drift_results = monitor.check_data_drift()

print(f"Drift detected: {drift_results['drift_detected']}")
print(f"Drift share: {drift_results['drift_share']:.2%}")
print(f"Drifted features: {drift_results['drifted_features']}")
```

## Drift Metrics

### Dataset-Level Drift

Overall drift across all features:

```python
{
    "drift_detected": True,
    "drift_share": 0.35,  # 35% of features drifted
    "drifted_features": 14,
    "requires_retraining": True  # > 30% threshold
}
```

### Feature-Level Drift

Individual feature drift scores:

```python
feature_drift = {
    "player_elo": 0.12,    # No drift
    "opponent_elo": 0.08,  # No drift
    "p_ace_mean_5": 0.45,  # Drift detected!
}
```

## Performance Monitoring

Track model accuracy over time:

```python
predictions = model.predict(X_current)
actuals = y_current

perf_results = monitor.check_model_performance(predictions, actuals)

print(f"Current accuracy: {perf_results['accuracy']:.2%}")
print(f"Performance degradation: {perf_results['performance_degradation']}")
```

## Retraining Triggers

Automatic retraining is triggered when:

1. **Drift Share > 30%**: Significant feature drift
2. **Accuracy < 60%**: Below minimum threshold
3. **Manual Trigger**: Via GitHub Actions

```python
def should_retrain(drift_results, performance_results):
    drift_trigger = drift_results['drift_share'] > 0.30
    perf_trigger = performance_results['accuracy'] < 0.60
    
    return drift_trigger or perf_trigger
```

## Monitoring Reports

Generate comprehensive HTML reports:

```python
monitor.generate_monitoring_report("reports/monitoring_report.html")
```

Report includes:
- Data drift metrics
- Feature distributions
- Model performance
- Data quality checks

## Configuration

Monitoring is configured via the model config:

```yaml
monitoring:
  drift_threshold: 0.30
  accuracy_threshold: 0.60
  check_frequency: "monthly"
```

## Scheduled Monitoring

The monitoring workflow runs automatically:

```yaml
# .github/workflows/monitoring.yml
on:
  schedule:
    - cron: '0 0 15 * *'  # 15th of each month
  workflow_dispatch:
```

## Data Quality Checks

Monitor data quality metrics:

```python
quality_results = monitor.check_data_quality()

print(f"Missing values: {quality_results['missing_values']}")
print(f"Quality score: {quality_results['data_quality_score']:.2%}")
```

Checks include:
- Missing values
- Duplicate records
- Outliers
- Data types
- Value ranges

## Alerting

When drift is detected, the workflow:

1. Generates monitoring report
2. Creates a GitHub issue
3. Optionally triggers retraining
4. Sends notifications

Example issue:

```markdown
# ⚠️ Data Drift Detected - Retraining Required

Significant data drift has been detected in the monitoring pipeline.

**Drift Metrics:**
- Drift Share: 35%
- Drifted Features: 14
- Accuracy: 58%

**Action Required**: Model retraining is recommended.

[View Monitoring Report](link-to-report)
```

## Best Practices

1. **Regular Monitoring**: Check monthly or after significant events
2. **Baseline Updates**: Refresh reference data periodically
3. **Track Trends**: Monitor drift over time, not just binary
4. **Investigate Causes**: Understand why drift occurs
5. **Document Changes**: Log all retraining decisions
6. **Test Alerts**: Ensure notification system works

## Troubleshooting

### False Positives

Adjust drift thresholds:

```python
monitor.drift_threshold = 0.40  # Less sensitive
```

### Missing Reports

Check file permissions and paths:

```bash
mkdir -p reports
chmod 755 reports
```

### High Drift Rate

Common causes:
- Seasonal changes in tennis
- Tournament schedule changes
- Player retirements
- Data collection issues

Investigate and decide if retraining is needed or if drift is expected.

## Visualization

Create drift visualizations:

```python
import matplotlib.pyplot as plt

# Plot drift over time
plt.figure(figsize=(10, 6))
plt.plot(dates, drift_scores)
plt.axhline(y=0.30, color='r', linestyle='--', label='Threshold')
plt.xlabel('Date')
plt.ylabel('Drift Share')
plt.title('Data Drift Over Time')
plt.legend()
plt.show()
```
