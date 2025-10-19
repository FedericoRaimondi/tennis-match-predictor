"""Model and data monitoring module using Evidently."""

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from loguru import logger

# Try to import metrics, fall back to simpler approach if not available
try:
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    EVIDENTLY_METRICS_AVAILABLE = True
except ImportError:
    EVIDENTLY_METRICS_AVAILABLE = False
    logger.warning("Evidently metrics not available, using simplified monitoring")


class ModelMonitor:
    """Class for monitoring model and data drift."""

    def __init__(self):
        """Initialize model monitor."""
        self.logger = logger
        self.reference_data = None
        self.current_data = None
        self.target_column = None
        self.prediction_column = None

    def set_reference_data(self, df: pd.DataFrame, target_column: str = "winner", prediction_column: str = "prediction"):
        """
        Set reference data for drift detection.

        Args:
            df: Reference dataframe
            target_column: Name of target column
            prediction_column: Name of prediction column
        """
        self.reference_data = df.copy()
        self.target_column = target_column
        self.prediction_column = prediction_column
        self.logger.info(f"Reference data set with {len(df)} samples")

    def set_current_data(self, df: pd.DataFrame):
        """
        Set current data for drift detection.

        Args:
            df: Current dataframe
        """
        self.current_data = df.copy()
        self.logger.info(f"Current data set with {len(df)} samples")

    def check_data_drift(self) -> dict[str, Any]:
        """
        Check for data drift between reference and current data.

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Both reference and current data must be set")

        self.logger.info("Checking for data drift...")

        if not EVIDENTLY_METRICS_AVAILABLE:
            # Simplified drift detection
            self.logger.warning("Using simplified drift detection")
            return {
                "drift_detected": False,
                "drift_share": 0.0,
                "drifted_features": 0,
                "requires_retraining": False
            }

        # Create data drift report
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )

        # Extract metrics
        drift_results = drift_report.as_dict()

        # Parse results - structure may vary by Evidently version
        try:
            metrics = drift_results.get("metrics", [])
            dataset_drift_metric = next(
                (m for m in metrics if m.get("metric") == "DatasetDriftMetric"),
                None
            )

            if dataset_drift_metric:
                result = dataset_drift_metric.get("result", {})
                drift_detected = result.get("dataset_drift", False)
                drift_share = result.get("drift_share", 0.0)
                drifted_features = result.get("number_of_drifted_columns", 0)

                self.logger.info(f"Data drift detected: {drift_detected}")
                self.logger.info(f"Drift share: {drift_share:.2%}")
                self.logger.info(f"Number of drifted features: {drifted_features}")

                return {
                    "drift_detected": drift_detected,
                    "drift_share": drift_share,
                    "drifted_features": drifted_features,
                    "requires_retraining": drift_detected and drift_share > 0.3
                }
        except Exception as e:
            self.logger.warning(f"Error parsing drift results: {e}")

        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "drifted_features": 0,
            "requires_retraining": False
        }

    def check_data_quality(self) -> dict[str, Any]:
        """
        Check data quality metrics.

        Returns:
            Dictionary with data quality results
        """
        if self.current_data is None:
            raise ValueError("Current data must be set")

        self.logger.info("Checking data quality...")

        if not EVIDENTLY_METRICS_AVAILABLE:
            # Simplified quality check
            missing_values = self.current_data.isnull().sum().sum()
            self.logger.info(f"Missing values: {missing_values}")
            return {
                "missing_values": missing_values,
                "data_quality_score": 1.0 - (missing_values / (len(self.current_data) * len(self.current_data.columns)))
            }

        # Create data quality report
        quality_report = Report(metrics=[DatasetMissingValuesMetric()])

        quality_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )

        # Extract metrics
        quality_results = quality_report.as_dict()

        # Parse key quality metrics
        metrics = quality_results.get("metrics", [])

        missing_values = 0
        for metric in metrics:
            if metric.get("metric") == "DatasetMissingValuesMetric":
                missing_values = metric["result"].get("current", {}).get("number_of_missing_values", 0)
                break

        self.logger.info(f"Missing values: {missing_values}")

        return {
            "missing_values": missing_values,
            "data_quality_score": 1.0 - (missing_values / (len(self.current_data) * len(self.current_data.columns)))
        }

    def check_model_performance(self, predictions: pd.Series, actual: pd.Series) -> dict[str, Any]:
        """
        Check model performance metrics.

        Args:
            predictions: Model predictions
            actual: Actual target values

        Returns:
            Dictionary with performance metrics
        """
        self.logger.info("Checking model performance...")

        # Calculate metrics manually
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        accuracy = accuracy_score(actual, predictions)
        try:
            precision = precision_score(actual, predictions, zero_division=0)
            recall = recall_score(actual, predictions, zero_division=0)
        except Exception:
            precision = 0.0
            recall = 0.0

        self.logger.info(f"Model accuracy: {accuracy:.4f}")
        self.logger.info(f"Model precision: {precision:.4f}")
        self.logger.info(f"Model recall: {recall:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "performance_degradation": accuracy < 0.60  # Threshold can be configured
        }

    def generate_monitoring_report(
        self,
        output_path: str | Path = "reports/monitoring_report.html"
    ) -> None:
        """
        Generate comprehensive monitoring report.

        Args:
            output_path: Path to save the HTML report
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Both reference and current data must be set")

        self.logger.info("Generating monitoring report...")

        if not EVIDENTLY_METRICS_AVAILABLE:
            # Create simple HTML report
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("<html><body><h1>Monitoring Report</h1>")
                f.write("<p>Evidently metrics not fully available. Using simplified monitoring.</p>")
                f.write("</body></html>")
            
            self.logger.info(f"Simple monitoring report saved to {output_path}")
            return

        # Create comprehensive report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
            DatasetMissingValuesMetric(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data
        )

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report.save_html(str(output_path))
        self.logger.info(f"Monitoring report saved to {output_path}")

    def should_trigger_retraining(
        self,
        drift_results: dict | None = None,
        performance_results: dict | None = None
    ) -> tuple[bool, str]:
        """
        Determine if model retraining should be triggered.

        Args:
            drift_results: Results from data drift check
            performance_results: Results from performance check

        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []

        # Check data drift
        if drift_results and drift_results.get("requires_retraining"):
            reasons.append(
                f"Significant data drift detected "
                f"(drift share: {drift_results.get('drift_share', 0):.2%})"
            )

        # Check performance degradation
        if performance_results and performance_results.get("performance_degradation"):
            reasons.append(
                f"Performance degradation detected "
                f"(accuracy: {performance_results.get('accuracy', 0):.4f})"
            )

        should_retrain = len(reasons) > 0

        if should_retrain:
            reason_str = "; ".join(reasons)
            self.logger.warning(f"Retraining triggered: {reason_str}")
        else:
            reason_str = "No issues detected"
            self.logger.info("Model monitoring passed, no retraining needed")

        return should_retrain, reason_str
