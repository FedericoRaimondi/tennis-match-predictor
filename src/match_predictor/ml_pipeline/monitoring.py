"""Model and data monitoring module using Evidently."""

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
from loguru import logger


class ModelMonitor:
    """Class for monitoring model and data drift."""

    def __init__(self):
        """Initialize model monitor."""
        self.logger = logger
        self.reference_data = None
        self.current_data = None
        self.column_mapping = None

    def set_reference_data(self, df: pd.DataFrame, target_column: str = "winner"):
        """
        Set reference data for drift detection.

        Args:
            df: Reference dataframe
            target_column: Name of target column
        """
        self.reference_data = df.copy()
        self.column_mapping = ColumnMapping(target=target_column, prediction="prediction")
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

        # Create data drift report
        drift_report = Report(metrics=[DataDriftPreset()])

        drift_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
        )

        # Extract metrics
        drift_results = drift_report.as_dict()

        # Parse results
        metrics = drift_results.get("metrics", [])
        dataset_drift_metric = next(
            (m for m in metrics if m.get("metric") == "DatasetDriftMetric"),
            None
        )

        if dataset_drift_metric:
            drift_detected = dataset_drift_metric["result"]["dataset_drift"]
            drift_share = dataset_drift_metric["result"]["drift_share"]
            drifted_features = dataset_drift_metric["result"].get("number_of_drifted_columns", 0)

            self.logger.info(f"Data drift detected: {drift_detected}")
            self.logger.info(f"Drift share: {drift_share:.2%}")
            self.logger.info(f"Number of drifted features: {drifted_features}")

            return {
                "drift_detected": drift_detected,
                "drift_share": drift_share,
                "drifted_features": drifted_features,
                "requires_retraining": drift_detected and drift_share > 0.3
            }

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

        # Create data quality report
        quality_report = Report(metrics=[DataQualityPreset()])

        quality_report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
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

        # Create a dataframe with predictions and actuals
        performance_df = pd.DataFrame({
            "prediction": predictions,
            "target": actual
        })

        # Create classification report
        classification_report = Report(metrics=[ClassificationPreset()])

        classification_report.run(
            reference_data=None,
            current_data=performance_df,
            column_mapping=ColumnMapping(target="target", prediction="prediction")
        )

        # Extract metrics
        perf_results = classification_report.as_dict()
        metrics = perf_results.get("metrics", [])

        accuracy = 0.0
        precision = 0.0
        recall = 0.0

        for metric in metrics:
            if metric.get("metric") == "ClassificationQualityMetric":
                result = metric.get("result", {}).get("current", {})
                accuracy = result.get("accuracy", 0.0)
                precision = result.get("precision", 0.0)
                recall = result.get("recall", 0.0)
                break

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

        # Create comprehensive report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=self.current_data,
            column_mapping=self.column_mapping
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
