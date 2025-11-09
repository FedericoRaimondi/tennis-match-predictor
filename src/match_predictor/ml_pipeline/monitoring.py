"""Model and data monitoring module with intelligent feature-aware drift detection."""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from match_predictor.ml_pipeline.drift_detection import FeatureAwareDriftDetector


class ModelMonitor:
    """Class for monitoring model and data drift with feature-aware detection."""

    def __init__(self, model=None, feature_names: list[str] | None = None):
        """Initialize model monitor.

        Args:
            model: Trained model with feature_names_in_ attribute (optional)
            feature_names: Explicit list of feature names to monitor (optional)
        """
        self.logger = logger
        self.drift_detector = FeatureAwareDriftDetector(model=model, feature_names=feature_names)
        self.reference_data = None
        self.current_data = None
        self.target_column = None
        self.prediction_column = None

    def set_reference_data(
        self, df: pd.DataFrame, target_column: str = "winner", prediction_column: str = "prediction"
    ):
        """Set reference data for drift detection.

        Args:
            df: Reference dataframe
            target_column: Name of target column
            prediction_column: Name of prediction column
        """
        self.reference_data = df.copy()
        self.target_column = target_column
        self.prediction_column = prediction_column

        # Filter out target and prediction columns for drift detection
        feature_columns = [col for col in df.columns if col not in [target_column, prediction_column]]
        if feature_columns:
            self.drift_detector.set_reference_data(df[feature_columns])
        else:
            self.drift_detector.set_reference_data(df)

        self.logger.info(f"Reference data set with {len(df)} samples")

    def set_current_data(self, df: pd.DataFrame):
        """Set current data for drift detection.

        Args:
            df: Current dataframe
        """
        self.current_data = df.copy()

        # Filter out target and prediction columns for drift detection
        feature_columns = [col for col in df.columns if col not in [self.target_column, self.prediction_column]]
        if feature_columns:
            self.drift_detector.set_current_data(df[feature_columns])
        else:
            self.drift_detector.set_current_data(df)

        self.logger.info(f"Current data set with {len(df)} samples")

    def check_data_drift(self, ks_threshold: float = 0.05, js_threshold: float = 0.1) -> dict[str, Any]:
        """Check for data drift between reference and current data using feature-aware detection.

        Args:
            ks_threshold: P-value threshold for KS test on numerical features
            js_threshold: Divergence threshold for JS divergence on categorical features

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Both reference and current data must be set")

        self.logger.info("Checking for data drift with feature-aware detection...")

        # Use the feature-aware drift detector
        drift_results = self.drift_detector.detect_drift(ks_threshold=ks_threshold, js_threshold=js_threshold)

        self.logger.info(f"Data drift detected: {drift_results['drift_detected']}")
        self.logger.info(f"Drift share: {drift_results['drift_share']:.2%}")
        self.logger.info(
            f"Number of drifted features: {drift_results['drifted_features']}/{drift_results['total_features']}"
        )

        return drift_results

    def check_data_quality(self) -> dict[str, Any]:
        """Check data quality metrics.

        Returns:
            Dictionary with data quality results
        """
        if self.current_data is None:
            raise ValueError("Current data must be set")

        self.logger.info("Checking data quality...")

        # Calculate missing values
        missing_values = self.current_data.isnull().sum().sum()
        total_cells = len(self.current_data) * len(self.current_data.columns)
        data_quality_score = 1.0 - (missing_values / total_cells) if total_cells > 0 else 1.0

        self.logger.info(f"Missing values: {missing_values}")
        self.logger.info(f"Data quality score: {data_quality_score:.4f}")

        return {
            "missing_values": int(missing_values),
            "data_quality_score": float(data_quality_score),
        }

    def check_model_performance(self, predictions: pd.Series, actual: pd.Series) -> dict[str, Any]:
        """Check model performance metrics using custom error metrics.

        Args:
            predictions: Model predictions
            actual: Actual target values

        Returns:
            Dictionary with performance metrics
        """
        self.logger.info("Checking model performance...")

        # Use feature-aware drift detector's performance metrics
        performance_results = self.drift_detector.calculate_performance_metrics(predictions, actual)

        self.logger.info(f"Model accuracy: {performance_results['accuracy']:.4f}")
        self.logger.info(f"Model precision: {performance_results['precision']:.4f}")
        self.logger.info(f"Model recall: {performance_results['recall']:.4f}")
        self.logger.info(f"Model F1: {performance_results.get('f1_score', 0):.4f}")

        return performance_results

    def generate_monitoring_report(self, output_path: str | Path = "reports/monitoring_report.html") -> None:
        """Generate comprehensive monitoring report.

        Args:
            output_path: Path to save the HTML report
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Both reference and current data must be set")

        self.logger.info("Generating monitoring report...")

        # Detect drift
        drift_results = self.check_data_drift()

        # Create HTML report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .metric {{ margin: 10px 0; }}
                .drift {{ color: #d9534f; font-weight: bold; }}
                .no-drift {{ color: #5cb85c; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .drifted {{ background-color: #ffcccc; }}
            </style>
        </head>
        <body>
            <h1>🎾 Tennis Match Predictor - Monitoring Report</h1>

            <h2>📊 Summary</h2>
            <div class="metric">
                <strong>Reference Samples:</strong> {len(self.reference_data)}
            </div>
            <div class="metric">
                <strong>Current Samples:</strong> {len(self.current_data)}
            </div>
            <div class="metric">
                <strong>Features Monitored:</strong> {drift_results["total_features"]}
            </div>
            <div class="metric">
                <strong>Drift Status:</strong>
                <span class="{"drift" if drift_results["drift_detected"] else "no-drift"}">
                    {"DRIFT DETECTED" if drift_results["drift_detected"] else "NO DRIFT"}
                </span>
            </div>
            <div class="metric">
                <strong>Drift Share:</strong> {drift_results["drift_share"]:.2%}
            </div>
            <div class="metric">
                <strong>Drifted Features:</strong> {drift_results["drifted_features"]} / {drift_results["total_features"]}
            </div>
            <div class="metric">
                <strong>Requires Retraining:</strong>
                <span class="{"drift" if drift_results["requires_retraining"] else "no-drift"}">
                    {"YES" if drift_results["requires_retraining"] else "NO"}
                </span>
            </div>

            <h2>🔍 Feature-Level Drift Details</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Test</th>
                    <th>Drift Status</th>
                    <th>Details</th>
                </tr>
        """

        for feature, details in drift_results.get("feature_drift_details", {}).items():
            drift_class = "drifted" if details.get("drift_detected", False) else ""
            drift_status = "DRIFT" if details.get("drift_detected", False) else "OK"

            test_details = ""
            if details.get("test") == "ks":
                test_details = f"Statistic: {details.get('statistic', 0):.4f}, P-value: {details.get('p_value', 0):.4f}"
            elif details.get("test") == "js":
                test_details = f"Divergence: {details.get('divergence', 0):.4f}"

            html_content += f"""
                <tr class="{drift_class}">
                    <td>{feature}</td>
                    <td>{details.get("type", "unknown")}</td>
                    <td>{details.get("test", "unknown").upper()}</td>
                    <td>{drift_status}</td>
                    <td>{test_details}</td>
                </tr>
            """

        html_content += """
            </table>

            <h2>ℹ️ About This Report</h2>
            <p>
                This report uses intelligent feature-aware drift detection:
            </p>
            <ul>
                <li><strong>Numerical Features:</strong> Kolmogorov-Smirnov (KS) test</li>
                <li><strong>Categorical Features:</strong> Jensen-Shannon (JS) divergence</li>
                <li>Only model-relevant features are monitored for efficiency</li>
                <li>Retraining is recommended when drift share exceeds 30%</li>
            </ul>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(f"Monitoring report saved to {output_path}")

    def should_trigger_retraining(
        self, drift_results: dict | None = None, performance_results: dict | None = None
    ) -> tuple[bool, str]:
        """Determine if model retraining should be triggered.

        Args:
            drift_results: Results from data drift check
            performance_results: Results from performance check

        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []

        # Check data drift
        if drift_results and drift_results.get("requires_retraining"):
            reasons.append(f"Significant data drift detected (drift share: {drift_results.get('drift_share', 0):.2%})")

        # Check performance degradation
        if performance_results and performance_results.get("performance_degradation"):
            reasons.append(f"Performance degradation detected (accuracy: {performance_results.get('accuracy', 0):.4f})")

        should_retrain = len(reasons) > 0

        if should_retrain:
            reason_str = "; ".join(reasons)
            self.logger.warning(f"Retraining triggered: {reason_str}")
        else:
            reason_str = "No issues detected"
            self.logger.info("Model monitoring passed, no retraining needed")

        return should_retrain, reason_str
