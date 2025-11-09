"""Custom feature-aware drift detection module.

This module provides intelligent drift detection that:
- Extracts actual model features using feature_names_in_ or similar attributes
- Applies appropriate statistical tests based on feature type
- Ignores unused dataset columns for optimal performance
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score


class FeatureAwareDriftDetector:
    """Feature-aware drift detector that focuses on model-relevant features."""

    def __init__(self, model=None, feature_names: list[str] | None = None):
        """Initialize the drift detector.

        Args:
            model: Trained model with feature_names_in_ attribute (optional)
            feature_names: Explicit list of feature names to monitor (optional)
        """
        self.logger = logger
        self.model = model
        self.feature_names = self._extract_feature_names(model, feature_names)
        self.reference_data = None
        self.current_data = None

    def _extract_feature_names(self, model, explicit_names: list[str] | None = None) -> list[str] | None:
        """Extract feature names from model or use explicit list.

        Args:
            model: Trained model
            explicit_names: Explicit feature names

        Returns:
            List of feature names or None
        """
        if explicit_names is not None:
            self.logger.info(f"Using {len(explicit_names)} explicit feature names")
            return explicit_names

        if model is None:
            self.logger.warning("No model or feature names provided")
            return None

        # Try to extract from model
        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
            self.logger.info(f"Extracted {len(features)} features from model.feature_names_in_")
            return features
        elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
            # XGBoost specific
            features = model.get_booster().feature_names
            self.logger.info(f"Extracted {len(features)} features from XGBoost model")
            return features
        elif hasattr(model, "feature_name_"):
            # LightGBM specific
            features = list(model.feature_name_)
            self.logger.info(f"Extracted {len(features)} features from model.feature_name_")
            return features

        self.logger.warning("Could not extract feature names from model")
        return None

    def set_reference_data(self, df: pd.DataFrame):
        """Set reference data for drift detection.

        Args:
            df: Reference dataframe
        """
        # Filter to only model features if available
        if self.feature_names is not None:
            available_features = [f for f in self.feature_names if f in df.columns]
            self.reference_data = df[available_features].copy()
            self.logger.info(
                f"Reference data set with {len(self.reference_data)} samples, {len(available_features)} features"
            )
        else:
            self.reference_data = df.copy()
            self.logger.info(f"Reference data set with {len(self.reference_data)} samples, all columns")

    def set_current_data(self, df: pd.DataFrame):
        """Set current data for drift detection.

        Args:
            df: Current dataframe
        """
        # Filter to only model features if available
        if self.feature_names is not None:
            available_features = [f for f in self.feature_names if f in df.columns]
            self.current_data = df[available_features].copy()
            self.logger.info(
                f"Current data set with {len(self.current_data)} samples, {len(available_features)} features"
            )
        else:
            self.current_data = df.copy()
            self.logger.info(f"Current data set with {len(self.current_data)} samples, all columns")

    def _is_numerical(self, series: pd.Series) -> bool:
        """Check if a series contains numerical data.

        Args:
            series: Pandas series

        Returns:
            True if numerical, False otherwise
        """
        return pd.api.types.is_numeric_dtype(series)

    def _is_categorical(self, series: pd.Series) -> bool:
        """Check if a series contains categorical data.

        Args:
            series: Pandas series

        Returns:
            True if categorical, False otherwise
        """
        # Consider as categorical if: object type, category type, or integer with few unique values
        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
            return True
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= 20:
            return True
        return False

    def _kolmogorov_smirnov_test(
        self, ref_series: pd.Series, cur_series: pd.Series, threshold: float = 0.05
    ) -> dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for numerical features.

        Args:
            ref_series: Reference data series
            cur_series: Current data series
            threshold: P-value threshold for drift detection

        Returns:
            Dictionary with test results
        """
        try:
            # Remove NaN values
            ref_clean = ref_series.dropna()
            cur_clean = cur_series.dropna()

            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return {"drift_detected": False, "statistic": 0.0, "p_value": 1.0, "test": "ks"}

            statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
            drift_detected = p_value < threshold

            return {
                "drift_detected": drift_detected,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "test": "ks",
            }
        except Exception as e:
            self.logger.warning(f"KS test failed: {e}")
            return {"drift_detected": False, "statistic": 0.0, "p_value": 1.0, "test": "ks", "error": str(e)}

    def _jensen_shannon_divergence(
        self, ref_series: pd.Series, cur_series: pd.Series, threshold: float = 0.1
    ) -> dict[str, Any]:
        """Calculate Jensen-Shannon divergence for categorical features.

        Args:
            ref_series: Reference data series
            cur_series: Current data series
            threshold: Divergence threshold for drift detection

        Returns:
            Dictionary with test results
        """
        try:
            # Get value counts and normalize to probabilities
            ref_counts = ref_series.value_counts(normalize=True, dropna=True)
            cur_counts = cur_series.value_counts(normalize=True, dropna=True)

            # Get all unique categories
            all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))

            if len(all_categories) == 0:
                return {"drift_detected": False, "divergence": 0.0, "test": "js"}

            # Create probability distributions with small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_probs = np.array([ref_counts.get(cat, 0) + epsilon for cat in all_categories])
            cur_probs = np.array([cur_counts.get(cat, 0) + epsilon for cat in all_categories])

            # Normalize
            ref_probs = ref_probs / ref_probs.sum()
            cur_probs = cur_probs / cur_probs.sum()

            # Calculate JS divergence
            divergence = jensenshannon(ref_probs, cur_probs)
            drift_detected = divergence > threshold

            return {
                "drift_detected": drift_detected,
                "divergence": float(divergence),
                "test": "js",
                "unique_categories": len(all_categories),
            }
        except Exception as e:
            self.logger.warning(f"JS divergence failed: {e}")
            return {"drift_detected": False, "divergence": 0.0, "test": "js", "error": str(e)}

    def detect_drift(self, ks_threshold: float = 0.05, js_threshold: float = 0.1) -> dict[str, Any]:
        """Detect drift on model-relevant features.

        Args:
            ks_threshold: P-value threshold for KS test
            js_threshold: Divergence threshold for JS divergence

        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Both reference and current data must be set")

        self.logger.info("Detecting feature-aware drift...")

        # Get common columns
        common_features = list(set(self.reference_data.columns) & set(self.current_data.columns))

        if len(common_features) == 0:
            self.logger.warning("No common features between reference and current data")
            return {
                "drift_detected": False,
                "drift_share": 0.0,
                "drifted_features": 0,
                "total_features": 0,
                "feature_drift_details": {},
                "requires_retraining": False,
            }

        feature_drift_details = {}
        drifted_features = []

        for feature in common_features:
            ref_series = self.reference_data[feature]
            cur_series = self.current_data[feature]

            # Determine feature type and apply appropriate test
            if self._is_numerical(ref_series) and not self._is_categorical(ref_series):
                # Numerical feature - use KS test
                result = self._kolmogorov_smirnov_test(ref_series, cur_series, ks_threshold)
                feature_drift_details[feature] = {"type": "numerical", **result}
            else:
                # Categorical feature - use JS divergence
                result = self._jensen_shannon_divergence(ref_series, cur_series, js_threshold)
                feature_drift_details[feature] = {"type": "categorical", **result}

            if result.get("drift_detected", False):
                drifted_features.append(feature)

        # Calculate drift metrics
        total_features = len(common_features)
        num_drifted = len(drifted_features)
        drift_share = num_drifted / total_features if total_features > 0 else 0.0
        drift_detected = num_drifted > 0

        self.logger.info(
            f"Drift analysis complete: {num_drifted}/{total_features} features drifted ({drift_share:.2%})"
        )

        # Determine if retraining is required (>30% drift)
        requires_retraining = drift_share > 0.3

        return {
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "drifted_features": num_drifted,
            "total_features": total_features,
            "drifted_feature_names": drifted_features,
            "feature_drift_details": feature_drift_details,
            "requires_retraining": requires_retraining,
        }

    def calculate_performance_metrics(self, predictions: pd.Series, actual: pd.Series) -> dict[str, Any]:
        """Calculate custom performance metrics.

        Args:
            predictions: Model predictions
            actual: Actual target values

        Returns:
            Dictionary with performance metrics
        """
        self.logger.info("Calculating performance metrics...")

        try:
            accuracy = accuracy_score(actual, predictions)
            precision = precision_score(actual, predictions, zero_division=0)
            recall = recall_score(actual, predictions, zero_division=0)

            # Calculate F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            self.logger.info(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
            )

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "performance_degradation": accuracy < 0.60,
            }
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "performance_degradation": True,
                "error": str(e),
            }
