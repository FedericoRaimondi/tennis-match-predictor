"""Prefect flows for training and monitoring pipelines."""

import pickle
from pathlib import Path

import pandas as pd
from loguru import logger
from prefect import flow, task

from match_predictor.config import DataConfig, ModelConfig
from match_predictor.data.data_loader import DataLoader
from match_predictor.ml_pipeline.feature_engineering import FeatureEngineer
from match_predictor.ml_pipeline.monitoring import ModelMonitor
from match_predictor.ml_pipeline.training import ModelTrainer


@task(name="load_data", retries=2, retry_delay_seconds=60)
def load_data_task(data_config: DataConfig) -> pd.DataFrame:
    """Load match data from source.

    Args:
        data_config: Data configuration

    Returns:
        DataFrame with match data
    """
    logger.info("Loading match data...")
    loader = DataLoader(data_config)
    matches = loader.load_matches()
    logger.info(f"Loaded {len(matches)} matches")
    return matches


@task(name="prepare_ml_data")
def prepare_ml_data_task(data_config: DataConfig, matches: pd.DataFrame) -> pd.DataFrame:
    """Prepare ML dataset from matches.

    Args:
        data_config: Data configuration
        matches: Match data

    Returns:
        ML-ready dataset
    """
    logger.info("Preparing ML dataset...")
    loader = DataLoader(data_config)
    ml_data = loader.get_ml_data(df=matches)
    logger.info(f"Prepared ML dataset with {len(ml_data)} samples")
    return ml_data


@task(name="train_model")
def train_model_task(ml_data: pd.DataFrame, model_config: ModelConfig, tune_hyperparameters: bool = False) -> dict:
    """Train the model.

    Args:
        ml_data: ML dataset
        model_config: Model configuration
        tune_hyperparameters: Whether to perform hyperparameter tuning

    Returns:
        Dictionary with training metrics
    """
    logger.info("Training model...")
    trainer = ModelTrainer(model_config)
    metrics = trainer.train(ml_data, tune_hyperparameters=tune_hyperparameters, log_to_mlflow=True)
    logger.info(f"Training complete. Test accuracy: {metrics['test_accuracy']:.4f}")

    # Save model
    model_path = Path(model_config.champion_model_path) / "latest_model.pkl"
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    return metrics


@task(name="evaluate_and_promote")
def evaluate_and_promote_task(metrics: dict, model_config: ModelConfig) -> bool:
    """Evaluate if the new model should be promoted to champion.

    Args:
        metrics: Training metrics
        model_config: Model configuration

    Returns:
        True if model was promoted, False otherwise
    """
    logger.info("Evaluating model for promotion...")

    new_accuracy = metrics["test_accuracy"]
    logger.info(f"New model accuracy: {new_accuracy:.4f}")

    # Check if champion model exists
    champion_path = Path(model_config.champion_model_path) / "champion_model.pkl"
    latest_path = Path(model_config.champion_model_path) / "latest_model.pkl"

    if champion_path.exists():
        with open(champion_path, "rb") as f:
            champion_data = pickle.load(f)
        champion_accuracy = champion_data["metrics"]["test_accuracy"]
        logger.info(f"Champion model accuracy: {champion_accuracy:.4f}")
    else:
        champion_accuracy = None
        logger.info("No champion model found")

    # Promote if better
    if champion_accuracy is None or new_accuracy > champion_accuracy:
        import shutil

        shutil.copy(latest_path, champion_path)
        logger.info("✓ New model promoted to champion!")
        return True
    else:
        logger.info("✗ New model not better than champion")
        return False


@task(name="save_data")
def save_data_task(matches: pd.DataFrame, data_config: DataConfig):
    """Save match data for inference.

    Args:
        matches: Match data
        data_config: Data configuration
    """
    logger.info("Saving match data...")
    data_path = Path(data_config.inference_data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Save matches
    matches_file = data_path / data_config.matches_results_file
    matches.to_csv(matches_file, index=False)
    logger.info(f"Saved matches to {matches_file}")

    # Initialize DataLoader for additional data saving
    loader = DataLoader(data_config)

    # Save tournament info
    tournament_info = loader.get_tournament_info(df=matches)
    tournament_file = data_path / data_config.tournament_info_file
    tournament_info.to_csv(tournament_file, index=False)
    logger.info(f"Saved tournament info to {tournament_file}")

    # Save latest player stats for inference
    player_stats_file = data_path / data_config.player_stats_file
    loader.save_latest_player_stats(df=matches, output_path=str(player_stats_file))
    logger.info(f"Saved latest player stats to {player_stats_file}")


@flow(name="Training Pipeline", log_prints=True)
def training_flow(
    tune_hyperparameters: bool = False,
    data_config_path: str = "config/data_config.yaml",
    model_config_path: str = "config/model_config.yaml",
):
    """Main training pipeline flow.

    Args:
        tune_hyperparameters: Whether to perform hyperparameter tuning
        data_config_path: Path to data configuration YAML
        model_config_path: Path to model configuration YAML
    """
    logger.info("Starting training pipeline...")

    # Load configurations
    data_config = DataConfig.from_yaml(data_config_path)
    model_config = ModelConfig.from_yaml(model_config_path)

    # Execute pipeline
    matches = load_data_task(data_config)
    save_data_task(matches, data_config)
    ml_data = prepare_ml_data_task(data_config, matches)
    metrics = train_model_task(ml_data, model_config, tune_hyperparameters)
    promoted = evaluate_and_promote_task(metrics, model_config)

    logger.info(f"Training pipeline complete. Model promoted: {promoted}")
    return {"metrics": metrics, "promoted": promoted}


@task(name="check_new_data")
def check_new_data_task(data_config: DataConfig) -> bool:
    """Check if new data is available.

    Args:
        data_config: Data configuration

    Returns:
        True if new data is available
    """
    logger.info("Checking for new data...")

    loader = DataLoader(data_config)
    new_matches = loader.load_matches()

    # Check against existing data
    data_path = Path(data_config.inference_data_path)
    matches_file = data_path / data_config.matches_results_file

    if matches_file.exists():
        with open(matches_file, "rb") as f:
            existing_matches = pickle.load(f)

        new_count = len(new_matches)
        existing_count = len(existing_matches)

        has_new_data = new_count > existing_count
        logger.info(f"Existing: {existing_count}, New: {new_count}, Has new data: {has_new_data}")
        return has_new_data
    else:
        logger.info("No existing data found, assuming new data available")
        return True


@task(name="detect_drift")
def detect_drift_task(data_config: DataConfig) -> dict:
    """Detect data drift.

    Args:
        data_config: Data configuration

    Returns:
        Dictionary with drift detection results
    """
    logger.info("Detecting data drift...")

    loader = DataLoader(data_config)

    # Load reference and current data
    data_path = Path(data_config.inference_data_path)
    matches_file = data_path / data_config.matches_results_file

    if not matches_file.exists():
        logger.warning("No reference data found, skipping drift detection")
        return {"drift_detected": False, "drift_share": 0.0, "requires_retraining": False}

    with open(matches_file, "rb") as f:
        reference_matches = pickle.load(f)

    current_matches = loader.load_matches()

    # Prepare datasets
    reference_ml = loader.get_ml_data(df=reference_matches)
    current_ml = loader.get_ml_data(df=current_matches)

    # Engineer features
    engineer = FeatureEngineer()
    reference_features = engineer.engineer_features(reference_ml)
    current_features = engineer.engineer_features(current_ml)

    # Prepare for monitoring
    X_ref, y_ref = engineer.prepare_features_for_training(reference_features)
    X_cur, y_cur = engineer.prepare_features_for_training(current_features)

    # Monitor drift
    monitor = ModelMonitor()
    monitor.set_reference_data(X_ref.assign(winner=y_ref))
    monitor.set_current_data(X_cur.assign(winner=y_cur))

    drift_results = monitor.check_data_drift()

    logger.info(f"Drift detected: {drift_results['drift_detected']}")
    logger.info(f"Drift share: {drift_results['drift_share']:.2%}")
    logger.info(f"Requires retraining: {drift_results['requires_retraining']}")

    # Generate monitoring report
    report_path = Path("reports") / "monitoring_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    monitor.generate_monitoring_report(report_path)
    logger.info(f"Monitoring report saved to {report_path}")

    return drift_results


@flow(name="Monitoring Pipeline", log_prints=True)
def monitoring_flow(
    data_config_path: str = "config/data_config.yaml", model_config_path: str = "config/model_config.yaml"
):
    """Main monitoring pipeline flow.

    Args:
        data_config_path: Path to data configuration YAML
        model_config_path: Path to model configuration YAML
    """
    logger.info("Starting monitoring pipeline...")

    # Load configurations
    data_config = DataConfig.from_yaml(data_config_path)
    _model_config = ModelConfig.from_yaml(model_config_path)

    # Check for new data
    has_new_data = check_new_data_task(data_config)

    if not has_new_data:
        logger.info("No new data available, skipping monitoring")
        return {"has_new_data": False, "should_retrain": False}

    # Detect drift
    drift_results = detect_drift_task(data_config)

    should_retrain = drift_results["requires_retraining"]

    logger.info(f"Monitoring pipeline complete. Should retrain: {should_retrain}")
    return {"has_new_data": has_new_data, "drift_results": drift_results, "should_retrain": should_retrain}


def run_training_flow():
    """Entrypoint for training flow CLI."""
    import sys

    tune_hyperparameters = "--tune" in sys.argv or "-t" in sys.argv

    logger.info("Running training flow...")
    result = training_flow(tune_hyperparameters=tune_hyperparameters)

    if result["promoted"]:
        logger.info("✓ Model successfully promoted to champion")
        sys.exit(0)
    else:
        logger.info("✗ Model not promoted")
        sys.exit(1)


def run_monitoring_flow():
    """Entrypoint for monitoring flow CLI."""
    import sys

    logger.info("Running monitoring flow...")
    result = monitoring_flow()

    if result["should_retrain"]:
        logger.warning("⚠️ Drift detected - retraining recommended")
        sys.exit(2)  # Exit code 2 indicates retraining needed
    elif result["has_new_data"]:
        logger.info("✓ Monitoring complete - no issues detected")
        sys.exit(0)
    else:
        logger.info("✓ No new data available")
        sys.exit(0)


if __name__ == "__main__":
    # For testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        run_training_flow()
    elif len(sys.argv) > 1 and sys.argv[1] == "monitor":
        run_monitoring_flow()
    else:
        print("Usage: python flows.py [train|monitor]")
        sys.exit(1)
