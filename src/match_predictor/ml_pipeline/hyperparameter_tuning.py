"""Hyperparameter tuning module using Optuna."""

import optuna
from loguru import logger
from sklearn.model_selection import cross_val_score

from match_predictor.config import ModelConfig


class HyperparameterTuner:
    """Class for hyperparameter tuning using Optuna."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize hyperparameter tuner.

        Args:
            config: Model configuration (optional, defaults to ModelConfig())
        """
        self.config = config or ModelConfig()
        self.tuning_config = self.config.hyperparameter_tuning
        self.logger = logger
        self.best_params = None
        self.study = None

    def objective(self, trial: optuna.Trial, model_class, X, y) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object
            model_class: ML model class to optimize
            X: Feature matrix
            y: Target vector

        Returns:
            Mean cross-validation score
        """
        # Define hyperparameter search space for XGBoost
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 5000, step=100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        # Add base parameters from config
        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "enable_categorical": True,
            "device": self.tuning_config.device,
            "seed": self.tuning_config.random_state,
            "verbosity": 0,
        }
        params.update(base_params)

        # Create model with suggested parameters
        model = model_class(**params)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=self.tuning_config.cv_folds, scoring="accuracy", n_jobs=-1)

        return cv_scores.mean()

    def tune(self, model_class, X, y) -> dict:
        """Perform hyperparameter tuning.

        Args:
            model_class: ML model class to optimize
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info("Starting hyperparameter tuning with Optuna...")

        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.tuning_config.random_state)
        )

        # Run optimization
        self.study.optimize(
            lambda trial: self.objective(trial, model_class, X, y),
            n_trials=self.tuning_config.n_trials,
            timeout=self.tuning_config.timeout,
            show_progress_bar=True,
        )

        self.best_params = self.study.best_params
        self.logger.info(f"Best hyperparameters found: {self.best_params}")
        self.logger.info(f"Best cross-validation accuracy: {self.study.best_value:.4f}")

        # Add base parameters
        best_params_full = {
            **self.best_params,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "enable_categorical": True,
            "seed": self.tuning_config.random_state,
        }

        return best_params_full

    def get_optimization_history(self) -> list[dict]:
        """Get optimization history.

        Returns:
            List of dictionaries containing trial information
        """
        if self.study is None:
            return []

        history = []
        for trial in self.study.trials:
            history.append(
                {"trial_number": trial.number, "value": trial.value, "params": trial.params, "state": trial.state.name}
            )

        return history

    def get_best_trial_info(self) -> dict:
        """Get information about the best trial.

        Returns:
            Dictionary with best trial information
        """
        if self.study is None:
            return {}

        best_trial = self.study.best_trial
        return {
            "trial_number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "duration": best_trial.duration.total_seconds() if best_trial.duration else None,
        }
