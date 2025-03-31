import optuna
from optuna.trial import Trial
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, Any, Tuple
import yaml
import json
import os
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger


class ModelTuner:
    def __init__(self):
        self.config = ConfigLoader.get_config()
        self.logger = CustomLogger.get_logger()
        self.best_params: Dict[str, Any] = {}
        self.best_score = 0.0

    def objective(self, trial: Trial, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
        param = {
            "objective": "binary:logistic",
            "eval_metric": self.config["model_configs"]["xgb"]["fixed_params"]["eval_metric"],
            "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            "random_state": self.config["random_state"],
        }

        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score = (accuracy + f1) / 2  # Combined metric

        if score > self.best_score:
            self.best_score = score
            self.best_params = param
            # Save the best model
            model.save_model(self.config["model_configs"]["xgb"]["model_path"])

        return score

    def tune_hyperparameters(
        self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, n_trials: int = 100
    ) -> Tuple[Dict[str, Any], float]:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_train, X_test, y_train, y_test), n_trials=n_trials)

        # Update config with best parameters
        self.config["model_configs"]["xgb"]["search_params"] = {
            "n_estimators": [self.best_params["n_estimators"]],
            "learning_rate": [self.best_params["learning_rate"]],
            "max_depth": [self.best_params["max_depth"]],
            "min_child_weight": [self.best_params["min_child_weight"]],
            "gamma": [self.best_params["gamma"]],
            "subsample": [self.best_params["subsample"]],
            "colsample_bytree": [self.best_params["colsample_bytree"]],
            "reg_alpha": [self.best_params["reg_alpha"]],
            "reg_lambda": [self.best_params["reg_lambda"]],
        }

        # Save updated config
        config_path = "app/config/config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Save study results
        study_results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study_trials": [{"number": trial.number, "params": trial.params, "value": trial.value} for trial in study.trials],
        }

        # Create reports directory if it doesn't exist
        os.makedirs(self.config["model_configs"]["xgb"]["report_path"], exist_ok=True)

        # Save study results
        study_path = os.path.join(self.config["model_configs"]["xgb"]["report_path"], "optuna_study_results.json")
        with open(study_path, "w") as f:
            json.dump(study_results, f, indent=4)

        self.logger.info(f"Best trial score: {self.best_score}")
        self.logger.info(f"Best parameters: {self.best_params}")

        return self.best_params, self.best_score
