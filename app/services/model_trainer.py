from sklearn.model_selection import train_test_split
import os
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
from app.libs.cupid_models import XGBoostModel
from app.libs.cupid_features import get_features
import json
from app.services.model_tuner import ModelTuner


class ModelTrainer:
    def __init__(self):
        self.config = ConfigLoader.get_config()
        self.logger = CustomLogger.get_logger()
        self.model = XGBoostModel(config=self.config, logger=self.logger)

    async def start_training(self, parameter_tuning: bool) -> str:
        try:
            X_train, X_test, y_train, y_test = await self.load_split_data()

            if parameter_tuning:
                tuner = ModelTuner()
                best_params, best_score = tuner.tune_hyperparameters(X_train, X_test, y_train, y_test)
                self.logger.info(f"Tuning completed. Best score: {best_score}")
                self.model.load_model()  # Load the best model saved during tuning
            else:
                self.model.train_model(X_train, X_test, y_train, y_test)

            return self.model.model_path

        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")

    async def load_split_data(self):
        # check if data exists
        try:
            data_path = self.config["data"]["processed_data"]
            if not os.path.exists(data_path):
                self.logger.error(
                    f"Data file not found at {data_path}, trigger /processData endpoint to process data and then trigger /trainModel endpoint to train model"
                )
                raise FileNotFoundError(
                    f"Data file not found at {data_path}, trigger /processData endpoint to process data and then trigger /trainModel endpoint to train model"
                )

            else:
                with open(data_path, "r") as f:
                    data = json.load(f)

                feature_matrix, labels = get_features(data)

            X_train, X_test, y_train, y_test = train_test_split(
                feature_matrix, labels, test_size=self.config["test_size"], random_state=self.config["random_state"]
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise Exception(f"Error loading split data: {str(e)}")
