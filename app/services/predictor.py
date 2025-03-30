import numpy as np
import xgboost as xgb
from typing import Dict
from app.utils.model_loader import ModelLoader
from app.config.model_config import ModelConfig


class Predictor:
    @staticmethod
    async def predict(features: Dict[str, float]) -> Dict[str, float]:
        model = ModelLoader.get_model()

        # Convert features to array in correct order
        feature_array = []
        for col in ModelConfig.FEATURE_COLUMNS:
            if col not in features:
                raise ValueError(f"Missing feature: {col}")
            feature_array.append(features[col])

        # Create DMatrix for prediction
        dtest = xgb.DMatrix([feature_array])

        # Make prediction
        prediction = model.predict(dtest)

        return {
            "prediction": float(prediction[0]),
            "probability": float(
                np.clip(prediction[0], 0, 1)
            ),  # Only if it's a probability
        }
