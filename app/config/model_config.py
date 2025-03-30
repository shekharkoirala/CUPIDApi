from pathlib import Path


class ModelConfig:
    MODEL_PATH = Path("models/xgboost_model.json")
    FEATURE_COLUMNS = [
        # Add your feature columns here
        "feature1",
        "feature2",
        "feature3",
    ]
