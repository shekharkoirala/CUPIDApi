import xgboost as xgb
from pathlib import Path
from fastapi import HTTPException


class ModelLoader:
    _model = None

    @classmethod
    def load_model(cls, model_path: Path):
        try:
            cls._model = xgb.Booster()
            cls._model.load_model(str(model_path))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    @classmethod
    def get_model(cls):
        if cls._model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        return cls._model
