from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
from app.libs.cupid_models import XGBoostModel, ModelManager, VectorizerManager
from app.libs.cupid_features import load_vectorizer_pkl_file
import os


def reload_model() -> bool:
    """Reload the model and vectorizer"""
    try:
        config = ConfigLoader.get_config()
        logger = CustomLogger.get_logger()
        model_path = config["model_configs"]["xgb"]["model_path"]

        if os.path.exists(model_path):
            model = XGBoostModel(config, logger)
            model.load_model()
            ModelManager.set_model(model)
            vectorizer = load_vectorizer_pkl_file()
            VectorizerManager.set_vectorizer(vectorizer)
            logger.info(f"Model reloaded successfully from {model_path}")
            return True
        else:
            logger.error(f"Model file not found at {model_path}")
            return False
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return False
