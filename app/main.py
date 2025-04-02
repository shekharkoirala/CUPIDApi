from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from contextlib import asynccontextmanager
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
from app.libs.cupid_models import XGBoostModel, ModelManager, VectorizerManager
from app.libs.cupid_features import load_vectorizer_pkl_file
from app.utils.model_reloader import reload_model
import os

# Load config at startup
config = ConfigLoader.load_config("app/config/config.yaml")
logger = CustomLogger.setup_logger(config["log_level"])


def reloader():
    """Reload the model and vectorizer"""
    return reload_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    model_path = config["model_configs"]["xgb"]["model_path"]
    if os.path.exists(model_path):
        model = XGBoostModel(config, logger)
        model.load_model()
        ModelManager.set_model(model)
        vectorizer = load_vectorizer_pkl_file()
        VectorizerManager.set_vectorizer(vectorizer)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning("No model file found. Please train the model first.")

    # Startup
    logger.info("Starting up application...")
    yield
    # Clean up resources if needed
    # Shutdown
    logger.info("Shutting down application...")


app = FastAPI(
    title="CUPID API",
    description="API for data processing, model training, and inference for ROOM Matching",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add reload_model to app state for access in endpoints
app.state.reload_model = reloader

app.include_router(router)
