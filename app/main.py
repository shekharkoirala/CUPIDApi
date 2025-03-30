from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.utils.model_loader import ModelLoader
from app.config.model_config import ModelConfig
from contextlib import asynccontextmanager
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger

# Load config at startup
config = ConfigLoader.load_config("app/config/config.yaml")
logger = CustomLogger.setup_logger(config["log_level"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    if ModelConfig.MODEL_PATH.exists():
        ModelLoader.load_model(ModelConfig.MODEL_PATH)
    else:
        print("Warning: No model file found. Please train the model first.")
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

app.include_router(router)
