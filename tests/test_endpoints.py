import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from dotenv import load_dotenv
from app.libs.cupid_models import XGBoostModel, ModelManager, VectorizerManager
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
from app.libs.cupid_features import load_vectorizer_pkl_file

load_dotenv()
client = TestClient(app)
API_KEY = os.getenv("API_KEY")


def test_process_data_endpoint():
    response = client.post("/processData", json={"force_update": True}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_train_model_endpoint():
    response = client.post("/trainModel", json={"parameter_tuning": False}, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert "task_id" in response.json()


@pytest.fixture(scope="module", autouse=True)
def setup_model():
    # Load config
    config = ConfigLoader.load_config("app/config/config.yaml")
    logger = CustomLogger.setup_logger(config["log_level"])

    # Initialize model
    model_path = config["model_configs"]["xgb"]["model_path"]
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}")

    model = XGBoostModel(config, logger)
    model.load_model()
    ModelManager.set_model(model)

    # Initialize vectorizer
    vectorizer = load_vectorizer_pkl_file()
    VectorizerManager.set_vectorizer(vectorizer)


def test_room_match_endpoint():
    request_data = {
        "debug": True,
        "referenceCatalog": [
            {
                "propertyId": "5122906",
                "referenceRoomInfo": [
                    {"roomId": "512290602", "roomName": "Classic Room"},
                    {"roomId": "512290603", "roomName": "Superior Room"},
                    {"roomId": "512290604", "roomName": "Superior Room with City View"},
                    {"roomId": "512290605", "roomName": "Balcony Room"},
                ],
            }
        ],
        "inputCatalog": [
            {
                "supplierId": "nuitee",
                "supplierRoomInfo": [
                    {"supplierRoomId": "2", "supplierRoomName": "Classic Room - Olympic Queen Bed - ROOM ONLY"},
                    {"supplierRoomId": "3", "supplierRoomName": "CLASSIC ROOM ADA - ROOM ONLY"},
                    {"supplierRoomId": "10", "supplierRoomName": "Superior Room - Olympic Queen Bed - ROOM ONLY"},
                ],
            }
        ],
    }

    response = client.post("/room_match", json=request_data, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert "Results" in response.json()
