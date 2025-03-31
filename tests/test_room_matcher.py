import pytest
from app.services.room_matcher import RoomMatcher
from app.models.schemas import RoomMatchRequest, RoomMatchResponse
from app.libs.cupid_models import XGBoostModel, ModelManager, VectorizerManager
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
from app.libs.cupid_features import load_vectorizer_pkl_file
import os


@pytest.fixture(scope="module")
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

    return model, vectorizer


@pytest.mark.asyncio
async def test_match_rooms(setup_model):
    # Real test data
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

    request = RoomMatchRequest(**request_data)
    result = await RoomMatcher.match_rooms(reference_catalog=request.referenceCatalog, input_catalog=request.inputCatalog, debug=request.debug)

    assert isinstance(result, RoomMatchResponse)
    # Verify we got some matches
    assert len(result.Results) > 0
    # assert len(result.UnmappedRooms) > 0
