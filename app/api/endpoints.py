from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from app.models.schemas import (
    ProcessDataRequest,
    ProcessDataResponse,
    TrainModelRequest,
    TrainModelResponse,
    RoomMatchRequest,
    RoomMatchResponse,
)
from app.services.data_processor import DataProcessor
from app.services.model_trainer import ModelTrainer
from app.services.room_matcher import RoomMatcher
from fastapi import Depends
from app.utils.auth import verify_api_key
from app.api.response_docs import process_data_responses

load_dotenv()

router = APIRouter()


@router.post(
    "/processData", response_model=ProcessDataResponse, responses=process_data_responses
)
async def process_data(
    request: ProcessDataRequest, api_key: str = Depends(verify_api_key)
):
    try:
        processed_data = await DataProcessor.process_data(request.force_update)
        return ProcessDataResponse(
            status="success", message="Data processed successfully", data=processed_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trainModel", response_model=TrainModelResponse)
async def train_model(
    request: TrainModelRequest, api_key: str = Depends(verify_api_key)
):
    try:
        training_result = await ModelTrainer.train_model(request.model_params)
        return TrainModelResponse(
            status="success",
            message="Model trained successfully",
            model_params=training_result,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/room_match", response_model=RoomMatchResponse)
async def room_match(request: RoomMatchRequest, api_key: str = Depends(verify_api_key)):
    try:
        if not request.user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        if not request.preferences:
            raise HTTPException(status_code=400, detail="Preferences are required")
        matched_room = await RoomMatcher.match_room(
            request.user_id, request.preferences
        )
        return RoomMatchResponse(
            status="success",
            message="Room matching completed",
            user_id=request.user_id,
            matched_room=matched_room,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
