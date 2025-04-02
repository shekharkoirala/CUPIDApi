from fastapi import APIRouter, HTTPException, Depends
from dotenv import load_dotenv
from app.models.schemas import (
    ProcessDataRequest,
    ProcessDataResponse,
    TrainModelRequest,
    TrainModelResponse,
    RoomMatchRequest,
    RoomMatchResponse,
    TrainModelStatusResponse,
)
from app.services.data_processor import DataProcessor
from app.services.model_trainer import ModelTrainer
from app.services.room_matcher import RoomMatcher
from app.utils.auth import verify_api_key
from app.api.response_docs import process_data_responses
from uuid import uuid4
from fastapi import BackgroundTasks
from typing import Dict, Optional
import asyncio
import traceback
from app.utils.config_loader import ConfigLoader

load_dotenv()

router = APIRouter()

# Dictionary to store training status (In production, use Redis or a database)
training_tasks: Dict[str, Dict[str, Optional[str]]] = {}


@router.post("/processData", response_model=ProcessDataResponse, responses=process_data_responses)
async def process_data(request: ProcessDataRequest, api_key: str = Depends(verify_api_key)):
    try:
        processed_data = await DataProcessor.process_data(request.force_update)
        return ProcessDataResponse(status="success", message="Data processed successfully", data=processed_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trainModel", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)):
    try:
        # check if already running background task
        if any(task["status"] == "processing" for task in training_tasks.values()):
            raise HTTPException(status_code=429, detail="Training already in progress, please wait for it to complete")

        task_id = str(uuid4())
        training_tasks[task_id] = {"status": "processing", "model_path": None, "error": None}

        background_tasks.add_task(start_training_in_background, task_id, request.parameter_tuning)

        return TrainModelResponse(status="accepted", message="Model training started", task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trainModel/{task_id}", response_model=TrainModelStatusResponse)
async def get_training_status(task_id: str, api_key: str = Depends(verify_api_key)):
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")

    task = training_tasks[task_id]
    return TrainModelStatusResponse(status=task["status"], message="Training status retrieved", model_path=task["model_path"], error=task["error"])


def start_training_in_background(task_id: str, parameter_tuning: bool):
    asyncio.run(run_training(task_id, parameter_tuning))


async def run_training(task_id: str, parameter_tuning: bool):
    try:
        training_tasks[task_id]["status"] = "processing"
        trainer = ModelTrainer()
        model_path = await trainer.start_training(parameter_tuning)
        training_tasks[task_id].update({"status": "completed", "model_path": model_path})
    except Exception as e:
        print(f"Error training model: {e}")
        print(traceback.format_exc())
        training_tasks[task_id].update({"status": "failed", "error": str(e)})


@router.post("/room_match", response_model=RoomMatchResponse)
async def room_match(request: RoomMatchRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Get threshold from request or config
        config = ConfigLoader.get_config()
        threshold = request.threshold if request.threshold is not None else config["model_configs"]["xgb"]["fixed_params"]["threshold"]

        results = await RoomMatcher.match_rooms(
            reference_catalog=request.referenceCatalog, input_catalog=request.inputCatalog, debug=request.debug, threshold=threshold
        )
        return results  # RoomMatchResponse
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
