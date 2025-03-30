from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# ... (previous schemas remain the same) ...


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float]
    status: str
    message: str


class ProcessDataRequest(BaseModel):
    force_update: bool


class ProcessDataResponse(BaseModel):
    status: str
    message: str
    data: Dict[str, Any]


class TrainModelRequest(BaseModel):
    model_params: Dict[str, Any]


class TrainModelResponse(BaseModel):
    status: str
    message: str
    model_path: str


class RoomMatchRequest(BaseModel):
    user_id: str
    preferences: Dict[str, Any]


class RoomMatchResponse(BaseModel):
    status: str
    message: str
    room_id: str


class ErrorResponse(BaseModel):
    detail: str | List[dict]
