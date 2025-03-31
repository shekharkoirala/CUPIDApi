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
    parameter_tuning: bool = False


class TrainModelResponse(BaseModel):
    status: str
    message: str
    task_id: str


class TrainModelStatusResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None
    error: Optional[str] = None


class RoomInfo(BaseModel):
    roomId: str
    roomName: str


class ReferenceProperty(BaseModel):
    propertyId: str
    referenceRoomInfo: List[RoomInfo]


class SupplierRoomInfo(BaseModel):
    supplierRoomId: str
    supplierRoomName: str


class SupplierCatalog(BaseModel):
    supplierId: str
    supplierRoomInfo: List[SupplierRoomInfo]


class RoomMatchRequest(BaseModel):
    debug: bool = False
    referenceCatalog: List[ReferenceProperty]
    inputCatalog: List[SupplierCatalog]

    class Config:
        json_schema_extra = {
            "example": {
                "debug": True,
                "referenceCatalog": [
                    {
                        "propertyId": "5122906",
                        "referenceRoomInfo": [
                            {"roomId": "512290602", "roomName": "Classic Room"},
                            {"roomId": "512290603", "roomName": "Superior Room"},
                            {"roomId": "512290604", "roomName": "Superior Room with City View"},
                            {"roomId": "512290605", "roomName": "Balcony Room"},
                            {"roomId": "512290608", "roomName": "Classic Room - Disability Access"},
                            {"roomId": "512290609", "roomName": "Superior Room - Disability Access"},
                            {"roomId": "512290610", "roomName": "Junior Suite - Disability Access"},
                        ],
                    }
                ],
                "inputCatalog": [
                    {
                        "supplierId": "nuitee",
                        "supplierRoomInfo": [
                            {"supplierRoomId": "2", "supplierRoomName": "Classic Room - Olympic Queen Bed - ROOM ONLY"},
                            {"supplierRoomId": "3", "supplierRoomName": "CLASSIC ROOM ADA - ROOM ONLY"},
                            {"supplierRoomId": "5", "supplierRoomName": "SUPERIOR ROOM ADA - ROOM ONLY"},
                            {"supplierRoomId": "10", "supplierRoomName": "Superior Room - Olympic Queen Bed - ROOM ONLY"},
                            {"supplierRoomId": "6", "supplierRoomName": "Superior City View - Olympic Queen Bed - ROOM ONLY"},
                            {"supplierRoomId": "7", "supplierRoomName": "Balcony Room - Olympic Queen Bed - ROOM ONLY"},
                        ],
                    }
                ],
            }
        }


class MappedRoom(BaseModel):
    cleanSupplierRoomName: str
    score: float
    supplierId: str
    supplierRoomId: str
    supplierRoomName: str


class UnmappedRoom(BaseModel):
    cleanSupplierRoomName: str
    supplierId: str
    supplierRoomId: str
    supplierRoomName: str


class RoomMatchResult(BaseModel):
    cleanRoomName: str
    mappedRooms: List[MappedRoom]
    propertyId: str
    roomId: str
    roomName: str


class RoomMatchResponse(BaseModel):
    Results: List[RoomMatchResult]
    UnmappedRooms: List[UnmappedRoom]


class ErrorResponse(BaseModel):
    detail: str | List[dict]
