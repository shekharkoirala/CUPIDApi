from fastapi import status
from app.models.schemas import ProcessDataResponse, ErrorResponse
from typing import Dict, Any

# Common error responses that can be reused across endpoints
common_error_responses = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorResponse,
        "description": "Unauthorized - Invalid or missing API key",
        "content": {"application/json": {"example": {"detail": "Invalid API Key"}}},
    },
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [{"loc": ["string"], "msg": "string", "type": "string"}]
                }
            }
        },
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {"detail": "Internal server error occurred"}
            }
        },
    },
}

process_data_responses = {
    status.HTTP_200_OK: {
        "model": ProcessDataResponse,
        "description": "Successfully processed data",
        "content": {
            "application/json": {
                "example": {
                    "status": "success",
                    "message": "Data processed successfully",
                    "data": {"processed_path": "path/to/file"},
                }
            }
        },
    },
    **common_error_responses,  # Include all common error responses
}

# Add other endpoint responses here
train_model_responses: Dict[str, Any] = {
    # Similar structure for train_model endpoint
}

room_match_responses: Dict[str, Any] = {
    # Similar structure for room_match endpoint
}
