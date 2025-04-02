import traceback
from typing import Any, Dict, List, Optional
from app.models.schemas import RoomMatchResponse, ReferenceProperty, SupplierCatalog
from app.libs.cupid_models import ModelManager, VectorizerManager
from app.libs.cupid_features import get_feature_inference
from app.utils.config_loader import ConfigLoader
from app.utils.helper import normalize_room_name
from app.utils.logger import CustomLogger


class RoomMatcher:
    @staticmethod
    async def match_rooms(
        reference_catalog: list[ReferenceProperty], input_catalog: list[SupplierCatalog], debug: bool = False, threshold: Optional[float] = None
    ) -> RoomMatchResponse:
        config = ConfigLoader.get_config()
        logger = CustomLogger.get_logger()
        model = ModelManager.get_model()
        if model is None or model.model is None:
            raise ValueError("Model not initialized")

        # Use provided threshold or fall back to config
        match_threshold = threshold if threshold is not None else config["model_configs"]["xgb"]["fixed_params"]["threshold"]
        logger.info(f"Using matching threshold: {match_threshold}")

        vectorizer = VectorizerManager.get_vectorizer()
        reference_room_names = [room.roomName for room in reference_catalog[0].referenceRoomInfo]
        supplier_room_names = [room.supplierRoomName for room in input_catalog[0].supplierRoomInfo]

        supplier_map: Dict[str, Dict[str, Any]] = {
            normalize_room_name(supplier.supplierRoomName): {
                "supplierRoomId": supplier.supplierRoomId,
                "supplierRoomName": supplier.supplierRoomName,
                "supplierId": input_catalog[0].supplierId,
                "cleanSupplierRoomName": supplier.supplierRoomName,
            }
            for supplier in input_catalog[0].supplierRoomInfo
        }

        reference_map: Dict[str, Dict[str, Any]] = {
            normalize_room_name(reference.roomName): {
                "roomId": reference.roomId,
                "roomName": reference.roomName,
                "cleanRoomName": reference.roomName,
                "propertyId": reference_catalog[0].propertyId,
            }
            for reference in reference_catalog[0].referenceRoomInfo
        }

        Results: List[Dict[str, Any]] = []
        UnmappedRooms: List[Dict[str, Any]] = []

        try:
            logger.info(f"----------------------------------Matching {supplier_room_names}----------------------------------")
            for supplier_room_name in supplier_room_names:
                best_match: str = ""
                best_score: float = 0.0
                logger.info(f"----------------------------------Matching {normalize_room_name(supplier_room_name)}----------------------------------")
                logger.info(f"Reference Room Names: {reference_room_names}")
                for reference_room_name in reference_room_names:
                    features = get_feature_inference(ref_name=reference_room_name, sup_name=supplier_room_name, vectorizer=vectorizer)
                    match_prob: float = model.model.predict_proba(features)[0, 1]
                    if match_prob >= best_score:
                        best_score = match_prob
                        best_match = normalize_room_name(reference_room_name)

                    logger.info(f"{normalize_room_name(supplier_room_name)} ->/t-> {normalize_room_name(reference_room_name)} score: {match_prob}")

                logger.info(
                    f"Best Match: {best_match} score: {best_score} {best_score >= match_threshold} of {normalize_room_name(supplier_room_name)}"
                )
                if best_score >= match_threshold:
                    mapped_Result = reference_map[best_match].copy()
                    matched_result = supplier_map[normalize_room_name(supplier_room_name)].copy()
                    matched_result["score"] = best_score
                    mapped_Result["mappedRooms"] = [matched_result]
                    Results.append(mapped_Result)
                else:
                    UnmappedRooms.append(supplier_map[normalize_room_name(supplier_room_name)])

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error matching rooms: {e}")
            raise e

        return RoomMatchResponse(Results=Results, UnmappedRooms=UnmappedRooms)
