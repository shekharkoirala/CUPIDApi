import pytest
import polars as pl
import os
from app.services.data_processor import DataProcessor
from app.utils.config_loader import ConfigLoader


@pytest.fixture
def mock_data_files(tmp_path):
    # Create test CSV files with more realistic data
    reference_data = tmp_path / "reference_data.csv"
    supplier_data = tmp_path / "supplier_data.csv"
    processed_data = tmp_path / "processed_data.json"

    # Write realistic sample data
    reference_df = pl.DataFrame(
        {
            "lp_id": range(1, 11),
            "hotel_id": ["h" + str(i) for i in range(1, 11)],
            "room_id": ["r" + str(i) for i in range(1, 11)],
            "room_name": [
                "Classic Room",
                "Superior Room",
                "Deluxe Room",
                "Suite",
                "Classic Room",
                "Superior Room",
                "Deluxe Room",
                "Suite",
                "Classic Room",
                "Superior Room",
            ],
        }
    )

    supplier_df = pl.DataFrame(
        {
            "lp_id": range(1, 11),
            "supplier_room_name": [
                "Classic Room",
                "Superior Room",
                "Deluxe Room",
                "Suite Room",
                "Classic Double",
                "Superior Double",
                "Deluxe Suite",
                "Suite",
                "Classic Twin",
                "Superior Twin",
            ],
        }
    )

    reference_df.write_csv(reference_data)
    supplier_df.write_csv(supplier_data)
    processed_data.write_text("[]")

    return {
        "reference_data": str(reference_data),
        "supplier_data": str(supplier_data),
        "processed_data": str(processed_data),
        "vectorizer": str(tmp_path / "vectorizer.pkl"),
    }


@pytest.mark.asyncio
async def test_process_data():
    _ = ConfigLoader.load_config("app/config/config.yaml")
    result = await DataProcessor.process_data(force_update=True)
    assert "processed_path" in result
    assert os.path.exists(result["processed_path"])
