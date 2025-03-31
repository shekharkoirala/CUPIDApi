import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    # Setup test environment variables
    os.environ["API_KEY"] = "test-key"

    # Create test directories
    test_data_dir = tmp_path / "test_data"
    test_data_dir.mkdir()

    load_dotenv()
    # Set matplotlib to use non-interactive backend
    matplotlib.use("Agg")

    # Ensure required files exist
    required_files = [
        "data/referance_rooms-1737378184366.csv",
        "data/updated_core_rooms.csv",
        "data/processed_data.json",
        "data/vectorizer.pkl",
        "mlmodels/xgb_model.json",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            pytest.skip(f"Required file {file_path} not found")

    yield

    # Cleanup is handled automatically by tmp_path


@pytest.fixture(autouse=True)
def setup_test_env_old():
    # Setup test environment
    os.environ["API_KEY"] = "test-key"

    # Create test directories
    os.makedirs("tests/fixtures", exist_ok=True)

    yield

    # Cleanup after tests
    if os.path.exists("tests/fixtures"):
        for file in os.listdir("tests/fixtures"):
            os.remove(f"tests/fixtures/{file}")
        os.rmdir("tests/fixtures")
