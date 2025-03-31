import pytest
import numpy as np


@pytest.fixture
def mock_data():
    return {
        "X_train": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "X_test": np.array([[9, 10, 11, 12]]),
        "y_train": np.array([0, 1]),
        "y_test": np.array([1]),
    }


# @pytest.mark.asyncio
# async def test_train_model():
#     with patch('app.libs.cupid_models.XGBoostModel') as mock_xgb:
#         trainer = ModelTrainer()
#         model_path = await trainer.start_training(parameter_tuning=False)
#         assert model_path is not None
