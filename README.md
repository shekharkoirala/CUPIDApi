# Cupid API

This project implements a Cupid API that matches the romms names ( suppliers data ) based on the reference data ( Nuitee data )

## Features

- Data processing to clean and prepare the data for the matching
- Model training pipeline to train the matching model
- API endpoints for matching the rooms names
- Explainability of the matching model ( TBD ) 

## Tech Stack

- Python 3.13+
- FastAPI for the API framework
- Pydantic for data validation
- pytest for testing
- Polars for data processing
- Scikit-learn for the model training
- XGBoost for the model training
- Shap for the explainability

## Setup Instructions

1. Create and activate a virtual environment:
Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # On Unix/macOS
# or other platform: https://docs.astral.sh/uv/getting-started/installation/ 
```

2. Install dependencies:
```bash
uv sync # install the dependencies
```

3. Set up environment variables: (TBD)
```bash
cp .env.example .env
# Edit .env with your configuration
```


4. Run the development server:
```bash
uvicorn app.main:app --reload
```

## Project Documentation

### Project Development stages

1. changelog file: [changelog.md](changelog.md)


## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
cupid_api/
├── app/
│   ├── api/
│   │   └── v1/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   └── services/
├── tests/
├── data/
└── docs/
```