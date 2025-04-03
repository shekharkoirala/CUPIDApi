# Cupid API

This project implements a Cupid API that matches the romms names ( suppliers data ) based on the reference data ( Nuitee data )
sample data of the room names:
```
------------------- supplier data -------------
Junior Suite, 1 King Bed (Moss)
Junior Suite, View (Lava View)
Junior Suite, View (Lava View)
Junior Suite, 2 Twin Beds (Moss)
Moss Junior Suite Twin
Lava View Junior Suite Twin
Moss Junior Suite
Lava View Junior Suite
------------------- reference data -------------
Moss Junior Suite Twin
Lava View Junior Suite Twin
Moss Junior Suite
Lava View Junior Suite
------------------- result-------------
--- supplier room name --- reference room name --- similarity score ---

['junior suite 1 king bed moss', 'moss junior suite', 0.7651753271023232]
['junior suite 2 twin beds moss', 'moss junior suite twin', 0.8049020736118273]
['junior suite view lava view', 'lava view junior suite', 0.8544829947966108]
['lava view junior suite', 'lava view junior suite', 1.0]
['moss junior suite twin', 'moss junior suite twin', 1.0000000000000002]
['moss junior suite', 'moss junior suite', 1.0]
['lava view junior suite twin', 'lava view junior suite twin', 1.0]
```


# DEMO [ hosted in Nvidia Jetson Orin ]:
- [Demo](https://8075-80-233-34-169.ngrok-free.app/docs#/default/room_match_room_match_post)
- Use the API key provided in the email.
- Update the threshold in the request body if you want. ( it is also configurable in the [config.yaml](./app/config/config.yaml) )
- You might have to wait for a few minutes to get the response. (serious optimization is needed)

## Documentation

- How to run the model locally: [How to run the model locally](./docs/How%20to%20run%20CUPID%20API.pdf)
- Training dataset creation: [Training dataset creation](./docs/Training%20Dataset%20Creation%20Documentation.pdf)
- Process Documentation: [Process Data Documentation](./docs/Process%20Documentation.pdf)
- Possible implementation for future use : [Possible implementation](./docs/Possible%20Implementation.pdf)



## Tech Stack

- Python 3.12
- FastAPI for the API framework
- Pydantic for data validation
- pytest for testing and coverage
- Polars for data processing
- Scikit-learn for the model training
- XGBoost for the model training
- Spacy for text similarity
- Sentence Transformers for feature extraction
- Optuna for model tuning
- Docker for deployment
- .pre-commit for code quality
- uv for package management
- ~~Shap for the explainability~~


## Project Development stages
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
│   ├── config/
│   ├── libs/
│   ├── models/
│   ├── services/
│   ├── schemas/
│   └── utils/
├── main.py
├── data/
├── docs
├── mlmodels
├── mlreports
├   |-- exp1
├   |-- exp2
├── notebooks
├── reports
└── tests
```
