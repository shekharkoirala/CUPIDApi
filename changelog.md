# Changelog

[2025-04-03]
- added process documentation
- added possible implementation
- added changelog
- added README


[2025-04-02]
- Dataset creation update ( using spacy )
- Dataset categorization ( using llm ) for better training data
- generated room hierarchy and room features and abbreviations
- TODO: Process Documentation


[2025-04-01]
- Docker check for reproducibility
- Background tasks for the model training using fastapi.
- Quick documentation for the reproducibility


[2025-03-31]
- Optimize the feature extraction : combination of numerical and sentence transformers
- evaluation and testing of the model
- LLM can be used ( discard as the project explicitly asks for the model training )
- xgboost model is optimized, since the training data is not accurate, performance is not good.


[2025-03-30]
- added parameter tuning to the model training
- added api
- added test cases
- tested inferences with the test data
- TODO: update hyper paramter tuning
- TODO: update Documentations
- TODO: use LLM instead of xgboost ?
- TODO: TEST in large scale data
- TODO: Dockerfile & wrapping


[2025-03-29]
- added pre-commit hooks
- experiment on the model training in the notebook : precision above 0.99 , using custom feature
- Tried sentence transformers for the embedding , but the performance is not good as expected
- used standard scalar
- used xgboost as the model
- used feature selector in the embedding but it didn't improve the performance ( slightly expected )
- Tested the model with the test data (Web)
- TODO: umap/PCA for the embedding
- TODO: notebook to script for the model training [ done ]
- TODO: API for the model inference [ done ]


[2024-03-28]
- Initial tfidf matching system for creating training data
- based on Notebooks/eda2.ipynb , find the merged data based on hotel_id and lp_id , picked lp_id as the reference id
- changed the python version to 3.12 = shap doesnot support 3.13
- added initial project structure
