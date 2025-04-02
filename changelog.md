# Changelog
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
