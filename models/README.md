# Trained models (reused, not retrained)

Copied unchanged from the MarketMind AI internship repository
(Small-business-sales-intelligence-platform-SB7-Team4, `AIML/week3/`):

| File | Source | Model |
|---|---|---|
| `catboost_daily_model.pkl` | `forecasting/catboost/` | CatBoostRegressor on log(1 + daily sales), 25 lag features |
| `kmeans_clustering_model.pkl` | `customer_grouping/` | scikit-learn KMeans, k = 3 |
| `churn_model.pkl` | `churn_prediction/` | scikit-learn RandomForestClassifier, 11 features |

`reference/` holds the outputs the internship notebooks saved. The tests compare this project's
MongoDB-based results against them.

The pickles were saved with scikit-learn 1.9, so install scikit-learn 1.9.x (see requirements.txt).
They are loaded through `analysis/model_loader.py`, which only lets a file rebuild objects from
the libraries listed there.
