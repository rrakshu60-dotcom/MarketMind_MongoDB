"""Central configuration for MarketMind AI (MongoDB edition).

All secrets come from environment variables or a local ``.env`` file.
Nothing sensitive is hard-coded here.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "MarketMindDB")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "5000"))
ML_MARKET = os.getenv("ML_MARKET", "US")

DATASET_PATH = BASE_DIR / "dataset" / "global_superstore_sales.csv"
MODELS_DIR = BASE_DIR / "models"
REFERENCE_DIR = MODELS_DIR / "reference"
RESULTS_DIR = BASE_DIR / "results"

# Collection names
SALES_TRANSACTIONS = "sales_transactions"
SALES_ANALYSIS = "sales_analysis"
SALES_FORECASTS = "sales_forecasts"
CUSTOMER_SEGMENTS = "customer_segments"
CHURN_PREDICTIONS = "churn_predictions"
SALES_ANOMALIES = "sales_anomalies"
PRODUCT_RECOMMENDATIONS = "product_recommendations"
IMPORT_LOG = "import_log"
