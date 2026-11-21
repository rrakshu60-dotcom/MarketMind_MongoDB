"""Analysis 6: customers. High-value customers, K-Means segments, churn risk.

Customer-level features are built by one MongoDB aggregation. The existing
trained models from the internship project (``kmeans_clustering_model.pkl``
and ``churn_model.pkl``) are then applied to those features. The feature
definitions copy the original notebooks exactly
(``AIML/week3/customer_grouping`` and ``AIML/week3/churn_prediction``), so the
models see the inputs they were trained on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database
from sklearn.preprocessing import StandardScaler

import config
from analysis.common import aggregate_df, records, replace_collection, scope_query, utcnow
from analysis.model_loader import load_model

# Same names and playbooks as AIML/Integrated_API/app.py
CLUSTER_NAMES = {0: "High-Value Customers", 1: "Regular Customers", 2: "Low-Value Customers"}
CLUSTER_STRATEGY = {
    0: "VIP retention, dedicated account support, loyalty exclusivity and early product access",
    1: "Cross-sell relevant bundles and targeted volume discounts to raise average order value",
    2: "Reactivation emails, flash discounts and entry-level promotion bundles",
}
SEGMENT_FEATURES = ["TotalSpending", "PurchaseFrequency", "AverageOrderValue"]
CHURN_FEATURES = [
    "PurchaseFrequency", "TotalSpending", "AverageOrderValue", "CustomerLifespanDays",
    "AvgDaysBetweenOrders", "TotalProfit", "AvgQuantity", "RecencyVsAvgGap",
    "OrderRatePerMonth", "SpendPerMonth", "ProfitMargin",
]
CHURN_THRESHOLD = 0.52  # decision threshold used by the original notebook and API
CHURN_INACTIVITY_DAYS = 180  # the notebook's definition of an actual churner


def customer_summary(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """One row per customer with spending and timing facts, computed in MongoDB."""
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {
            "_id": "$customer_id",
            "customer_name": {"$first": "$customer_name"},
            "segment": {"$first": "$segment"},
            "country": {"$first": "$country"},
            "total_sales": {"$sum": "$sales"},
            "total_profit": {"$sum": "$profit"},
            "line_items": {"$sum": 1},
            "avg_line_value": {"$avg": "$sales"},
            "avg_quantity": {"$avg": "$quantity"},
            "total_quantity": {"$sum": "$quantity"},
            "orders": {"$addToSet": "$order_id"},
            "first_purchase": {"$min": "$order_date"},
            "last_purchase": {"$max": "$order_date"},
        }},
        {"$project": {
            "_id": 0, "customer_id": "$_id", "customer_name": 1, "segment": 1, "country": 1,
            "total_sales": 1, "total_profit": 1, "line_items": 1, "avg_line_value": 1,
            "avg_quantity": 1, "total_quantity": 1, "order_count": {"$size": "$orders"},
            "first_purchase": 1, "last_purchase": 1,
        }},
        {"$sort": {"total_sales": -1}},
    ]
    df = aggregate_df(pipeline, db)
    if not df.empty:
        df["first_purchase"] = pd.to_datetime(df["first_purchase"])
        df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    return df


def high_value_customers(n: int = 10, market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """Top-N customers by revenue, computed fully in MongoDB."""
    if n < 1:
        raise ValueError("n must be at least 1")
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {
            "_id": "$customer_id",
            "customer_name": {"$first": "$customer_name"},
            "segment": {"$first": "$segment"},
            "country": {"$first": "$country"},
            "total_sales": {"$sum": "$sales"},
            "total_profit": {"$sum": "$profit"},
            "orders": {"$addToSet": "$order_id"},
            "last_purchase": {"$max": "$order_date"},
        }},
        {"$sort": {"total_sales": -1}},
        {"$limit": int(n)},
        {"$project": {
            "_id": 0, "customer_id": "$_id", "customer_name": 1, "segment": 1, "country": 1,
            "total_sales": {"$round": ["$total_sales", 2]},
            "total_profit": {"$round": ["$total_profit", 2]},
            "order_count": {"$size": "$orders"},
            "avg_order_value": {"$round": [{"$divide": ["$total_sales", {"$size": "$orders"}]}, 2]},
            "last_purchase": 1,
        }},
    ]
    return aggregate_df(pipeline, db)


# ---------------------------------------------------------------------------
# K-Means segmentation (existing model)
# ---------------------------------------------------------------------------

def segmentation_features(summary: pd.DataFrame) -> pd.DataFrame:
    """Features as in customer_grouping.ipynb: PurchaseFrequency is the count of
    line items (``Order ID`` count), AverageOrderValue the mean line amount."""
    return pd.DataFrame({
        "Customer ID": summary["customer_id"],
        "TotalSpending": summary["total_sales"],
        "PurchaseFrequency": summary["line_items"],
        "AverageOrderValue": summary["avg_line_value"],
    })


def segment_customers(market: Optional[str] = None, db: Optional[Database] = None,
                      summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    market = market or config.ML_MARKET
    summary = summary if summary is not None else customer_summary(market, db)
    if summary.empty:
        return pd.DataFrame()
    feats = segmentation_features(summary)
    # The notebook fitted a StandardScaler on the same customer table before
    # training K-Means, so we do the same here and then call the saved model.
    scaled = StandardScaler().fit_transform(feats[SEGMENT_FEATURES])
    kmeans = load_model("kmeans")
    clusters = kmeans.predict(scaled)
    out = feats.rename(columns={"Customer ID": "customer_id"}).copy()
    out["customer_name"] = summary["customer_name"].values
    out["cluster"] = clusters.astype(int)
    out["segment_name"] = out["cluster"].map(CLUSTER_NAMES)
    out["strategy"] = out["cluster"].map(CLUSTER_STRATEGY)
    return out


def segment_profile(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return segments
    prof = (segments.groupby(["cluster", "segment_name"])
            .agg(customers=("customer_id", "count"),
                 avg_total_spending=("TotalSpending", "mean"),
                 avg_purchase_frequency=("PurchaseFrequency", "mean"),
                 avg_order_value=("AverageOrderValue", "mean"),
                 total_revenue=("TotalSpending", "sum"))
            .reset_index().round(2))
    return prof


# ---------------------------------------------------------------------------
# Churn prediction (existing Random Forest model)
# ---------------------------------------------------------------------------

def churn_features(summary: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Feature engineering copied from random_forest_churn.ipynb."""
    ref = reference_date if reference_date is not None else summary["last_purchase"].max()
    df = pd.DataFrame({
        "customer_id": summary["customer_id"],
        "PurchaseFrequency": summary["order_count"],
        "TotalSpending": summary["total_sales"],
        "AverageOrderValue": summary["avg_line_value"],
        "TotalProfit": summary["total_profit"],
        "AvgQuantity": summary["avg_quantity"],
        "FirstPurchaseDate": summary["first_purchase"],
        "LastPurchaseDate": summary["last_purchase"],
    })
    df["DaysSinceLastPurchase"] = (ref - df["LastPurchaseDate"]).dt.days
    df["CustomerLifespanDays"] = (df["LastPurchaseDate"] - df["FirstPurchaseDate"]).dt.days
    df["AvgDaysBetweenOrders"] = np.where(
        df["PurchaseFrequency"] > 1,
        df["CustomerLifespanDays"] / (df["PurchaseFrequency"] - 1),
        df["DaysSinceLastPurchase"],
    )
    df["RecencyVsAvgGap"] = df["DaysSinceLastPurchase"] / (df["AvgDaysBetweenOrders"] + 1)
    df["OrderRatePerMonth"] = df["PurchaseFrequency"] / ((df["CustomerLifespanDays"] + 30) / 30)
    df["SpendPerMonth"] = df["TotalSpending"] / ((df["CustomerLifespanDays"] + 30) / 30)
    df["ProfitMargin"] = df["TotalProfit"] / (df["TotalSpending"] + 1e-5)
    df["ActualChurn"] = (df["DaysSinceLastPurchase"] > CHURN_INACTIVITY_DAYS).astype(int)
    return df


def risk_label(prob: float) -> str:
    if prob >= CHURN_THRESHOLD:
        return "High Risk"
    if prob >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def predict_churn(market: Optional[str] = None, db: Optional[Database] = None,
                  summary: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    market = market or config.ML_MARKET
    summary = summary if summary is not None else customer_summary(market, db)
    if summary.empty:
        return pd.DataFrame()
    feats = churn_features(summary)
    model = load_model("churn")
    probs = model.predict_proba(feats[CHURN_FEATURES])[:, 1]
    out = feats.copy()
    out["customer_name"] = summary["customer_name"].values
    out["churn_probability"] = probs.round(4)
    out["predicted_churn"] = (probs >= CHURN_THRESHOLD).astype(int)
    out["risk_level"] = [risk_label(p) for p in probs]
    return out


def churn_summary(churn: pd.DataFrame) -> dict:
    if churn.empty:
        return {}
    actual, pred = churn["ActualChurn"], churn["predicted_churn"]
    return {
        "customers": int(len(churn)),
        "high_risk": int((churn["risk_level"] == "High Risk").sum()),
        "medium_risk": int((churn["risk_level"] == "Medium Risk").sum()),
        "low_risk": int((churn["risk_level"] == "Low Risk").sum()),
        "agreement_with_180_day_rule_pct": round(100 * float((actual == pred).mean()), 2),
        "note": ("The model's churn label is 'no purchase in the last 180 days', and the feature "
                 "RecencyVsAvgGap is built from the same recency value, so agreement with that rule "
                 "is high by construction. Treat this as a risk ranking, not an independent forecast."),
    }


# ---------------------------------------------------------------------------
# Persist to MongoDB
# ---------------------------------------------------------------------------

def store_segments(segments: pd.DataFrame, market: str, db: Optional[Database] = None) -> int:
    docs = records(segments)
    now = utcnow()
    for d in docs:
        d.update({"market_scope": market, "model": "KMeans (k=3) kmeans_clustering_model.pkl", "generated_at": now})
    return replace_collection(config.CUSTOMER_SEGMENTS, docs, db)


def store_churn(churn: pd.DataFrame, market: str, db: Optional[Database] = None) -> int:
    cols = ["customer_id", "customer_name", "churn_probability", "predicted_churn", "risk_level",
            "ActualChurn", "DaysSinceLastPurchase", "PurchaseFrequency", "TotalSpending",
            "LastPurchaseDate"]
    docs = records(churn[cols])
    now = utcnow()
    for d in docs:
        d.update({"market_scope": market, "model": "RandomForestClassifier churn_model.pkl",
                  "threshold": CHURN_THRESHOLD, "generated_at": now})
    return replace_collection(config.CHURN_PREDICTIONS, docs, db)
