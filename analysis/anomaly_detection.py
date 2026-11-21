"""Analysis 7: unusual sales.

Two levels:

1. **Daily anomalies** reuse the internship project's method
   (``AIML/week3/anomaly_detection``): total sales per day, flagged when outside
   Q1 - 3*IQR .. Q3 + 3*IQR.
2. **Transaction anomalies** (added for this project) look at individual
   transaction amounts. Single sales are heavily right-skewed (median about
   $85, maximum about $22,600), so a plain IQR rule would flag thousands of
   ordinary large orders. The Tukey rule (1.5*IQR) is therefore applied to
   log(1 + amount), which flags only the truly unusual sales.

Both use data aggregated from MongoDB, and both are stored in ``sales_anomalies``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database

import config
from analysis.common import records, replace_collection, scope_query, transactions, utcnow
from analysis.sales_analysis import daily_sales

IQR_MULTIPLIER = 3.0


def iqr_limits(values: pd.Series, k: float = IQR_MULTIPLIER) -> dict:
    q1, q3 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q3 - q1
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": q1 - k * iqr, "upper": q3 + k * iqr,
            "mean": float(values.mean()), "std": float(values.std())}


def daily_anomalies(market: Optional[str] = None, db: Optional[Database] = None,
                    daily: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, dict]:
    market = market or config.ML_MARKET
    daily = daily if daily is not None else daily_sales(market, db)
    if daily.empty:
        return daily, {}
    lim = iqr_limits(daily["total_amount"])
    out = daily.copy()
    out["anomaly"] = (out["total_amount"] < lim["lower"]) | (out["total_amount"] > lim["upper"])
    out["z_score"] = ((out["total_amount"] - lim["mean"]) / lim["std"]).round(2)
    return out, lim


TXN_IQR_MULTIPLIER = 1.5


def _severity(z: float) -> str:
    if z >= 4.0:
        return "Critical"
    if z >= 3.5:
        return "High"
    return "Elevated"


def transaction_anomalies(market: Optional[str] = None, db: Optional[Database] = None) -> tuple[pd.DataFrame, dict]:
    """Single transactions far outside the normal range for the chosen market."""
    coll = transactions(db)
    q = scope_query(market)
    # Only the amounts are pulled to compute quartiles; the outlier rows
    # themselves are then fetched with a MongoDB range query.
    amounts = pd.Series([d["sales"] for d in coll.find(q, {"_id": 0, "sales": 1})], dtype=float)
    if amounts.empty:
        return pd.DataFrame(), {}
    log_lim = iqr_limits(np.log1p(amounts), TXN_IQR_MULTIPLIER)
    lim = {"lower": max(0.0, float(np.expm1(log_lim["lower"]))), "upper": float(np.expm1(log_lim["upper"])),
           "log_mean": log_lim["mean"], "log_std": log_lim["std"], "median": float(amounts.median()),
           "transactions_checked": int(len(amounts))}
    query = dict(q)
    query["$or"] = [{"sales": {"$gt": lim["upper"]}}, {"sales": {"$lt": lim["lower"]}}]
    proj = {"_id": 1, "order_id": 1, "order_date": 1, "customer_id": 1, "customer_name": 1, "country": 1,
            "market": 1, "category": 1, "product_name": 1, "sales": 1, "quantity": 1, "discount": 1, "profit": 1}
    out = pd.DataFrame(list(coll.find(query, proj).sort("sales", -1)))
    if out.empty:
        return out, lim
    out = out.rename(columns={"_id": "row_id"})
    out["z_score"] = ((np.log1p(out["sales"]) - lim["log_mean"]) / lim["log_std"]).round(2)
    out["severity"] = out["z_score"].apply(_severity)
    return out, lim


def store_anomalies(daily: pd.DataFrame, daily_lim: dict, txn: pd.DataFrame, txn_lim: dict,
                    daily_market: str, txn_market: str, db: Optional[Database] = None) -> int:
    now = utcnow()
    docs = []
    for r in records(daily[daily["anomaly"]]):
        docs.append({"level": "daily", "market_scope": daily_market, "date": r["order_date"],
                     "total_amount": r["total_amount"], "z_score": r["z_score"],
                     "upper_limit": daily_lim["upper"], "method": "3x IQR on daily totals", "generated_at": now})
    for r in records(txn):
        r.update({"level": "transaction", "market_scope": txn_market, "date": r.pop("order_date"),
                  "upper_limit": txn_lim["upper"], "method": "1.5x IQR on log(1 + transaction amount)",
                  "generated_at": now})
        docs.append(r)
    return replace_collection(config.SALES_ANOMALIES, docs, db)
