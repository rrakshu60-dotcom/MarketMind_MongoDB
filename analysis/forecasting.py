"""Sales forecasting with the existing CatBoost model (catboost_daily_model.pkl).

Workflow: MongoDB daily totals (aggregation) -> lag/rolling features (same as
AIML/Integrated_API/app.py ``build_forecast_row``) -> CatBoost predicts
log(1 + sales) -> back-transform -> store in ``sales_forecasts``.

The model predicts one day at a time, so a 30-day forecast is recursive:
each prediction becomes the "history" for the next day.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pymongo.database import Database

import config
from analysis.common import replace_collection, utcnow
from analysis.model_loader import load_model
from analysis.sales_analysis import daily_sales

FORECAST_FEATURE_COLS = [
    "day", "month", "year", "dayofweek", "weekofyear", "quarter", "is_weekend",
    "lag1", "lag2", "lag3", "lag7", "lag14", "lag30",
    "log_lag1", "log_lag2", "log_lag3", "log_lag7", "log_lag14", "log_lag30",
    "rolling7", "rolling30", "log_rolling7", "log_rolling30",
    "ema7", "log_ema7",
]
MODEL_NAME = "CatBoost (log target) catboost_daily_model.pkl"
MIN_HISTORY_DAYS = 30


def build_forecast_row(amounts: np.ndarray, future_date: pd.Timestamp) -> pd.DataFrame:
    """Features for one future day, from the sales history before it.

    Reused from the internship project's Integrated API so the model receives
    exactly the inputs it was trained with.
    """
    amounts = np.asarray(amounts, dtype=float)
    log_amounts = np.log1p(amounts)

    def lag(arr, k):
        return float(arr[-k]) if len(arr) >= k else float(arr[0])

    def rolling(arr, w):
        return float(np.mean(arr[-w:] if len(arr) >= w else arr))

    def ema(arr, span):
        return float(pd.Series(arr).ewm(span=span).mean().iloc[-1])

    row = {
        "day": future_date.day, "month": future_date.month, "year": future_date.year,
        "dayofweek": future_date.dayofweek, "weekofyear": int(future_date.isocalendar().week),
        "quarter": future_date.quarter, "is_weekend": int(future_date.dayofweek in (5, 6)),
    }
    for k in (1, 2, 3, 7, 14, 30):
        row[f"lag{k}"] = lag(amounts, k)
        row[f"log_lag{k}"] = lag(log_amounts, k)
    row.update({
        "rolling7": rolling(amounts, 7), "rolling30": rolling(amounts, 30),
        "log_rolling7": rolling(log_amounts, 7), "log_rolling30": rolling(log_amounts, 30),
        "ema7": ema(amounts, 7), "log_ema7": ema(log_amounts, 7),
    })
    return pd.DataFrame([row])[FORECAST_FEATURE_COLS]


def forecast_sales(days: int = 30, market: Optional[str] = None, db: Optional[Database] = None,
                   history: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Recursive day-by-day forecast after the last date in MongoDB."""
    if not 1 <= days <= 365:
        raise ValueError("days must be between 1 and 365")
    market = market or config.ML_MARKET
    history = history if history is not None else daily_sales(market, db)
    if len(history) < MIN_HISTORY_DAYS:
        raise ValueError(f"Need at least {MIN_HISTORY_DAYS} days of sales history, found {len(history)}")
    model = load_model("forecast")
    amounts = list(history["total_amount"].astype(float))
    last_date = pd.Timestamp(history["order_date"].iloc[-1])
    recent = np.array(amounts[-30:])
    recent_mean, recent_std = float(recent.mean()), float(recent.std(ddof=1))

    rows = []
    for i in range(1, days + 1):
        d = last_date + pd.Timedelta(days=i)
        pred = max(0.0, float(np.expm1(model.predict(build_forecast_row(np.array(amounts), d))[0])))
        rows.append({"forecast_date": d, "predicted_sales": round(pred, 2)})
        amounts.append(pred)
    out = pd.DataFrame(rows)
    out["recent_30d_mean"] = round(recent_mean, 2)
    out["recent_30d_std"] = round(recent_std, 2)
    return out


def backtest(days: int = 60, market: Optional[str] = None, db: Optional[Database] = None,
             history: Optional[pd.DataFrame] = None) -> dict:
    """One-step-ahead check on the last ``days`` real days in MongoDB.

    For each day, the model sees only the actual sales before that day.
    Note: the saved model was trained on this same dataset, so these numbers
    are an in-sample sanity check, not a true hold-out evaluation.
    """
    market = market or config.ML_MARKET
    history = history if history is not None else daily_sales(market, db)
    if len(history) <= days + MIN_HISTORY_DAYS:
        raise ValueError("Not enough history for this backtest window")
    model = load_model("forecast")
    amounts = history["total_amount"].astype(float).to_numpy()
    dates = pd.to_datetime(history["order_date"]).to_list()
    rows = []
    for idx in range(len(amounts) - days, len(amounts)):
        X = build_forecast_row(amounts[:idx], pd.Timestamp(dates[idx]))
        pred = max(0.0, float(np.expm1(model.predict(X)[0])))
        rows.append({"date": dates[idx], "actual": float(amounts[idx]), "predicted": pred})
    bt = pd.DataFrame(rows)
    err = (bt["actual"] - bt["predicted"]).abs()
    weekly = bt.set_index("date").resample("W").sum()
    weekly = weekly[weekly["actual"] > 0]
    return {
        "days": days,
        "mae": round(float(err.mean()), 2),
        "wape_pct": round(float(100 * err.sum() / bt["actual"].sum()), 2),
        "weekly_mape_pct": round(float(100 * ((weekly["actual"] - weekly["predicted"]).abs()
                                              / weekly["actual"]).mean()), 2),
        "series": bt,
    }


def store_forecast(fc: pd.DataFrame, market: str, db: Optional[Database] = None) -> int:
    now = utcnow()
    docs = [{
        "model": MODEL_NAME,
        "market_scope": market,
        "forecast_date": r.forecast_date.to_pydatetime(),
        "predicted_sales": float(r.predicted_sales),
        "horizon_day": i + 1,
        "generated_at": now,
    } for i, r in enumerate(fc.itertuples())]
    return replace_collection(config.SALES_FORECASTS, docs, db, {"model": MODEL_NAME})
