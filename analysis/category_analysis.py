"""Analysis 2 (category-wise sales) and Analysis 3 (country / market sales).

All grouping happens inside MongoDB.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pymongo.database import Database

from analysis.common import aggregate_df, scope_query


def _group_by(field: str, label: str, market: Optional[str], db: Optional[Database],
              extra_match: Optional[dict] = None, limit: Optional[int] = None) -> pd.DataFrame:
    pipeline = [
        {"$match": scope_query(market, extra_match)},
        {"$group": {
            "_id": f"${field}",
            "revenue": {"$sum": "$sales"},
            "profit": {"$sum": "$profit"},
            "quantity": {"$sum": "$quantity"},
            "transactions": {"$sum": 1},
            "orders": {"$addToSet": "$order_id"},
        }},
        {"$project": {
            "_id": 0, label: "$_id",
            "revenue": {"$round": ["$revenue", 2]},
            "profit": {"$round": ["$profit", 2]},
            "quantity": 1, "transactions": 1,
            "orders": {"$size": "$orders"},
        }},
        {"$sort": {"revenue": -1}},
    ]
    df = aggregate_df(pipeline, db)
    if df.empty:
        return df
    total = df["revenue"].sum()
    df["revenue_share_pct"] = (100 * df["revenue"] / total).round(2) if total else 0.0
    df["profit_margin_pct"] = (100 * df["profit"] / df["revenue"]).round(2)
    # Shares are computed over all groups first, then the list is cut to `limit`.
    return df.head(int(limit)).reset_index(drop=True) if limit else df


def category_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    return _group_by("category", "category", market, db)


def sub_category_sales(market: Optional[str] = None, category: Optional[str] = None,
                       db: Optional[Database] = None) -> pd.DataFrame:
    extra = {"category": category} if category else None
    pipeline = [
        {"$match": scope_query(market, extra)},
        {"$group": {"_id": {"category": "$category", "sub_category": "$sub_category"},
                    "revenue": {"$sum": "$sales"}, "profit": {"$sum": "$profit"},
                    "quantity": {"$sum": "$quantity"}}},
        {"$project": {"_id": 0, "category": "$_id.category", "sub_category": "$_id.sub_category",
                      "revenue": {"$round": ["$revenue", 2]}, "profit": {"$round": ["$profit", 2]},
                      "quantity": 1}},
        {"$sort": {"revenue": -1}},
    ]
    return aggregate_df(pipeline, db)


def country_sales(market: Optional[str] = None, limit: Optional[int] = None,
                  db: Optional[Database] = None) -> pd.DataFrame:
    return _group_by("country", "country", market, db, limit=limit)


def market_sales(db: Optional[Database] = None) -> pd.DataFrame:
    return _group_by("market", "market", None, db)


def region_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    return _group_by("region", "region", market, db)


def best_of(df: pd.DataFrame, label: str) -> Optional[dict]:
    """The top row of a revenue-sorted result (highest revenue)."""
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    return {label: row[label], "revenue": float(row["revenue"]),
            "revenue_share_pct": float(row.get("revenue_share_pct", 0.0))}
