"""Analysis 1 (overall KPIs) and Analysis 5 (monthly trend), plus daily/weekly series.

All numbers are computed inside MongoDB with aggregation pipelines. pandas is
only used afterwards for month-over-month growth, which needs the previous row.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from pymongo.database import Database

from analysis.common import aggregate_df, scope_query


def overall_kpis(market: Optional[str] = None, extra_match: Optional[dict] = None,
                 db: Optional[Database] = None) -> dict:
    """Total revenue, profit, transactions, quantity, customers, orders, average values."""
    pipeline = [
        {"$match": scope_query(market, extra_match)},
        {"$group": {
            "_id": None,
            "total_revenue": {"$sum": "$sales"},
            "total_profit": {"$sum": "$profit"},
            "total_transactions": {"$sum": 1},
            "total_quantity": {"$sum": "$quantity"},
            "avg_transaction_value": {"$avg": "$sales"},
            "customers": {"$addToSet": "$customer_id"},
            "orders": {"$addToSet": "$order_id"},
            "products": {"$addToSet": "$product_id"},
            "countries": {"$addToSet": "$country"},
            "first_date": {"$min": "$order_date"},
            "last_date": {"$max": "$order_date"},
        }},
        {"$project": {
            "_id": 0,
            "total_revenue": {"$round": ["$total_revenue", 2]},
            "total_profit": {"$round": ["$total_profit", 2]},
            "total_transactions": 1,
            "total_quantity": 1,
            "avg_transaction_value": {"$round": ["$avg_transaction_value", 2]},
            "total_customers": {"$size": "$customers"},
            "total_orders": {"$size": "$orders"},
            "total_products": {"$size": "$products"},
            "total_countries": {"$size": "$countries"},
            "first_date": 1,
            "last_date": 1,
        }},
    ]
    df = aggregate_df(pipeline, db)
    if df.empty:
        return {"total_revenue": 0.0, "total_profit": 0.0, "total_transactions": 0, "total_quantity": 0,
                "avg_transaction_value": 0.0, "total_customers": 0, "total_orders": 0,
                "total_products": 0, "total_countries": 0, "first_date": None, "last_date": None,
                "profit_margin_pct": 0.0, "avg_order_value": 0.0}
    k = df.iloc[0].to_dict()
    k["profit_margin_pct"] = round(100 * k["total_profit"] / k["total_revenue"], 2) if k["total_revenue"] else 0.0
    k["avg_order_value"] = round(k["total_revenue"] / k["total_orders"], 2) if k["total_orders"] else 0.0
    return k


def monthly_sales(market: Optional[str] = None, extra_match: Optional[dict] = None,
                  db: Optional[Database] = None) -> pd.DataFrame:
    """Revenue, profit, orders per calendar month, with month-over-month growth."""
    pipeline = [
        {"$match": scope_query(market, extra_match)},
        {"$group": {
            "_id": {"year": {"$year": "$order_date"}, "month": {"$month": "$order_date"}},
            "revenue": {"$sum": "$sales"},
            "profit": {"$sum": "$profit"},
            "quantity": {"$sum": "$quantity"},
            "transactions": {"$sum": 1},
            "orders": {"$addToSet": "$order_id"},
        }},
        {"$project": {
            "_id": 0, "year": "$_id.year", "month": "$_id.month",
            "revenue": {"$round": ["$revenue", 2]}, "profit": {"$round": ["$profit", 2]},
            "quantity": 1, "transactions": 1, "orders": {"$size": "$orders"},
        }},
        {"$sort": {"year": 1, "month": 1}},
    ]
    df = aggregate_df(pipeline, db)
    if df.empty:
        return df
    df["year_month"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    df["growth_pct"] = (df["revenue"].pct_change() * 100).round(2)
    df["trend"] = df["growth_pct"].apply(
        lambda g: "start" if pd.isna(g) else ("growth" if g > 0 else ("decline" if g < 0 else "flat"))
    )
    df["rolling_3m_revenue"] = df["revenue"].rolling(3, min_periods=1).mean().round(2)
    return df


def yearly_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {"_id": {"$year": "$order_date"}, "revenue": {"$sum": "$sales"},
                    "profit": {"$sum": "$profit"}, "transactions": {"$sum": 1}}},
        {"$project": {"_id": 0, "year": "$_id", "revenue": {"$round": ["$revenue", 2]},
                      "profit": {"$round": ["$profit", 2]}, "transactions": 1}},
        {"$sort": {"year": 1}},
    ]
    df = aggregate_df(pipeline, db)
    if not df.empty:
        df["growth_pct"] = (df["revenue"].pct_change() * 100).round(2)
    return df


def daily_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """Total sales per order date (the series the forecasting and anomaly models use)."""
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {"_id": "$order_date", "total_amount": {"$sum": "$sales"}, "transactions": {"$sum": 1}}},
        {"$project": {"_id": 0, "order_date": "$_id", "total_amount": 1, "transactions": 1}},
        {"$sort": {"order_date": 1}},
    ]
    df = aggregate_df(pipeline, db)
    if not df.empty:
        df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def weekly_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """Weekly totals by ISO year and week (same grouping as the week-1 notebook)."""
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {
            "_id": {"year": {"$isoWeekYear": "$order_date"}, "week": {"$isoWeek": "$order_date"}},
            "total_amount": {"$sum": "$sales"},
            "week_start": {"$min": "$order_date"},
        }},
        {"$project": {"_id": 0, "iso_year": "$_id.year", "iso_week": "$_id.week",
                      "total_amount": {"$round": ["$total_amount", 2]}, "week_start": 1}},
        {"$sort": {"iso_year": 1, "iso_week": 1}},
    ]
    return aggregate_df(pipeline, db)


def segment_sales(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """Revenue by customer segment (Consumer / Corporate / Home Office)."""
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {"_id": "$segment", "revenue": {"$sum": "$sales"}, "profit": {"$sum": "$profit"},
                    "customers": {"$addToSet": "$customer_id"}}},
        {"$project": {"_id": 0, "segment": "$_id", "revenue": {"$round": ["$revenue", 2]},
                      "profit": {"$round": ["$profit", 2]}, "customers": {"$size": "$customers"}}},
        {"$sort": {"revenue": -1}},
    ]
    return aggregate_df(pipeline, db)
