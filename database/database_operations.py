"""Read operations on the ``sales_transactions`` collection.

Every function here queries MongoDB. Nothing reads the CSV file.
Results come back as pandas DataFrames so they drop straight into charts and
tables; pass ``as_dataframe=False`` to get the raw documents instead.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time
from typing import Any, Optional, Union

import pandas as pd
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

import config
from database.mongodb_connection import get_database

DateLike = Union[str, date, datetime, pd.Timestamp]

# Fields shown in tables (skips internal bookkeeping fields)
DEFAULT_PROJECTION = {
    "_id": 1, "order_id": 1, "order_date": 1, "customer_id": 1, "customer_name": 1, "segment": 1,
    "country": 1, "market": 1, "region": 1, "category": 1, "sub_category": 1, "product_name": 1,
    "sales": 1, "quantity": 1, "discount": 1, "profit": 1,
}
MAX_LIMIT = 100_000


def _db(db: Optional[Database]) -> Database:
    return db if db is not None else get_database()


def _coll(db: Optional[Database]):
    return _db(db)[config.SALES_TRANSACTIONS]


def _to_datetime(value: DateLike, end_of_day: bool = False) -> datetime:
    """Parse user input into a datetime, raising ValueError on bad input."""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.max if end_of_day else time.min).replace(microsecond=0)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc
        if end_of_day and len(value.strip()) == 10:
            parsed = datetime.combine(parsed.date(), time(23, 59, 59))
        return parsed
    raise ValueError(f"Unsupported date value: {value!r}")


def _check_limit(limit: Optional[int]) -> int:
    if limit is None:
        return 0  # 0 means "no limit" in PyMongo
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return min(limit, MAX_LIMIT)


def _result(cursor, as_dataframe: bool):
    docs = list(cursor)
    if not as_dataframe:
        return docs
    df = pd.DataFrame(docs)
    if not df.empty and "_id" in df.columns:
        df = df.rename(columns={"_id": "row_id"})
    return df


def build_filter(
    customer_id: Optional[str] = None,
    category: Optional[str] = None,
    country: Optional[str] = None,
    market: Optional[str] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    product_search: Optional[str] = None,
) -> dict:
    """Build a MongoDB query document from optional filters."""
    query: dict[str, Any] = {}
    if customer_id:
        query["customer_id"] = customer_id.strip().upper()
    if category:
        query["category"] = category
    if country:
        query["country"] = country
    if market:
        query["market"] = market
    if start_date or end_date:
        rng = {}
        if start_date:
            rng["$gte"] = _to_datetime(start_date)
        if end_date:
            rng["$lte"] = _to_datetime(end_date, end_of_day=True)
        if "$gte" in rng and "$lte" in rng and rng["$gte"] > rng["$lte"]:
            raise ValueError("start_date must be on or before end_date")
        query["order_date"] = rng
    if product_search:
        query["product_name"] = {"$regex": re.escape(product_search.strip()), "$options": "i"}
    return query


# ----------------------------------------------------------------------------
# Retrieval functions
# ----------------------------------------------------------------------------

def count_transactions(query: Optional[dict] = None, db: Optional[Database] = None) -> int:
    return _coll(db).count_documents(query or {})


def fetch_all_sales(limit: Optional[int] = 1000, skip: int = 0, db: Optional[Database] = None,
                    as_dataframe: bool = True):
    """Fetch sales records, newest first. ``limit=None`` fetches every record."""
    cursor = (_coll(db).find({}, DEFAULT_PROJECTION)
              .sort([("order_date", DESCENDING), ("_id", ASCENDING)])
              .skip(max(skip, 0)).limit(_check_limit(limit)))
    return _result(cursor, as_dataframe)


def get_by_customer(customer_id: str, db: Optional[Database] = None, as_dataframe: bool = True):
    if not customer_id or not customer_id.strip():
        raise ValueError("customer_id is required")
    cursor = _coll(db).find(build_filter(customer_id=customer_id), DEFAULT_PROJECTION).sort("order_date", ASCENDING)
    return _result(cursor, as_dataframe)


def get_by_category(category: str, limit: Optional[int] = 1000, db: Optional[Database] = None,
                    as_dataframe: bool = True):
    cursor = (_coll(db).find(build_filter(category=category), DEFAULT_PROJECTION)
              .sort("order_date", DESCENDING).limit(_check_limit(limit)))
    return _result(cursor, as_dataframe)


def get_by_country(country: str, limit: Optional[int] = 1000, db: Optional[Database] = None,
                   as_dataframe: bool = True):
    cursor = (_coll(db).find(build_filter(country=country), DEFAULT_PROJECTION)
              .sort("order_date", DESCENDING).limit(_check_limit(limit)))
    return _result(cursor, as_dataframe)


def get_by_date_range(start_date: DateLike, end_date: DateLike, limit: Optional[int] = None,
                      db: Optional[Database] = None, as_dataframe: bool = True):
    cursor = (_coll(db).find(build_filter(start_date=start_date, end_date=end_date), DEFAULT_PROJECTION)
              .sort("order_date", ASCENDING).limit(_check_limit(limit)))
    return _result(cursor, as_dataframe)


def search_transactions(limit: Optional[int] = 500, db: Optional[Database] = None,
                        as_dataframe: bool = True, **filters):
    """Combined search used by the dashboard's Transaction Explorer."""
    query = build_filter(**filters)
    cursor = (_coll(db).find(query, DEFAULT_PROJECTION)
              .sort([("order_date", DESCENDING), ("_id", ASCENDING)]).limit(_check_limit(limit)))
    return _result(cursor, as_dataframe)


def get_top_products(n: int = 10, by: str = "revenue", query: Optional[dict] = None,
                     db: Optional[Database] = None) -> pd.DataFrame:
    """Top-N products by revenue or by quantity, computed in MongoDB."""
    if by not in ("revenue", "quantity"):
        raise ValueError("by must be 'revenue' or 'quantity'")
    if n < 1:
        raise ValueError("n must be at least 1")
    sort_field = "total_sales" if by == "revenue" else "total_quantity"
    pipeline = [
        {"$match": query or {}},
        {"$group": {
            "_id": "$product_name",
            "category": {"$first": "$category"},
            "sub_category": {"$first": "$sub_category"},
            "total_sales": {"$sum": "$sales"},
            "total_quantity": {"$sum": "$quantity"},
            "total_profit": {"$sum": "$profit"},
            "orders": {"$addToSet": "$order_id"},
        }},
        {"$project": {
            "_id": 0, "product_name": "$_id", "category": 1, "sub_category": 1,
            "total_sales": {"$round": ["$total_sales", 2]},
            "total_quantity": 1,
            "total_profit": {"$round": ["$total_profit", 2]},
            "order_count": {"$size": "$orders"},
        }},
        {"$sort": {sort_field: -1, "product_name": 1}},
        {"$limit": int(n)},
    ]
    return pd.DataFrame(list(_coll(db).aggregate(pipeline, allowDiskUse=True)))


def distinct_values(field: str, query: Optional[dict] = None, db: Optional[Database] = None) -> list:
    allowed = {"category", "sub_category", "country", "market", "region", "segment", "ship_mode"}
    if field not in allowed:
        raise ValueError(f"field must be one of {sorted(allowed)}")
    return sorted(v for v in _coll(db).distinct(field, query or {}) if v)


def date_bounds(db: Optional[Database] = None) -> tuple[Optional[datetime], Optional[datetime]]:
    coll = _coll(db)
    first = coll.find_one({}, {"order_date": 1}, sort=[("order_date", ASCENDING)])
    last = coll.find_one({}, {"order_date": 1}, sort=[("order_date", DESCENDING)])
    if not first:
        return None, None
    return first["order_date"], last["order_date"]
