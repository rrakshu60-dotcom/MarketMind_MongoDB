"""Small helpers shared by the analysis modules."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from pymongo.database import Database

import config
from database.mongodb_connection import get_database


def db_or_default(db: Optional[Database]) -> Database:
    return db if db is not None else get_database()


def transactions(db: Optional[Database] = None):
    return db_or_default(db)[config.SALES_TRANSACTIONS]


def scope_query(market: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    """Base $match for an analysis: everything, or one market (e.g. "US")."""
    query = dict(extra or {})
    if market and market.upper() != "ALL":
        query["market"] = market
    return query


def scope_name(market: Optional[str]) -> str:
    return market if market and market.upper() != "ALL" else "ALL"


def aggregate_df(pipeline: list[dict], db: Optional[Database] = None) -> pd.DataFrame:
    return pd.DataFrame(list(transactions(db).aggregate(pipeline, allowDiskUse=True)))


def _plain(value: Any) -> Any:
    """Convert numpy/pandas scalars so PyMongo can store them."""
    if isinstance(value, pd.DataFrame):
        return [_plain(r) for r in value.to_dict("records")]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [_plain(r) for r in df.to_dict("records")]


def save_analysis(analysis_type: str, data: Any, scope: str = "ALL", db: Optional[Database] = None,
                  meta: Optional[dict] = None) -> None:
    """Upsert one analysis result into ``sales_analysis`` (one document per type and scope)."""
    if isinstance(data, pd.DataFrame):
        data = records(data)
    doc = {
        "analysis_type": analysis_type,
        "scope": scope,
        "data": _plain(data),
        "generated_at": datetime.now(timezone.utc),
        "source": f"MongoDB {config.SALES_TRANSACTIONS} aggregation",
    }
    if meta:
        doc["meta"] = _plain(meta)
    db_or_default(db)[config.SALES_ANALYSIS].replace_one(
        {"analysis_type": analysis_type, "scope": scope}, doc, upsert=True
    )


def load_analysis(analysis_type: str, scope: str = "ALL", db: Optional[Database] = None) -> Optional[dict]:
    return db_or_default(db)[config.SALES_ANALYSIS].find_one(
        {"analysis_type": analysis_type, "scope": scope}, {"_id": 0}
    )


def replace_collection(name: str, docs: list[dict], db: Optional[Database] = None,
                       filter_: Optional[dict] = None) -> int:
    """Replace the stored results of one run (delete the old ones, insert the new)."""
    coll = db_or_default(db)[name]
    coll.delete_many(filter_ or {})
    if docs:
        coll.insert_many([_plain(d) for d in docs], ordered=False)
    return len(docs)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
