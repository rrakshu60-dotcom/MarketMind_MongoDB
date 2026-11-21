"""MongoDB connection, validation and database initialisation.

Every other module gets its database handle from :func:`get_database`, so the
connection settings live in one place (``config.py`` / ``.env``).
"""
from __future__ import annotations

import logging
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import (
    CollectionInvalid,
    ConfigurationError,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

import config

logger = logging.getLogger(__name__)

# One client per (uri, timeout). Clients are never closed behind a caller's back.
_clients: dict[tuple, MongoClient] = {}


class DatabaseConnectionError(RuntimeError):
    """Raised when MongoDB cannot be reached or the URI is invalid."""


def get_client(uri: Optional[str] = None, timeout_ms: Optional[int] = None) -> MongoClient:
    """Return a connected, validated :class:`MongoClient`.

    The client is cached and reused, because a MongoClient owns a connection
    pool and is meant to be long-lived.
    """
    uri = uri or config.MONGO_URI
    timeout_ms = timeout_ms or config.MONGO_TIMEOUT_MS
    key = (uri, timeout_ms)
    if key in _clients:
        return _clients[key]

    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            appname="MarketMindAI",
        )
        # "ping" forces a round trip, so a bad host fails here, not later.
        client.admin.command("ping")
    except (ServerSelectionTimeoutError, ConfigurationError, OperationFailure) as exc:
        raise DatabaseConnectionError(
            f"Could not connect to MongoDB ({_mask_uri(uri)}): {exc.__class__.__name__}: "
            f"{str(exc).splitlines()[0].split(' (configured')[0]}"
        ) from exc
    except PyMongoError as exc:
        raise DatabaseConnectionError(f"MongoDB error: {exc}") from exc

    _clients[key] = client
    logger.info("Connected to MongoDB at %s", _mask_uri(uri))
    return client


def get_database(db_name: Optional[str] = None, uri: Optional[str] = None) -> Database:
    """Return the MarketMindDB database handle (or the one named)."""
    return get_client(uri)[db_name or config.MONGO_DB_NAME]


def close_connection() -> None:
    for client in _clients.values():
        client.close()
    _clients.clear()


def check_connection(uri: Optional[str] = None) -> dict:
    """Return basic server facts, or raise DatabaseConnectionError."""
    client = get_client(uri)
    info = client.server_info()
    return {"connected": True, "server_version": info.get("version"), "uri": _mask_uri(uri or config.MONGO_URI)}


def _mask_uri(uri: str) -> str:
    """Hide the password part of a URI so it never reaches logs or the UI."""
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:****@{host}"


# --------------------------------------------------------------------------
# Schema validation and indexes
# --------------------------------------------------------------------------

SALES_TRANSACTION_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": [
            "_id", "order_id", "order_date", "customer_id", "product_id",
            "product_name", "category", "country", "market", "sales", "quantity", "profit",
        ],
        "properties": {
            "_id": {"bsonType": ["int", "long"], "description": "Original Row ID (unique)"},
            "order_id": {"bsonType": "string"},
            "order_date": {"bsonType": "date"},
            "ship_date": {"bsonType": ["date", "null"]},
            "customer_id": {"bsonType": "string"},
            "product_id": {"bsonType": "string"},
            "product_name": {"bsonType": "string"},
            "category": {"bsonType": "string"},
            "country": {"bsonType": "string"},
            "market": {"bsonType": "string"},
            "sales": {"bsonType": ["double", "int", "long"], "minimum": 0},
            "quantity": {"bsonType": ["int", "long"], "minimum": 1},
            "discount": {"bsonType": ["double", "int", "null"], "minimum": 0, "maximum": 1},
            "profit": {"bsonType": ["double", "int", "long"]},
        },
    }
}

INDEXES = {
    config.SALES_TRANSACTIONS: [
        ([("order_date", ASCENDING)], {}),
        ([("customer_id", ASCENDING), ("order_date", ASCENDING)], {}),
        ([("category", ASCENDING), ("sub_category", ASCENDING)], {}),
        ([("country", ASCENDING)], {}),
        ([("market", ASCENDING), ("order_date", ASCENDING)], {}),
        ([("product_id", ASCENDING)], {}),
        ([("order_id", ASCENDING)], {}),
        ([("sales", DESCENDING)], {}),
    ],
    config.SALES_ANALYSIS: [([("analysis_type", ASCENDING), ("scope", ASCENDING)], {"unique": True})],
    config.SALES_FORECASTS: [([("model", ASCENDING), ("forecast_date", ASCENDING)], {"unique": True})],
    config.CUSTOMER_SEGMENTS: [([("segment_name", ASCENDING)], {})],
    config.CHURN_PREDICTIONS: [([("risk_level", ASCENDING), ("churn_probability", DESCENDING)], {})],
    config.SALES_ANOMALIES: [([("level", ASCENDING), ("date", ASCENDING)], {})],
    config.PRODUCT_RECOMMENDATIONS: [([("product_name", ASCENDING), ("rank", ASCENDING)], {})],
}


def initialize_database(db: Optional[Database] = None) -> dict:
    """Create collections (with validation) and indexes. Safe to run repeatedly."""
    db = db if db is not None else get_database()
    existing = set(db.list_collection_names())
    created = []

    for name in INDEXES:
        if name in existing:
            continue
        try:
            if name == config.SALES_TRANSACTIONS:
                db.create_collection(name, validator=SALES_TRANSACTION_VALIDATOR, validationLevel="strict")
            else:
                db.create_collection(name)
            created.append(name)
        except CollectionInvalid:
            pass  # created by another process in the meantime

    if config.SALES_TRANSACTIONS in existing:
        # Keep the validator up to date on an existing collection.
        db.command("collMod", config.SALES_TRANSACTIONS, validator=SALES_TRANSACTION_VALIDATOR,
                   validationLevel="strict")

    index_names = {}
    for name, specs in INDEXES.items():
        index_names[name] = [db[name].create_index(keys, **opts) for keys, opts in specs]

    return {"database": db.name, "created_collections": created, "indexes": index_names}
