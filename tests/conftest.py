"""Shared fixtures. Tests run against a real MongoDB server (MONGO_URI from .env)
in a separate database, MarketMindDB_test, which is dropped at the end."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from database.import_data import import_dataset, read_csv  # noqa: E402
from database.mongodb_connection import DatabaseConnectionError, get_client  # noqa: E402

TEST_DB = "MarketMindDB_test"


@pytest.fixture(scope="session")
def client():
    try:
        return get_client(timeout_ms=3000)
    except DatabaseConnectionError as exc:
        pytest.skip(f"MongoDB not reachable, start it first: {exc}")


@pytest.fixture(scope="session")
def db(client):
    client.drop_database(TEST_DB)
    database = client[TEST_DB]
    yield database
    client.drop_database(TEST_DB)


@pytest.fixture(scope="session")
def imported(db):
    """Import the full dataset once for the whole test session."""
    return import_dataset(db=db, write_rejects=False, verbose=False)


@pytest.fixture(scope="session")
def source_df():
    """The CSV read independently with pandas, used as the expected answer."""
    df = read_csv(config.DATASET_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    for c in ["Total amount", "Profit", "Quantity"]:
        df[c] = pd.to_numeric(df[c])
    return df
