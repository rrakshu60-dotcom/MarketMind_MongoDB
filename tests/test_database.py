"""Tests for the MongoDB connection, import and retrieval layer."""
import time
from datetime import datetime

import pandas as pd
import pytest
from pymongo.errors import WriteError

import config
from database import database_operations as ops
from database.import_data import DatasetValidationError, clean_dataframe, import_dataset, read_csv
from database.mongodb_connection import (DatabaseConnectionError, _mask_uri, check_connection, get_client,
                                         initialize_database)


# --- connection -------------------------------------------------------------

def test_connection_ping(client):
    info = check_connection()
    assert info["connected"] is True
    assert info["server_version"]


def test_bad_uri_fails_fast_with_clear_error():
    start = time.perf_counter()
    with pytest.raises(DatabaseConnectionError):
        get_client("mongodb://127.0.0.1:1", timeout_ms=500)
    assert time.perf_counter() - start < 5


def test_password_is_masked():
    assert _mask_uri("mongodb+srv://bob:secret@host/x") == "mongodb+srv://bob:****@host/x"
    assert "secret" not in _mask_uri("mongodb://bob:secret@localhost:27017")


# --- initialisation -----------------------------------------------------------

def test_collections_and_indexes_created(db):
    result = initialize_database(db)
    names = set(db.list_collection_names())
    for c in [config.SALES_TRANSACTIONS, config.SALES_ANALYSIS, config.SALES_FORECASTS]:
        assert c in names
    idx = db[config.SALES_TRANSACTIONS].index_information()
    keys = {tuple(k for k, _ in v["key"]) for v in idx.values()}
    assert ("order_date",) in keys and ("customer_id", "order_date") in keys and ("country",) in keys
    # initialising twice is harmless
    assert initialize_database(db)["created_collections"] == []
    assert result["database"] == db.name


def test_schema_validator_rejects_bad_document(db, imported):
    good = db[config.SALES_TRANSACTIONS].find_one({})
    bad = dict(good, _id=999_999_999, sales=-5.0)
    with pytest.raises(WriteError):
        db[config.SALES_TRANSACTIONS].insert_one(bad)


# --- import -------------------------------------------------------------------

def test_import_counts_match_source(imported, source_df, db):
    assert imported["rows_read"] == len(source_df) == 51290
    assert imported["rows_rejected"] == 0
    assert imported["inserted"] == 51290
    assert db[config.SALES_TRANSACTIONS].count_documents({}) == 51290


def test_reimport_creates_no_duplicates(imported, db):
    again = import_dataset(db=db, write_rejects=False, verbose=False)
    assert again["inserted"] == 0
    assert again["unchanged"] == 51290
    assert db[config.SALES_TRANSACTIONS].count_documents({}) == 51290


def test_stored_types(db, imported):
    doc = db[config.SALES_TRANSACTIONS].find_one({"_id": 32298})
    assert isinstance(doc["order_date"], datetime)
    assert doc["order_date"] == datetime(2012, 7, 31)
    assert isinstance(doc["sales"], float) and isinstance(doc["quantity"], int)
    assert doc["order_id"] == "CA-2012-124891" and doc["year_month"] == "2012-07"


def test_cleaning_rejects_invalid_rows():
    raw = read_csv(config.DATASET_PATH).head(6).copy()
    raw.loc[0, "Order Date"] = "31-31-2012"       # impossible date
    raw.loc[1, "Total amount"] = "abc"            # not a number
    raw.loc[2, "Quantity"] = "0"                  # quantity below 1
    raw.loc[3, "Customer ID"] = "  "              # missing customer
    raw.loc[5, "Row ID"] = raw.loc[4, "Row ID"]   # row 5 repeats row 4's id
    clean, rejected, report = clean_dataframe(raw)
    assert report["rows_valid"] == 1 and report["rows_rejected"] == 5
    assert list(rejected["reject_reason"]) == [
        "invalid or missing Order Date", "invalid or missing Total amount",
        "quantity below 1", "missing Customer ID", "duplicate Row ID in file"]


def test_missing_columns_and_empty_file(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Order Date,Total amount\n01-01-2014,10\n")
    with pytest.raises(DatasetValidationError, match="Missing required"):
        clean_dataframe(read_csv(p))
    e = tmp_path / "empty.csv"
    e.write_text("")
    with pytest.raises(DatasetValidationError):
        read_csv(e)
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "nope.csv")


# --- retrieval ----------------------------------------------------------------

def test_fetch_all(db, imported):
    df = ops.fetch_all_sales(limit=50, db=db)
    assert len(df) == 50 and df["order_date"].is_monotonic_decreasing
    assert len(ops.fetch_all_sales(limit=None, db=db)) == 51290


def test_get_by_customer(db, imported, source_df):
    cid = "TB-11280"
    expected = source_df[source_df["Customer ID"] == cid]
    got = ops.get_by_customer(cid.lower(), db=db)  # case-insensitive input
    assert len(got) == len(expected) > 0
    assert set(got["customer_id"]) == {cid}
    assert got["sales"].sum() == pytest.approx(expected["Total amount"].sum())


def test_filter_by_category_and_country(db, imported, source_df):
    assert len(ops.get_by_category("Technology", limit=None, db=db)) == (source_df["Category"] == "Technology").sum()
    assert len(ops.get_by_country("India", limit=None, db=db)) == (source_df["Country"] == "India").sum()


def test_date_range(db, imported, source_df):
    got = ops.get_by_date_range("2014-01-01", "2014-01-31", db=db)
    exp = source_df[(source_df["Order Date"] >= "2014-01-01") & (source_df["Order Date"] <= "2014-01-31")]
    assert len(got) == len(exp)
    assert got["order_date"].min() >= pd.Timestamp("2014-01-01")


def test_top_products_match_pandas(db, imported, source_df):
    got = ops.get_top_products(10, "revenue", db=db)
    exp = source_df.groupby("Product Name")["Total amount"].sum().sort_values(ascending=False).head(10)
    assert list(got["product_name"]) == list(exp.index)
    assert got["total_sales"].iloc[0] == pytest.approx(exp.iloc[0], abs=0.01)


def test_combined_search(db, imported, source_df):
    got = ops.search_transactions(limit=None, db=db, category="Furniture", country="France",
                                  start_date="2013-01-01", end_date="2013-12-31")
    exp = source_df[(source_df["Category"] == "Furniture") & (source_df["Country"] == "France")
                    & (source_df["Order Date"].dt.year == 2013)]
    assert len(got) == len(exp) > 0


def test_invalid_inputs(db, imported):
    with pytest.raises(ValueError):
        ops.get_by_date_range("2014-02-01", "2014-01-01", db=db)
    with pytest.raises(ValueError):
        ops.get_by_date_range("not-a-date", "2014-01-01", db=db)
    with pytest.raises(ValueError):
        ops.get_by_customer("  ", db=db)
    with pytest.raises(ValueError):
        ops.get_top_products(10, by="colour", db=db)
    with pytest.raises(ValueError):
        ops.fetch_all_sales(limit=-1, db=db)
    assert ops.get_by_customer("ZZ-00000", db=db).empty
    # regex characters in a product search are treated as plain text
    assert ops.search_transactions(db=db, product_search="(.*").empty
