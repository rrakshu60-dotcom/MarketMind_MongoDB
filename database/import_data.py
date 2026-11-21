"""Import the Global Superstore sales CSV into MongoDB.

Usage (from the project folder):

    python -m database.import_data                 # import dataset/global_superstore_sales.csv
    python -m database.import_data --csv other.csv # import another file with the same layout
    python -m database.import_data --reset         # drop sales_transactions first, then import

What it does
1. Reads the CSV with pandas.
2. Validates it: required columns, dates, numbers, value ranges.
3. Cleans it: trims text, converts dates to real dates and numbers to numbers,
   drops rows that cannot be repaired and writes them to results/rejected_rows.csv.
4. Writes the documents in batches with ``bulk_write`` upserts keyed on the
   original Row ID (stored as ``_id``). Running the import again therefore
   updates rows instead of duplicating them.
5. Prints and returns the counts, and checks them against MongoDB.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pymongo import UpdateOne
from pymongo.database import Database
from pymongo.errors import BulkWriteError

# Allow "python database/import_data.py" as well as "python -m database.import_data"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from database.mongodb_connection import get_database, initialize_database  # noqa: E402

logger = logging.getLogger(__name__)

# CSV column -> MongoDB field
COLUMN_MAP = {
    "Row ID": "_id",
    "Order ID": "order_id",
    "Order Date": "order_date",
    "Ship Date": "ship_date",
    "Ship Mode": "ship_mode",
    "Customer ID": "customer_id",
    "Customer Name": "customer_name",
    "Segment": "segment",
    "City": "city",
    "State": "state",
    "Country": "country",
    "Postal Code": "postal_code",
    "Market": "market",
    "Region": "region",
    "Product ID": "product_id",
    "Category": "category",
    "Sub-Category": "sub_category",
    "Product Name": "product_name",
    "Total amount": "sales",
    "Quantity": "quantity",
    "Discount": "discount",
    "Profit": "profit",
    "Shipping Cost": "shipping_cost",
    "Order Priority": "order_priority",
    "unit_price": "unit_price",
}
# Other spellings seen in copies of the Global Superstore file
COLUMN_ALIASES = {"Sales": "Total amount", "Row_ID": "Row ID", "Unit Price": "unit_price"}

REQUIRED_COLUMNS = [
    "Row ID", "Order ID", "Order Date", "Customer ID", "Product ID", "Product Name",
    "Category", "Country", "Market", "Total amount", "Quantity", "Profit",
]
TEXT_COLUMNS = [
    "Order ID", "Ship Mode", "Customer ID", "Customer Name", "Segment", "City", "State",
    "Country", "Market", "Region", "Product ID", "Category", "Sub-Category", "Product Name",
    "Order Priority",
]
NUMERIC_COLUMNS = ["Total amount", "Quantity", "Discount", "Profit", "Shipping Cost", "unit_price"]

BATCH_SIZE = 5000


class DatasetValidationError(ValueError):
    """The file cannot be imported at all (for example, required columns are missing)."""


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The CSV file is empty.") from exc
    # Strip byte-order-mark debris and whitespace from headers (the original file
    # has a header like "ï»¿Row ID").
    df.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns and v not in df.columns})
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validate and clean the raw frame.

    Returns ``(clean_df, rejected_df, report)``. ``rejected_df`` keeps the raw
    values plus a ``reject_reason`` column.
    """
    if df.empty:
        raise DatasetValidationError("The CSV file has no data rows.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required column(s): {', '.join(missing)}")

    raw = df.copy()
    df = df.copy()
    report = {"rows_read": len(df)}

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

    df["Row ID"] = pd.to_numeric(df["Row ID"].str.strip(), errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", "", regex=False),
                                    errors="coerce")
    df["Order Date"] = pd.to_datetime(df["Order Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df["Ship Date"].str.strip(), format="%d-%m-%Y", errors="coerce")
    if "Postal Code" in df.columns:
        # Stored as text: postal codes are labels, and some start with 0.
        pc = pd.to_numeric(df["Postal Code"].str.strip(), errors="coerce")
        df["Postal Code"] = pc.map(lambda v: None if pd.isna(v) else str(int(v)))

    reasons = pd.Series("", index=df.index)

    def flag(mask: pd.Series, reason: str) -> None:
        reasons.loc[mask & (reasons == "")] = reason

    for col in REQUIRED_COLUMNS:
        if col in ("Order Date",):
            flag(df[col].isna(), "invalid or missing Order Date")
        elif col in ("Row ID", "Total amount", "Quantity", "Profit"):
            flag(df[col].isna(), f"invalid or missing {col}")
        else:
            flag(df[col].isna(), f"missing {col}")
    flag(df["Total amount"] < 0, "negative sales amount")
    flag(df["Quantity"] < 1, "quantity below 1")
    if "Discount" in df.columns:
        flag((df["Discount"] < 0) | (df["Discount"] > 1), "discount outside 0-1")
    flag(df["Row ID"].duplicated(keep="first") & df["Row ID"].notna(), "duplicate Row ID in file")

    bad = reasons != ""
    rejected = raw.loc[bad].copy()
    rejected["reject_reason"] = reasons[bad]
    clean = df.loc[~bad].copy()
    clean["Row ID"] = clean["Row ID"].astype("int64")
    clean["Quantity"] = clean["Quantity"].astype("int64")

    report.update({
        "rows_valid": int(len(clean)),
        "rows_rejected": int(bad.sum()),
        "reject_reasons": reasons[bad].value_counts().to_dict(),
        "missing_postal_codes": int(clean["Postal Code"].isna().sum()) if "Postal Code" in clean else 0,
    })
    return clean, rejected, report


def to_documents(df: pd.DataFrame) -> list[dict]:
    """Turn the clean frame into MongoDB documents with proper BSON types."""
    cols = [c for c in COLUMN_MAP if c in df.columns]
    out = df[cols].rename(columns=COLUMN_MAP)
    out = out.astype(object).where(pd.notna(out), None)
    docs = out.to_dict("records")
    for d in docs:
        d["_id"] = int(d["_id"])
        d["quantity"] = int(d["quantity"])
        for f in ("sales", "profit", "discount", "shipping_cost", "unit_price"):
            if d.get(f) is not None:
                d[f] = float(d[f])
        for f in ("order_date", "ship_date"):
            if d.get(f) is not None:
                d[f] = d[f].to_pydatetime()
        od = d["order_date"]
        # Pre-computed calendar fields make month/year grouping cheap and readable.
        d["order_year"] = od.year
        d["order_month"] = od.month
        d["year_month"] = od.strftime("%Y-%m")
    return docs


def _batches(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def import_dataset(
    csv_path: Optional[Path] = None,
    db: Optional[Database] = None,
    batch_size: int = BATCH_SIZE,
    reset: bool = False,
    write_rejects: bool = True,
    verbose: bool = True,
) -> dict:
    """Import a CSV into ``sales_transactions``. Returns a summary dict."""
    csv_path = Path(csv_path or config.DATASET_PATH)
    db = db if db is not None else get_database()
    started = time.perf_counter()

    if reset:
        db[config.SALES_TRANSACTIONS].drop()
    initialize_database(db)
    coll = db[config.SALES_TRANSACTIONS]
    count_before = coll.count_documents({})

    raw = read_csv(csv_path)
    clean, rejected, report = clean_dataframe(raw)
    if write_rejects and len(rejected):
        config.RESULTS_DIR.mkdir(exist_ok=True)
        rejected.to_csv(config.RESULTS_DIR / "rejected_rows.csv", index=False)

    docs = to_documents(clean)
    now = datetime.now(timezone.utc)
    inserted = modified = matched = 0
    for batch in _batches(docs, batch_size):
        ops = []
        for doc in batch:
            _id = doc.pop("_id")
            ops.append(UpdateOne(
                {"_id": _id},
                {"$set": doc, "$setOnInsert": {"imported_at": now}},
                upsert=True,
            ))
        try:
            result = coll.bulk_write(ops, ordered=False)
        except BulkWriteError as exc:
            details = exc.details.get("writeErrors", [])[:3]
            raise RuntimeError(f"Bulk write failed for {len(exc.details.get('writeErrors', []))} rows: {details}") from exc
        inserted += result.upserted_count
        modified += result.modified_count
        matched += result.matched_count

    count_after = coll.count_documents({})
    elapsed = time.perf_counter() - started
    summary = {
        "source_file": str(csv_path.name),
        **report,
        "inserted": inserted,
        "updated": modified,
        "unchanged": matched - modified,
        "documents_before": count_before,
        "documents_after": count_after,
        "seconds": round(elapsed, 2),
        "batch_size": batch_size,
        "finished_at": datetime.now(timezone.utc),
    }
    db[config.IMPORT_LOG].insert_one(dict(summary))

    if verbose:
        print(f"Source file           : {csv_path}")
        print(f"Rows read             : {report['rows_read']:,}")
        print(f"Rows valid            : {report['rows_valid']:,}")
        print(f"Rows rejected         : {report['rows_rejected']:,} {report['reject_reasons'] or ''}")
        print(f"New documents inserted: {inserted:,}")
        print(f"Existing docs updated : {modified:,}")
        print(f"Existing docs same    : {matched - modified:,}")
        print(f"Documents in MongoDB  : {count_before:,} before -> {count_after:,} after")
        print(f"Time taken            : {elapsed:.2f} s")
        ok = count_after >= report["rows_valid"]
        print("Verification          : " + ("OK, every valid row is stored" if ok else "MISMATCH"))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Import sales CSV data into MongoDB")
    parser.add_argument("--csv", type=Path, default=config.DATASET_PATH, help="CSV file to import")
    parser.add_argument("--reset", action="store_true", help="drop sales_transactions before importing")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        import_dataset(args.csv, reset=args.reset, batch_size=args.batch_size)
    except (DatasetValidationError, FileNotFoundError) as exc:
        print(f"Import stopped: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # connection problems and anything unexpected
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
