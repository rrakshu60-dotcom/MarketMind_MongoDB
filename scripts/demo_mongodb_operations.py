"""Terminal walk-through of every MongoDB operation, for the faculty demo.

    python scripts/demo_mongodb_operations.py

Shows: connection -> stored data -> retrieval queries -> aggregation pipelines
-> stored ML results. Everything is read from MongoDB.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from analysis import category_analysis, customer_analysis, sales_analysis  # noqa: E402
from database import database_operations as ops  # noqa: E402
from database.mongodb_connection import DatabaseConnectionError, check_connection, get_database  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_colwidth", 40)


def step(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    step("1. CONNECT TO MONGODB")
    try:
        info = check_connection()
    except DatabaseConnectionError as exc:
        print(exc)
        return 1
    db = get_database()
    print(f"Connected to {info['uri']} (server {info['server_version']}), database '{db.name}'")
    print("Collections:", ", ".join(sorted(db.list_collection_names())))

    step("2. STORED DATA: db.sales_transactions.countDocuments({})")
    print(f"{ops.count_transactions():,} documents")
    print("Sample document:", db[config.SALES_TRANSACTIONS].find_one({}, {"imported_at": 0}))

    step("3. RETRIEVAL: find({customer_id: 'TB-11280'})")
    print(ops.get_by_customer("TB-11280")[["order_date", "order_id", "product_name", "sales", "profit"]].head(8))

    step("4. RETRIEVAL: find({category: 'Technology', country: 'India'}).limit(5)")
    print(ops.search_transactions(limit=5, category="Technology", country="India")
          [["order_date", "customer_name", "product_name", "sales"]])

    step("5. RETRIEVAL: date range 2014-12-01 .. 2014-12-07")
    r = ops.get_by_date_range("2014-12-01", "2014-12-07")
    print(f"{len(r):,} transactions, revenue ${r['sales'].sum():,.2f}")

    step("6. AGGREGATION: overall KPIs ($group)")
    for k, v in sales_analysis.overall_kpis().items():
        print(f"  {k:24s} {v}")

    step("7. AGGREGATION: revenue by category ($group + $sort)")
    print(category_analysis.category_sales()[["category", "revenue", "profit", "revenue_share_pct"]])

    step("8. AGGREGATION: top 5 countries")
    print(category_analysis.country_sales(limit=5)[["country", "revenue", "orders", "revenue_share_pct"]])

    step("9. AGGREGATION: top 10 products by revenue ($group + $sort + $limit)")
    print(ops.get_top_products(10)[["product_name", "category", "total_sales", "total_quantity"]])

    step("10. AGGREGATION: last 6 months with growth")
    print(sales_analysis.monthly_sales().tail(6)[["year_month", "revenue", "growth_pct", "trend"]])

    step("11. AGGREGATION: top 5 customers")
    print(customer_analysis.high_value_customers(5)[["customer_id", "customer_name", "country", "total_sales"]])

    step("12. STORED ML RESULTS")
    for name in [config.SALES_FORECASTS, config.CUSTOMER_SEGMENTS, config.CHURN_PREDICTIONS,
                 config.SALES_ANOMALIES, config.PRODUCT_RECOMMENDATIONS]:
        print(f"  {name:26s} {db[name].count_documents({}):>6,} documents")
    fc = pd.DataFrame(list(db[config.SALES_FORECASTS].find({}, {"_id": 0, "forecast_date": 1,
                                                                 "predicted_sales": 1}).sort("forecast_date", 1).limit(7)))
    if not fc.empty:
        print("\nFirst week of the stored forecast:\n", fc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
