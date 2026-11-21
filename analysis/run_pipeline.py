"""Run every analysis and machine learning step and store the results in MongoDB.

    python -m analysis.run_pipeline

Flow: MongoDB (sales_transactions) -> aggregation / retrieval -> processing ->
existing ML models -> results -> MongoDB (sales_analysis, sales_forecasts,
customer_segments, churn_predictions, sales_anomalies, product_recommendations).

It also writes CSV copies of the main results to ``results/`` as sample outputs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from analysis import (anomaly_detection, category_analysis, customer_analysis,  # noqa: E402
                      forecasting, product_analysis, sales_analysis)
from analysis.common import save_analysis  # noqa: E402
from database.mongodb_connection import get_database, initialize_database  # noqa: E402


def _save_csv(df: pd.DataFrame, name: str, write: bool) -> None:
    if write and df is not None and not df.empty:
        config.RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(config.RESULTS_DIR / name, index=False)


def run_descriptive(db, scope_market: Optional[str] = None, write_csv: bool = True) -> dict:
    """Analyses 1-6 (database-side aggregations) for one scope (ALL or a market)."""
    scope = scope_market or "ALL"
    kpis = sales_analysis.overall_kpis(scope_market, db=db)
    save_analysis("overall_kpis", kpis, scope, db)

    monthly = sales_analysis.monthly_sales(scope_market, db=db)
    save_analysis("monthly_sales", monthly, scope, db)
    yearly = sales_analysis.yearly_sales(scope_market, db=db)
    save_analysis("yearly_sales", yearly, scope, db)
    weekly = sales_analysis.weekly_sales(scope_market, db=db)
    save_analysis("weekly_sales", weekly, scope, db)
    save_analysis("segment_sales", sales_analysis.segment_sales(scope_market, db=db), scope, db)

    cats = category_analysis.category_sales(scope_market, db=db)
    save_analysis("category_sales", cats, scope, db, meta={"top": category_analysis.best_of(cats, "category")})
    save_analysis("sub_category_sales", category_analysis.sub_category_sales(scope_market, db=db), scope, db)
    countries = category_analysis.country_sales(scope_market, db=db)
    save_analysis("country_sales", countries, scope, db,
                  meta={"top": category_analysis.best_of(countries, "country")})
    if scope == "ALL":
        markets = category_analysis.market_sales(db=db)
        save_analysis("market_sales", markets, scope, db, meta={"top": category_analysis.best_of(markets, "market")})
        _save_csv(markets, "market_sales.csv", write_csv)

    top_rev = product_analysis.top_products(10, "revenue", scope_market, db=db)
    top_qty = product_analysis.top_products(10, "quantity", scope_market, db=db)
    save_analysis("top_products_revenue", top_rev, scope, db)
    save_analysis("top_products_quantity", top_qty, scope, db)

    hv = customer_analysis.high_value_customers(10, scope_market, db=db)
    save_analysis("high_value_customers", hv, scope, db)

    if scope == "ALL":
        _save_csv(monthly, "monthly_sales.csv", write_csv)
        _save_csv(cats, "category_sales.csv", write_csv)
        _save_csv(countries, "country_sales.csv", write_csv)
        _save_csv(top_rev, "top10_products_revenue.csv", write_csv)
        _save_csv(hv, "high_value_customers.csv", write_csv)
    return {"kpis": kpis, "months": len(monthly), "categories": len(cats), "countries": len(countries)}


def run_ml(db, market: Optional[str] = None, write_csv: bool = True) -> dict:
    """Analyses 6-7 and the predictive modules, using the existing trained models."""
    market = market or config.ML_MARKET
    out: dict = {"ml_market": market}

    summary = customer_analysis.customer_summary(market, db=db)
    segments = customer_analysis.segment_customers(market, db=db, summary=summary)
    profile = customer_analysis.segment_profile(segments)
    customer_analysis.store_segments(segments, market, db)
    save_analysis("segment_profile", profile, market, db)
    out["segments"] = profile[["segment_name", "customers"]].to_dict("records")

    churn = customer_analysis.predict_churn(market, db=db, summary=summary)
    customer_analysis.store_churn(churn, market, db)
    churn_info = customer_analysis.churn_summary(churn)
    save_analysis("churn_summary", churn_info, market, db)
    out["churn"] = {k: v for k, v in churn_info.items() if k != "note"}

    daily = sales_analysis.daily_sales(market, db=db)
    fc = forecasting.forecast_sales(30, market, db=db, history=daily)
    forecasting.store_forecast(fc, market, db)
    bt = forecasting.backtest(60, market, db=db, history=daily)
    series = bt.pop("series")
    save_analysis("forecast_backtest", {**bt, "series": series}, market, db)
    out["forecast"] = {"days": len(fc), "backtest": bt}

    d_anom, d_lim = anomaly_detection.daily_anomalies(market, db=db, daily=daily)
    t_anom, t_lim = anomaly_detection.transaction_anomalies("ALL", db=db)
    anomaly_detection.store_anomalies(d_anom, d_lim, t_anom, t_lim, market, "ALL", db)
    save_analysis("anomaly_limits", {"daily": d_lim, "transaction": t_lim}, market, db)
    out["anomalies"] = {"daily": int(d_anom["anomaly"].sum()), "transaction": int(len(t_anom))}

    rules = product_analysis.association_rules(market, db=db)
    product_analysis.store_recommendations(rules, market, db)
    save_analysis("recommendation_summary", {
        "rules": int(len(rules)), "products": int(rules["product_name"].nunique()),
        "total_orders": rules.attrs.get("total_orders"), "multi_item_orders": rules.attrs.get("multi_item_orders"),
        "most_recommended": product_analysis.most_recommended(rules, 10).to_dict("records"),
    }, market, db)
    out["recommendations"] = int(len(rules))

    _save_csv(segments, "customer_segments.csv", write_csv)
    _save_csv(churn[["customer_id", "customer_name", "churn_probability", "risk_level", "ActualChurn"]]
              .sort_values("churn_probability", ascending=False), "churn_predictions.csv", write_csv)
    _save_csv(fc, "sales_forecast_30_days.csv", write_csv)
    _save_csv(d_anom[d_anom["anomaly"]], "daily_sales_anomalies.csv", write_csv)
    _save_csv(t_anom, "transaction_anomalies.csv", write_csv)
    _save_csv(rules.head(200), "product_recommendations_sample.csv", write_csv)
    return out


def run_all(db=None, write_csv: bool = True, verbose: bool = True) -> dict:
    db = db if db is not None else get_database()
    initialize_database(db)
    if db[config.SALES_TRANSACTIONS].estimated_document_count() == 0:
        raise RuntimeError("sales_transactions is empty. Run `python -m database.import_data` first.")
    t0 = time.perf_counter()
    result = {"descriptive_all": run_descriptive(db, None, write_csv)}
    for market in category_analysis.market_sales(db=db)["market"]:
        run_descriptive(db, market, write_csv=False)
    result["ml"] = run_ml(db, config.ML_MARKET, write_csv)
    result["seconds"] = round(time.perf_counter() - t0, 2)
    if verbose:
        print(json.dumps(result, indent=2, default=str))
        counts = {c: db[c].count_documents({}) for c in sorted(db.list_collection_names())}
        print("Documents per collection:", json.dumps(counts, indent=2))
    return result


if __name__ == "__main__":
    try:
        run_all()
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
