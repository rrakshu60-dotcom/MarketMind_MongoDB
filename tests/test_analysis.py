"""Tests for the aggregation pipelines and the ML integration.

Aggregation results are compared with an independent pandas calculation on the
CSV. ML results are compared with the outputs the internship project saved
(models/reference/*.csv), which proves the MongoDB-fed pipeline reproduces them.
"""
import os
import pickle

import pandas as pd
import pytest

import config
from analysis import (anomaly_detection, category_analysis, customer_analysis, forecasting,
                      product_analysis, sales_analysis)
from analysis.model_loader import UnsafeModelError, safe_load
from analysis.run_pipeline import run_all

REF = config.REFERENCE_DIR


def test_overall_kpis_match_pandas(db, imported, source_df):
    k = sales_analysis.overall_kpis(db=db)
    assert k["total_transactions"] == len(source_df)
    assert k["total_revenue"] == pytest.approx(source_df["Total amount"].sum(), abs=0.01)
    assert k["total_profit"] == pytest.approx(source_df["Profit"].sum(), abs=0.01)
    assert k["total_quantity"] == source_df["Quantity"].sum()
    assert k["total_customers"] == source_df["Customer ID"].nunique()
    assert k["avg_transaction_value"] == pytest.approx(source_df["Total amount"].mean(), abs=0.01)


def test_market_scope_filters(db, imported, source_df):
    us = sales_analysis.overall_kpis("US", db=db)
    assert us["total_transactions"] == (source_df["Market"] == "US").sum() == 9994


def test_category_sales(db, imported, source_df):
    got = category_analysis.category_sales(db=db).set_index("category")["revenue"]
    exp = source_df.groupby("Category")["Total amount"].sum()
    for cat in exp.index:
        assert got[cat] == pytest.approx(exp[cat], abs=0.01)
    assert category_analysis.best_of(category_analysis.category_sales(db=db), "category")["category"] == exp.idxmax()


def test_country_sales(db, imported, source_df):
    got = category_analysis.country_sales(db=db)
    exp = source_df.groupby("Country")["Total amount"].sum().sort_values(ascending=False)
    assert len(got) == len(exp) == 147
    assert got.iloc[0]["country"] == exp.index[0]
    assert got["revenue_share_pct"].sum() == pytest.approx(100, abs=0.1)


def test_monthly_sales_and_growth(db, imported, source_df):
    m = sales_analysis.monthly_sales(db=db)
    exp = source_df.groupby(source_df["Order Date"].dt.to_period("M"))["Total amount"].sum()
    assert len(m) == len(exp) == 48
    assert list(m["revenue"].round(0)) == list(exp.round(2).round(0))
    g = m.iloc[1]["growth_pct"]
    assert g == pytest.approx(100 * (exp.iloc[1] / exp.iloc[0] - 1), abs=0.01)
    assert set(m["trend"]) <= {"start", "growth", "decline", "flat"}


def test_high_value_customers(db, imported, source_df):
    hv = customer_analysis.high_value_customers(5, db=db)
    exp = source_df.groupby("Customer ID")["Total amount"].sum().sort_values(ascending=False).head(5)
    assert list(hv["customer_id"]) == list(exp.index)


def test_segmentation_reproduces_original_model_output(db, imported):
    seg = customer_analysis.segment_customers("US", db=db)
    ref = pd.read_csv(REF / "customer_segment.csv")
    merged = seg.merge(ref, left_on="customer_id", right_on="Customer ID")
    assert len(seg) == len(ref) == len(merged) == 793
    assert (merged["cluster"] == merged["Cluster"]).all()


def test_churn_reproduces_original_model_output(db, imported):
    churn = customer_analysis.predict_churn("US", db=db)
    ref = pd.read_csv(REF / "churn_customers.csv")
    merged = churn.merge(ref, left_on="customer_id", right_on="Customer ID")
    assert len(merged) == 793
    assert (merged["churn_probability"] - merged["PredictedChurnProb"]).abs().max() < 1e-3
    assert (merged["ActualChurn_x"] == merged["ActualChurn_y"]).all()


def test_daily_anomalies_match_original(db, imported):
    d, lim = anomaly_detection.daily_anomalies("US", db=db)
    ref = pd.read_csv(REF / "anomaly_detection_results.csv", parse_dates=["Order Date"])
    assert int(d["anomaly"].sum()) == int(ref["Anomaly"].sum()) == 22
    assert set(d.loc[d["anomaly"], "order_date"]) == set(ref.loc[ref["Anomaly"], "Order Date"])
    assert lim["upper"] == pytest.approx(8462.84, abs=0.01)


def test_transaction_anomalies(db, imported):
    t, lim = anomaly_detection.transaction_anomalies("ALL", db=db)
    assert len(t) > 0 and (t["sales"] > lim["upper"]).all()
    assert lim["transactions_checked"] == 51290


def test_recommendations_match_original(db, imported):
    rules = product_analysis.association_rules("US", db=db)
    ref = pd.read_csv(REF / "product_recommendations.csv")
    assert len(rules) == len(ref)
    assert rules["product_name"].nunique() == ref["Product Name"].nunique()
    merged = rules.merge(ref, left_on=["product_name", "recommended_product"],
                         right_on=["Product Name", "Recommended Product"])
    # Ties in co-occurrence can be ordered differently; every shared pair has identical metrics.
    assert len(merged) > 0.5 * len(ref)
    assert (merged["lift"].round(2) == merged["Lift"]).all()
    assert (merged["confidence"].round(4) == merged["Confidence"]).all()


def test_forecast(db, imported):
    fc = forecasting.forecast_sales(30, "US", db=db)
    assert len(fc) == 30 and (fc["predicted_sales"] >= 0).all()
    assert fc["forecast_date"].min() == pd.Timestamp("2015-01-01")
    with pytest.raises(ValueError):
        forecasting.forecast_sales(0, "US", db=db)
    bt = forecasting.backtest(60, "US", db=db)
    assert 0 < bt["wape_pct"] < 100


def test_pipeline_stores_results_in_mongodb(db, imported):
    result = run_all(db=db, write_csv=False, verbose=False)
    assert result["ml"]["recommendations"] > 0
    assert db[config.SALES_FORECASTS].count_documents({}) == 30
    assert db[config.CUSTOMER_SEGMENTS].count_documents({}) == 793
    assert db[config.CHURN_PREDICTIONS].count_documents({}) == 793
    assert db[config.SALES_ANOMALIES].count_documents({"level": "daily"}) == 22
    assert db[config.SALES_ANALYSIS].find_one({"analysis_type": "overall_kpis", "scope": "ALL"})["data"][
        "total_transactions"] == 51290
    # running again replaces results instead of piling up duplicates
    run_all(db=db, write_csv=False, verbose=False)
    assert db[config.SALES_FORECASTS].count_documents({}) == 30


def test_pipeline_refuses_empty_database(client):
    empty = client["MarketMindDB_empty_test"]
    try:
        with pytest.raises(RuntimeError, match="empty"):
            run_all(db=empty, write_csv=False, verbose=False)
    finally:
        client.drop_database("MarketMindDB_empty_test")


def test_model_loader_blocks_unsafe_pickles(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(os.getcwd))  # harmless stand-in for a code-running pickle
    with pytest.raises(UnsafeModelError):
        safe_load(p)
