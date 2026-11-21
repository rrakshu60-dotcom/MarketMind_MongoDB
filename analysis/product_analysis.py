"""Analysis 4 (top products) and product recommendations.

Recommendations reuse the internship project's market-basket method
(``AIML/week3/product_recommendation``): products bought in the same order are
counted as pairs, then scored with support, confidence and lift. Here the
baskets are built by a MongoDB ``$group`` on ``order_id`` instead of reading
the CSV.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Optional

import pandas as pd
from pymongo.database import Database

import config
from analysis.common import aggregate_df, records, replace_collection, scope_query, utcnow
from database.database_operations import get_top_products


def top_products(n: int = 10, by: str = "revenue", market: Optional[str] = None,
                 db: Optional[Database] = None) -> pd.DataFrame:
    return get_top_products(n=n, by=by, query=scope_query(market), db=db)


def order_baskets(market: Optional[str] = None, db: Optional[Database] = None) -> pd.DataFrame:
    """One row per order with the distinct products in it (MongoDB $group + $addToSet)."""
    pipeline = [
        {"$match": scope_query(market)},
        {"$group": {"_id": "$order_id", "products": {"$addToSet": "$product_name"}}},
        {"$project": {"_id": 0, "order_id": "$_id", "products": 1}},
    ]
    return aggregate_df(pipeline, db)


def association_rules(market: Optional[str] = None, db: Optional[Database] = None,
                      top_per_product: int = 5) -> pd.DataFrame:
    market = market or config.ML_MARKET
    baskets = order_baskets(market, db)
    if baskets.empty:
        return pd.DataFrame()
    total_orders = len(baskets)
    item_orders: Counter = Counter()
    pair_counts: Counter = Counter()
    for products in baskets["products"]:
        items = sorted(set(products))
        item_orders.update(items)
        if len(items) > 1:
            pair_counts.update(combinations(items, 2))

    rows = []
    for (a, b), n in pair_counts.items():
        for x, y in ((a, b), (b, a)):
            conf = n / item_orders[x]
            rows.append({
                "product_name": x, "recommended_product": y, "co_occurrence": n,
                "support": n / total_orders, "confidence": conf,
                "lift": conf / (item_orders[y] / total_orders),
            })
    rules = pd.DataFrame(rows)
    rules = (rules.sort_values(["product_name", "co_occurrence", "lift"], ascending=[True, False, False])
             .groupby("product_name", group_keys=False).head(top_per_product).reset_index(drop=True))
    rules["rank"] = rules.groupby("product_name").cumcount() + 1
    rules.attrs["total_orders"] = total_orders
    rules.attrs["multi_item_orders"] = int((baskets["products"].str.len() > 1).sum())
    return rules


def most_recommended(rules: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Products that show up most often as someone else's recommendation."""
    if rules.empty:
        return rules
    return (rules.groupby("recommended_product")["co_occurrence"].sum()
            .sort_values(ascending=False).head(n).reset_index()
            .rename(columns={"co_occurrence": "total_co_occurrence"}))


def store_recommendations(rules: pd.DataFrame, market: str, db: Optional[Database] = None) -> int:
    docs = records(rules.round({"support": 6, "confidence": 4, "lift": 2}))
    now = utcnow()
    for d in docs:
        d.update({"market_scope": market, "method": "market basket pairs (support, confidence, lift)",
                  "generated_at": now})
    return replace_collection(config.PRODUCT_RECOMMENDATIONS, docs, db)
