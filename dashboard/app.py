"""MarketMind AI - Small Business Sales Intelligence Dashboard (MongoDB edition).

Run from the project folder:

    streamlit run dashboard/app.py

Every number on this dashboard is read from MongoDB:
- KPIs, trends, categories, countries, products and customers come from live
  aggregation pipelines on ``sales_transactions``;
- segments, churn, forecasts, anomalies and recommendations come from the
  collections written by ``python -m analysis.run_pipeline`` (which reads its
  input from MongoDB too). A button in the sidebar re-runs that pipeline.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from analysis import category_analysis, customer_analysis, sales_analysis  # noqa: E402
from analysis.common import load_analysis  # noqa: E402
from database import database_operations as ops  # noqa: E402
from database.mongodb_connection import DatabaseConnectionError, check_connection, get_database  # noqa: E402

st.set_page_config(page_title="MarketMind AI Dashboard", page_icon="📊", layout="wide")

# Validated categorical palette (light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRID = "#e6e5e0"
MUTED = "#52514e"
CACHE_TTL = 300

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: var(--secondary-background-color); border-radius: 10px;
  padding: 14px 16px; border: 1px solid rgba(128,128,128,0.18);}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem;}
.mm-source {font-size: 0.8rem; opacity: 0.7; margin-top: -0.4rem;}
.mm-note {font-size: 0.9rem; padding: 10px 14px; border-left: 3px solid #2a78d6;
  background: var(--secondary-background-color); border-radius: 4px; margin: 4px 0 12px 0;}
</style>
""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), showlegend=legend,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font=dict(color=MUTED))
    return fig


def money(v: float) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.1f}K"
    return f"${v:,.2f}"


def show_df(df: pd.DataFrame, **kwargs) -> None:
    """st.dataframe with dates shown as YYYY-MM-DD instead of full timestamps."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    st.dataframe(df, **kwargs)


def source(text: str) -> None:
    st.markdown(f"<div class='mm-source'>Source: {text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached MongoDB reads
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_kpis(market):
    return sales_analysis.overall_kpis(market)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_monthly(market):
    return sales_analysis.monthly_sales(market)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_categories(market):
    return category_analysis.category_sales(market), category_analysis.sub_category_sales(market)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_countries(market):
    return category_analysis.country_sales(market)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_markets():
    return category_analysis.market_sales()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_top_products(market, by):
    return ops.get_top_products(10, by, query={"market": market} if market else None)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_high_value(market, n):
    return customer_analysis.high_value_customers(n, market)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_collection(name, query=None, sort=None, limit=0):
    cur = get_database()[name].find(query or {}, {"_id": 0})
    if sort:
        cur = cur.sort(sort)
    return pd.DataFrame(list(cur.limit(limit)))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_analysis(kind, scope):
    return load_analysis(kind, scope)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def q_filters():
    return {
        "markets": ops.distinct_values("market"),
        "categories": ops.distinct_values("category"),
        "countries": ops.distinct_values("country"),
        "bounds": ops.date_bounds(),
        "count": ops.count_transactions(),
    }


# ---------------------------------------------------------------------------
# Connection check
# ---------------------------------------------------------------------------

try:
    conn = check_connection()
    filters = q_filters()
except DatabaseConnectionError as exc:
    st.title("MarketMind AI – Small Business Sales Intelligence Dashboard")
    st.error(f"Cannot connect to MongoDB. {exc}")
    st.markdown(
        "**How to fix it**\n\n"
        "1. Start MongoDB (local) or check your Atlas cluster is running and your IP is allowed.\n"
        "2. Check `MONGO_URI` in your `.env` file (see `.env.example`).\n"
        "3. Reload this page."
    )
    st.stop()

if filters["count"] == 0:
    st.title("MarketMind AI – Small Business Sales Intelligence Dashboard")
    st.warning("MongoDB is connected, but `sales_transactions` is empty. Run "
               "`python -m database.import_data` and then `python -m analysis.run_pipeline`.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 MarketMind AI")
    st.caption("Sales intelligence on MongoDB")
    section = st.radio(
        "Section",
        ["Dashboard Overview", "Sales Analytics", "Customer Analytics", "Predictive Analytics",
         "Transaction Explorer", "MongoDB Status"],
        label_visibility="collapsed",
    )
    st.divider()
    market_choice = st.selectbox("Market", ["All markets"] + filters["markets"],
                                 help="Filters the overview and sales analytics")
    market = None if market_choice == "All markets" else market_choice
    st.divider()
    st.markdown(f"🟢 **Connected** · MongoDB {conn['server_version']}")
    st.caption(f"Database `{config.MONGO_DB_NAME}` · {filters['count']:,} transactions")
    if st.button("Re-run analytics & ML pipeline", width="stretch",
                 help="Recomputes every stored result from MongoDB data"):
        from analysis.run_pipeline import run_all
        with st.spinner("Running aggregations and models on MongoDB data..."):
            run_all(write_csv=False, verbose=False)
        st.cache_data.clear()
        st.success("Pipeline finished. Results refreshed.")
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.title("MarketMind AI – Small Business Sales Intelligence Dashboard")
scope_label = market_choice if market else "all 7 markets"

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

if section == "Dashboard Overview":
    k = q_kpis(market)
    st.caption(f"Global Superstore sales, {k['first_date']:%b %Y} to {k['last_date']:%b %Y}, {scope_label}")
    c = st.columns(5)
    c[0].metric("Total Revenue", money(k["total_revenue"]))
    c[1].metric("Total Profit", money(k["total_profit"]))
    c[2].metric("Total Transactions", f"{k['total_transactions']:,}")
    c[3].metric("Total Customers", f"{k['total_customers']:,}")
    c[4].metric("Total Quantity Sold", f"{k['total_quantity']:,}")
    c = st.columns(5)
    c[0].metric("Orders", f"{k['total_orders']:,}")
    c[1].metric("Avg Transaction Value", money(k["avg_transaction_value"]))
    c[2].metric("Avg Order Value", money(k["avg_order_value"]))
    c[3].metric("Profit Margin", f"{k['profit_margin_pct']}%")
    c[4].metric("Products / Countries", f"{k['total_products']:,} / {k['total_countries']}")
    source("live $group aggregation on sales_transactions")

    left, right = st.columns([3, 2])
    with left:
        m = q_monthly(market)
        fig = px.line(m, x="year_month", y="revenue", markers=False,
                      labels={"year_month": "Month", "revenue": "Revenue ($)"},
                      title="Monthly revenue")
        fig.update_traces(line=dict(width=2, color=SERIES[0]),
                          hovertemplate="%{x}<br>Revenue: $%{y:,.0f}<extra></extra>")
        st.plotly_chart(style_fig(fig), width="stretch")
    with right:
        cats, _ = q_categories(market)
        fig = px.bar(cats, x="revenue", y="category", orientation="h", title="Revenue by category",
                     labels={"revenue": "Revenue ($)", "category": ""}, text="revenue_share_pct")
        fig.update_traces(marker_color=SERIES[0], texttemplate="%{text:.1f}%", textposition="outside",
                          cliponaxis=False, hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_xaxes(range=[0, cats["revenue"].max() * 1.18])
        st.plotly_chart(style_fig(fig), width="stretch")

    if not market:
        mk = q_markets()
        fig = px.bar(mk, x="market", y="revenue", title="Revenue by market",
                     labels={"market": "Market", "revenue": "Revenue ($)"}, text="revenue_share_pct")
        fig.update_traces(marker_color=SERIES[0], texttemplate="%{text:.1f}%", textposition="outside",
                          cliponaxis=False, hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

elif section == "Sales Analytics":
    st.caption(f"Scope: {scope_label}. All charts are MongoDB aggregation pipelines.")
    m = q_monthly(market)
    st.subheader("Monthly sales trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m["year_month"], y=m["revenue"], name="Monthly revenue",
                             line=dict(color=SERIES[0], width=2),
                             hovertemplate="%{x}<br>Revenue: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=m["year_month"], y=m["rolling_3m_revenue"], name="3-month average",
                             line=dict(color=SERIES[1], width=2, dash="dot"),
                             hovertemplate="%{x}<br>3-month avg: $%{y:,.0f}<extra></extra>"))
    fig.update_yaxes(title="Revenue ($)")
    st.plotly_chart(style_fig(fig, 400, legend=True), width="stretch")
    g = m.dropna(subset=["growth_pct"])
    growth_months = int((g["growth_pct"] > 0).sum())
    best = m.loc[m["revenue"].idxmax()]
    worst_drop = g.loc[g["growth_pct"].idxmin()]
    yearly = sales_analysis.yearly_sales(market)
    c = st.columns(4)
    c[0].metric(f"Best month ({best['year_month']})", money(best["revenue"]))
    c[1].metric("Months with growth", f"{growth_months} of {len(g)}")
    c[2].metric("Sharpest monthly drop", worst_drop["year_month"], f"{worst_drop['growth_pct']:.1f}%")
    if len(yearly) > 1:
        c[3].metric(f"{int(yearly.iloc[-1]['year'])} vs {int(yearly.iloc[-2]['year'])}",
                    money(yearly.iloc[-1]["revenue"]), f"{yearly.iloc[-1]['growth_pct']:.1f}%")
    with st.expander("Month-by-month table with growth / decline"):
        show_df(m[["year_month", "revenue", "profit", "orders", "growth_pct", "trend"]],
                     width="stretch", hide_index=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Category-wise revenue")
        cats, subs = q_categories(market)
        top = cats.iloc[0]
        st.markdown(f"<div class='mm-note'>Highest revenue category: <b>{top['category']}</b> "
                    f"({money(top['revenue'])}, {top['revenue_share_pct']}% of revenue)</div>",
                    unsafe_allow_html=True)
        fig = px.bar(subs, x="revenue", y="sub_category", color="category", orientation="h",
                     color_discrete_sequence=SERIES, labels={"revenue": "Revenue ($)", "sub_category": ""},
                     category_orders={"category": list(cats["category"])})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_traces(hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
        st.plotly_chart(style_fig(fig, 520, legend=True), width="stretch")
    with right:
        st.subheader("Country-wise sales")
        countries = q_countries(market)
        top = countries.iloc[0]
        st.markdown(f"<div class='mm-note'>Highest performing market: <b>{top['country']}</b> "
                    f"({money(top['revenue'])}, {top['revenue_share_pct']}% of revenue)</div>",
                    unsafe_allow_html=True)
        n = st.slider("Countries shown", 5, 30, 15)
        t = countries.head(n)
        fig = px.bar(t, x="revenue", y="country", orientation="h",
                     labels={"revenue": "Revenue ($)", "country": ""}, custom_data=["profit", "orders"])
        fig.update_traces(marker_color=SERIES[0],
                          hovertemplate="%{y}<br>Revenue $%{x:,.0f}<br>Profit $%{customdata[0]:,.0f}"
                                        "<br>Orders %{customdata[1]:,}<extra></extra>")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig, 460), width="stretch")
        if not market:
            fig = px.choropleth(countries, locations="country", locationmode="country names",
                                color="revenue", color_continuous_scale="Blues",
                                labels={"revenue": "Revenue ($)"})
            fig.update_layout(geo=dict(showframe=False, projection_type="natural earth"))
            st.plotly_chart(style_fig(fig, 300), width="stretch")

    st.divider()
    st.subheader("Top 10 products")
    by = st.radio("Rank by", ["revenue", "quantity"], horizontal=True)
    tp = q_top_products(market, by)
    metric = "total_sales" if by == "revenue" else "total_quantity"
    fig = px.bar(tp, x=metric, y="product_name", orientation="h", color="category",
                 color_discrete_sequence=SERIES,
                 labels={metric: "Revenue ($)" if by == "revenue" else "Units sold", "product_name": ""})
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(style_fig(fig, 440, legend=True), width="stretch")
    show_df(tp, width="stretch", hide_index=True)
    source("$group by product_name, $sort, $limit 10")

elif section == "Customer Analytics":
    st.subheader("High-value customers")
    n = st.slider("Number of customers", 5, 50, 10)
    hv = q_high_value(market, n)
    show_df(hv, width="stretch", hide_index=True,
                 column_config={"total_sales": st.column_config.NumberColumn("Total sales", format="$%.2f"),
                                "total_profit": st.column_config.NumberColumn("Total profit", format="$%.2f"),
                                "avg_order_value": st.column_config.NumberColumn("Avg order", format="$%.2f")})
    source(f"live aggregation, scope {scope_label}")

    st.divider()
    seg = q_collection(config.CUSTOMER_SEGMENTS)
    if seg.empty:
        st.info("No stored segments yet. Use 'Re-run analytics & ML pipeline' in the sidebar.")
    else:
        ml_market = seg["market_scope"].iloc[0]
        st.subheader(f"Customer segmentation (K-Means, {ml_market} market)")
        st.markdown("<div class='mm-note'>The internship project's trained K-Means model "
                    "(<code>kmeans_clustering_model.pkl</code>, k = 3) applied to customer features "
                    "aggregated from MongoDB.</div>", unsafe_allow_html=True)
        names = ["High-Value Customers", "Regular Customers", "Low-Value Customers"]
        cmap = dict(zip(names, SERIES[:3]))
        prof = customer_analysis.segment_profile(seg)
        c = st.columns(3)
        for i, r in enumerate(prof.itertuples()):
            c[i].metric(f"{r.segment_name} · avg spend {money(r.avg_total_spending)}", f"{r.customers} customers")
        left, right = st.columns([3, 2])
        with left:
            fig = px.scatter(seg, x="PurchaseFrequency", y="TotalSpending", color="segment_name",
                             color_discrete_map=cmap, category_orders={"segment_name": names},
                             hover_data=["customer_id", "customer_name", "AverageOrderValue"],
                             labels={"PurchaseFrequency": "Line items purchased",
                                     "TotalSpending": "Total spending ($)", "segment_name": "Segment"})
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
            st.plotly_chart(style_fig(fig, 420, legend=True), width="stretch")
        with right:
            fig = px.bar(prof, x="segment_name", y="total_revenue", color="segment_name",
                         color_discrete_map=cmap, labels={"segment_name": "", "total_revenue": "Revenue ($)"},
                         title="Revenue by segment")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
        with st.expander("Segment profiles and strategies"):
            prof["strategy"] = prof["cluster"].map(customer_analysis.CLUSTER_STRATEGY)
            show_df(prof, width="stretch", hide_index=True)

    st.divider()
    churn = q_collection(config.CHURN_PREDICTIONS, sort=[("churn_probability", -1)])
    if not churn.empty:
        st.subheader(f"Churn risk analysis (Random Forest, {churn['market_scope'].iloc[0]} market)")
        info = q_analysis("churn_summary", churn["market_scope"].iloc[0])["data"]
        c = st.columns(4)
        c[0].metric("Customers scored", info["customers"])
        c[1].metric("High risk", info["high_risk"])
        c[2].metric("Medium risk", info["medium_risk"])
        c[3].metric("Low risk", info["low_risk"])
        st.markdown(f"<div class='mm-note'><b>Read with care:</b> {info['note']}</div>", unsafe_allow_html=True)
        left, right = st.columns([2, 3])
        with left:
            order = ["High Risk", "Medium Risk", "Low Risk"]
            counts = churn["risk_level"].value_counts().reindex(order).fillna(0).reset_index()
            counts.columns = ["risk_level", "customers"]
            fig = px.bar(counts, x="risk_level", y="customers", labels={"risk_level": "", "customers": "Customers"},
                         color="risk_level", color_discrete_map={"High Risk": "#d03b3b", "Medium Risk": "#eda100",
                                                                 "Low Risk": "#2a78d6"}, text="customers")
            st.plotly_chart(style_fig(fig, 360), width="stretch")
        with right:
            risk = st.selectbox("Show", ["High Risk", "Medium Risk", "Low Risk"])
            show_df(churn[churn["risk_level"] == risk][
                ["customer_id", "customer_name", "churn_probability", "DaysSinceLastPurchase",
                 "PurchaseFrequency", "TotalSpending"]],
                width="stretch", hide_index=True, height=320)

elif section == "Predictive Analytics":
    fc = q_collection(config.SALES_FORECASTS, sort=[("forecast_date", 1)])
    st.subheader("Future sales forecast (CatBoost)")
    if fc.empty:
        st.info("No stored forecast yet. Use 'Re-run analytics & ML pipeline' in the sidebar.")
    else:
        ml_market = fc["market_scope"].iloc[0]
        daily = sales_analysis.daily_sales(ml_market)
        hist = daily[daily["order_date"] >= daily["order_date"].max() - pd.Timedelta(days=120)]
        bt = q_analysis("forecast_backtest", ml_market)["data"]
        c = st.columns(4)
        c[0].metric("Forecast horizon", f"{len(fc)} days")
        c[1].metric("Forecast total", money(fc["predicted_sales"].sum()))
        c[2].metric("Backtest WAPE (60 days)", f"{bt['wape_pct']}%")
        c[3].metric("Backtest weekly MAPE", f"{bt['weekly_mape_pct']}%")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["order_date"], y=hist["total_amount"], name="Actual daily sales",
                                 line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
        bts = pd.DataFrame(bt["series"])
        fig.add_trace(go.Scatter(x=bts["date"], y=bts["predicted"], name="Backtest (one day ahead)",
                                 line=dict(color=SERIES[2], width=2),
                                 hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fc["forecast_date"], y=fc["predicted_sales"], name="30-day forecast",
                                 line=dict(color=SERIES[1], width=2, dash="dash"),
                                 hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Daily sales ($)")
        st.plotly_chart(style_fig(fig, 420, legend=True), width="stretch")
        st.markdown(
            "<div class='mm-note'>Model: the internship project's <code>catboost_daily_model.pkl</code>, "
            f"fed daily totals aggregated from MongoDB ({ml_market} market). The backtest predicts each of "
            "the last 60 days from the real days before it. The model was trained on this same data, so "
            "that is an in-sample check. The 30-day forecast is recursive (each day uses earlier "
            "predictions), and it drifts towards low values after the first week. Use it for the short-term "
            "level, not exact daily amounts.</div>", unsafe_allow_html=True)
        with st.expander("Forecast table (sales_forecasts collection)"):
            show_df(fc[["forecast_date", "predicted_sales", "horizon_day", "model"]],
                         width="stretch", hide_index=True)

    st.divider()
    st.subheader("Sales anomaly detection")
    an = q_collection(config.SALES_ANOMALIES)
    if not an.empty:
        lims = q_analysis("anomaly_limits", config.ML_MARKET)["data"]
        d = an[an["level"] == "daily"].sort_values("date")
        t = an[an["level"] == "transaction"].sort_values("sales", ascending=False)
        tab1, tab2 = st.tabs([f"Unusual sales days ({len(d)})", f"Unusual transactions ({len(t)})"])
        with tab1:
            daily_all = sales_analysis.daily_sales(config.ML_MARKET)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_all["order_date"], y=daily_all["total_amount"], name="Daily sales",
                                     line=dict(color=SERIES[0], width=1.5),
                                     hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=d["date"], y=d["total_amount"], mode="markers", name="Anomaly",
                                     marker=dict(color="#d03b3b", size=9, line=dict(width=2, color="white")),
                                     hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f} (anomaly)<extra></extra>"))
            fig.add_hline(y=lims["daily"]["upper"], line_dash="dash", line_color=MUTED,
                          annotation_text=f"Upper limit {money(lims['daily']['upper'])}")
            fig.update_yaxes(title="Daily sales ($)")
            st.plotly_chart(style_fig(fig, 400, legend=True), width="stretch")
            st.caption(f"Reused method from the internship project: 3×IQR on daily totals, "
                       f"{config.ML_MARKET} market. Upper limit {money(lims['daily']['upper'])}.")
            show_df(d[["date", "total_amount", "z_score"]], width="stretch", hide_index=True)
        with tab2:
            st.caption(f"Added for this project: Tukey 1.5×IQR on log(1 + amount) across all "
                       f"{lims['transaction']['transactions_checked']:,} transactions. A sale above "
                       f"{money(lims['transaction']['upper'])} is flagged (median sale is "
                       f"{money(lims['transaction']['median'])}).")
            show_df(t[["row_id", "order_id", "date", "customer_name", "country", "product_name",
                            "sales", "quantity", "profit", "z_score", "severity"]],
                         width="stretch", hide_index=True)

    st.divider()
    st.subheader("Product recommendations")
    summ = q_analysis("recommendation_summary", config.ML_MARKET)
    if summ:
        s = summ["data"]
        st.markdown(f"<div class='mm-note'>Market-basket pairs from {s['total_orders']:,} orders in MongoDB "
                    f"({s['multi_item_orders']:,} contain two or more products). Confidence = share of orders with "
                    "product A that also contain B. Lift above 1 means B is bought with A more often than "
                    "by chance. Most pairs appear in only one or two orders, so treat individual rules as "
                    "suggestions, not certainties.</div>", unsafe_allow_html=True)
        products = sorted(get_database()[config.PRODUCT_RECOMMENDATIONS].distinct("product_name"))
        default = products.index("Staples") if "Staples" in products else 0
        prod = st.selectbox(f"Choose a product ({len(products):,} with recommendations)", products, index=default)
        rec = q_collection(config.PRODUCT_RECOMMENDATIONS, {"product_name": prod}, sort=[("rank", 1)])
        show_df(rec[["rank", "recommended_product", "co_occurrence", "confidence", "lift"]],
                     width="stretch", hide_index=True)
        mr = pd.DataFrame(s["most_recommended"])
        fig = px.bar(mr, x="total_co_occurrence", y="recommended_product", orientation="h",
                     title="Products recommended most often",
                     labels={"total_co_occurrence": "Co-occurrences", "recommended_product": ""})
        fig.update_traces(marker_color=SERIES[0])
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(style_fig(fig, 380), width="stretch")

elif section == "Transaction Explorer":
    st.subheader("Transaction Explorer")
    st.caption("Each search runs a MongoDB find() query with the filters below.")
    lo, hi = filters["bounds"]
    with st.form("explorer"):
        c = st.columns(3)
        cust = c[0].text_input("Customer ID", placeholder="e.g. TB-11280")
        cat = c[1].selectbox("Category", ["Any"] + filters["categories"])
        country = c[2].selectbox("Country", ["Any"] + filters["countries"])
        c = st.columns(3)
        start = c[0].date_input("From", value=lo.date(), min_value=lo.date(), max_value=hi.date())
        end = c[1].date_input("To", value=hi.date(), min_value=lo.date(), max_value=hi.date())
        product = c[2].text_input("Product name contains", placeholder="e.g. Phone")
        limit = st.select_slider("Maximum rows", [100, 500, 1000, 5000], value=500)
        submitted = st.form_submit_button("Search MongoDB", type="primary")
    try:
        query = ops.build_filter(customer_id=cust or None, category=None if cat == "Any" else cat,
                                 country=None if country == "Any" else country,
                                 start_date=start, end_date=end, product_search=product or None)
        total = ops.count_transactions(query)
        df = ops.search_transactions(limit=limit, customer_id=cust or None,
                                     category=None if cat == "Any" else cat,
                                     country=None if country == "Any" else country,
                                     start_date=start, end_date=end, product_search=product or None)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if total == 0:
        st.warning("No transactions match these filters." +
                   (" Check the Customer ID format, for example TB-11280." if cust else ""))
    else:
        c = st.columns(4)
        c[0].metric("Matching transactions", f"{total:,}")
        c[1].metric("Shown", f"{len(df):,}")
        c[2].metric("Revenue (shown rows)", money(df["sales"].sum()))
        c[3].metric("Profit (shown rows)", money(df["profit"].sum()))
        show_df(df, width="stretch", hide_index=True, height=480)
        with st.expander("MongoDB query used"):
            st.code(repr(query), language="python")
        st.download_button("Download shown rows as CSV", df.to_csv(index=False), "transactions.csv", "text/csv")

elif section == "MongoDB Status":
    db = get_database()
    st.subheader("MongoDB database status")
    st.write(f"Server version **{conn['server_version']}** · database **{db.name}** · URI `{conn['uri']}`")
    stats = []
    for name in sorted(db.list_collection_names()):
        s = db.command("collStats", name)
        stats.append({"collection": name, "documents": s.get("count", 0),
                      "size_MB": round(s.get("size", 0) / 1e6, 2),
                      "indexes": s.get("nindexes", 0)})
    show_df(pd.DataFrame(stats), width="stretch", hide_index=True)
    st.markdown("**Indexes on `sales_transactions`**")
    st.code("\n".join(f"{k}: {v['key']}" for k, v in db[config.SALES_TRANSACTIONS].index_information().items()))
    logs = list(db[config.IMPORT_LOG].find({}, {"_id": 0}).sort("finished_at", -1).limit(5))
    if logs:
        st.markdown("**Recent imports**")
        show_df(pd.DataFrame(logs)[["finished_at", "source_file", "rows_read", "rows_valid", "rows_rejected",
                                         "inserted", "updated", "unchanged", "documents_after", "seconds"]],
                     width="stretch", hide_index=True)
    st.markdown("**Sample document**")
    st.json(db[config.SALES_TRANSACTIONS].find_one({}, sort=[("_id", 1)]), expanded=False)
