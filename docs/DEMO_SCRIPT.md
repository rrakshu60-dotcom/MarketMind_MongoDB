# Demo script: presenting MarketMind AI (MongoDB) to faculty

About 8–10 minutes. Follow the five activity requirements in order, so the examiner can tick each
one off.

## Before you start (5 minutes beforehand)

1. Make sure MongoDB is running: local MongoDB service, or an Atlas cluster with your IP allowed.
2. Open a terminal in the project folder and activate the virtual environment.
3. Optional, for a clean live import: `python -m database.import_data --reset`. Otherwise just make
   sure the data is already imported.
4. Run `streamlit run dashboard/app.py` and keep the browser tab open on **Dashboard Overview**.
5. Open MongoDB Compass (or the Atlas "Browse Collections" page) connected to `MarketMindDB`.

---

## 1. Introduction (1 minute)

> "This is MarketMind AI, a sales intelligence platform I worked on during my Infosys internship.
> For this activity I made MongoDB the central database. The flow is: collect the data, store it in
> MongoDB, retrieve it, process and analyse it, including our existing machine learning models, and
> show the results on a dashboard."

Show the architecture diagram in `README.md` (section 4).

## 2. Requirement 1: Collect / import the dataset (1 minute)

> "The dataset is Global Superstore: 51,290 sales transactions from 147 countries between 2011 and
> 2014, with 25 fields covering orders, customers, products, sales, profit, dates and geography."

Open `dataset/global_superstore_sales.csv` briefly, or show the table in README section 5.

## 3. Requirement 2: Store in MongoDB (2 minutes)

Run:
```bash
python -m database.import_data
```
Point out:
- "It validates and cleans every row: dates, numbers, quantities and discounts. Invalid rows would
  go to a rejects file."
- "It writes in batches of 5,000 using bulk_write."
- **Run it again** and say: "The second run inserts 0 documents, because the original Row ID is the
  MongoDB `_id`. Re-importing can never create duplicates."

Then in Compass, or the dashboard's **MongoDB Status** section, show:
- the `sales_transactions` collection with 51,290 documents and a sample document (real dates and
  numbers);
- the indexes, and the schema validator (Compass → Validation tab).

## 4. Requirement 3: Retrieve records (1.5 minutes)

Dashboard → **Transaction Explorer**:
1. Type customer ID `TB-11280` and click **Search MongoDB**. It shows 15 transactions.
2. Clear it, choose Category = Furniture, Country = France and a 2013 date range, then search.
3. Open **"MongoDB query used"**. "This is the exact `find()` filter sent to MongoDB."

Optional terminal view: `python scripts/demo_mongodb_operations.py` (steps 1–5).

## 5. Requirement 4: Process and analyse (2.5 minutes)

**Aggregation pipelines.** Dashboard → **Dashboard Overview**, then **Sales Analytics**:
- "These KPIs are calculated live by a `$group` aggregation inside MongoDB: $12.64M revenue,
  $1.47M profit, 51,290 transactions, 1,590 customers."
- Monthly trend: "Revenue grew every year, by 18%, 27% and 26%, with a seasonal peak from
  September to December."
- Category: "Technology is the top category at 37.5% of revenue."
- Country: "The US is the top country, and APAC is the top market."
- Top 10 products: "These come from `$group` → `$sort` → `$limit 10`."
- Change the **Market** filter in the sidebar to show the numbers updating from MongoDB.

**Machine learning.** Dashboard → **Customer Analytics** and **Predictive Analytics**:
- "These are the trained models from the internship. I didn't retrain them. The input features are
  built from MongoDB aggregations, and the predictions are stored back in MongoDB collections."
- Segmentation: "K-Means puts customers into three groups. It gives exactly the same clusters as
  the original notebook output, for all 793 customers."
- Churn: "The Random Forest ranks customers by churn risk." **Mention the caveat:** "The label is
  based on 180 days of inactivity, and one feature uses recency, so the accuracy is optimistic. I
  treat it as a ranking."
- Forecast: "CatBoost forecasts the next 30 days, and the backtest shows about 37% weighted error
  on daily sales."
- Anomalies: "There are 22 unusual sales days, the same as the original analysis, plus 28 unusual
  single transactions."
- Recommendations: pick "Staples" from the list and explain confidence and lift.

Optionally click **Re-run analytics & ML pipeline** in the sidebar: "This recomputes everything
from MongoDB and overwrites the stored results."

## 6. Requirement 5: Display results (covered above, 30 seconds)

> "Every chart and table on this dashboard is read from MongoDB, either live through aggregation or
> from the result collections. Nothing is hard-coded."

Show the result collections in Compass: `sales_analysis`, `sales_forecasts`, `customer_segments`,
`churn_predictions`, `sales_anomalies`, `product_recommendations`.

## 7. Testing and error handling (1 minute)

```bash
python -m pytest -v
```
> "There are 32 tests, run against a real MongoDB database. They check the import count,
> duplicate prevention, that every query and aggregation matches an independent pandas
> calculation, and that the ML results match the original project."

Optional: stop MongoDB and refresh the dashboard to show the friendly connection error, then
start it again.

## 8. Closing (30 seconds)

> "In summary: a real dataset, stored in MongoDB with validation and indexes, retrieved with
> queries, processed with aggregation pipelines and our trained models, and visualised on a
> dashboard. To be honest about scale, 51,000 records is moderate. The project shows the MongoDB
> techniques, not distributed processing. Next steps would be sharding on Atlas and change streams
> for real-time updates."

---

## Likely questions and short answers

| Question | Answer |
|---|---|
| Why MongoDB instead of MySQL? | Each sale is a self-contained document, the fields vary by country (postal codes), and the aggregation pipeline lets us summarise on the server. The schema can evolve without migrations. |
| How do you prevent duplicates? | The original Row ID is `_id`, and the import uses upserts. Running it twice gives the same 51,290 documents. |
| What does the aggregation pipeline do? | A sequence of stages (`$match`, `$group`, `$sort`, `$limit`, `$project`) executed inside MongoDB, so only the small summary is sent back. |
| Which indexes did you create and why? | order_date (date filters and trends), customer_id + order_date (customer search), category, country, market + order_date, product_id, order_id, and sales descending (top-N and outliers). |
| Is this big data? | It is a moderate dataset (51,290 records). The design uses big-data practices: batching, indexing and server-side aggregation. It could scale with Atlas sharding, but I didn't implement distributed processing. |
| How do you know the ML integration is correct? | The tests compare the outputs with the internship's saved results: 100% identical clusters, the same 22 anomaly dates, and churn probabilities within 0.00005. |
| Where are the credentials? | In a `.env` file that git ignores. `.env.example` shows the format. Nothing is hard-coded. |
