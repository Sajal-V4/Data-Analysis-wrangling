# Sales Data Wrangling, Preprocessing & Customer Segmentation

A small end-to-end project demonstrating data cleaning, feature engineering,
visualization, and machine learning on retail sales data.

**Stack:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## Objective

Build a reusable, production-style pipeline that turns messy, raw
transactional sales data into clean data, visual insights, and
data-driven customer segments — with logging and error handling so
failures are diagnosable rather than silent crashes.

## Problem statement

Raw sales exports are rarely analysis-ready: duplicate records, missing
fields, inconsistent category labels (`"Tech"` vs `"Technology"`), and
outliers all distort reporting and any model built on top of them.
Beyond the data quality issue, a business selling to hundreds of
customers can't treat them all the same — some are high-value repeat
buyers, others are lapsing and at risk of churn. This project addresses
both: it cleans and validates the raw data end-to-end, then applies
unsupervised learning (RFM feature engineering + KMeans) to group
customers by purchase behavior so marketing/retention effort can be
targeted by segment instead of applied uniformly.

## What it does

1. **Generates a messy raw dataset** (`raw_sales_data.csv`) — synthetic retail
   sales data with the problems real-world exports have: duplicate rows,
   missing values, inconsistent category casing (`"Tech"` vs `"Technology"`),
   and injected outliers.

2. **Cleans & preprocesses it** with pandas/numpy:
   - Drops duplicate rows
   - Standardizes inconsistent text fields
   - Imputes missing `Sales` using category-wise median
   - Fills missing `Discount` with 0
   - Caps outliers in `Sales` using the IQR method (winsorizing, not deletion)
   - Fixes data types (dates, integers)

3. **Engineers features**: order month, order weekday, net sales
   (after discount), profit margin.

4. **Visualizes** (`eda_overview.png`) — a 4-panel figure:
   - Monthly net sales trend
   - Total sales by category
   - Raw vs. cleaned sales distribution (shows the outlier fix)
   - Total sales by region

5. **Applies scikit-learn** for customer segmentation:
   - Builds RFM (Recency, Frequency, Monetary) features per customer
   - Scales them with `StandardScaler`
   - Runs `KMeans` (k=4) to segment customers into Champions / Loyal /
     At Risk / Low Value
   - Uses `PCA` to reduce the 3 RFM dimensions to 2D for plotting
     (`customer_segments.png`)
   - Compares segment profiles in a normalized bar chart
     (`segment_profiles.png`)

## Files

| File | Description |
|---|---|
| `sales_insights_project.py` | The full pipeline script — run it to regenerate everything |
| `raw_sales_data.csv` | Synthetic messy dataset before cleaning |
| `cleaned_sales_data.csv` | Dataset after cleaning + feature engineering |
| `customer_segments.csv` | Per-customer RFM values and assigned segment |
| `eda_overview.png` | 4-panel exploratory data analysis chart |
| `customer_segments.png` | PCA scatter plot of the 4 customer segments |
| `segment_profiles.png` | Normalized Recency/Frequency/Monetary comparison across segments |
| `pipeline.log` | Log of each pipeline stage, including any recoverable warnings |

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python sales_insights_project.py
```

Outputs are written to an `outputs/` folder next to the script; progress and
any warnings are also written to `pipeline.log`.

To run it on your own CSV instead of the synthetic sample data:

```python
from sales_insights_project import main
main("path/to/your_sales_data.csv")
```

## Error handling

The pipeline is broken into small functions (`load_data`, `clean_data`,
`engineer_features`, `segment_customers`, etc.), each wrapped in
try/except so a problem in one stage is caught, logged with context, and
either recovered from or reported clearly — instead of crashing with a
raw traceback:

- **Missing/invalid input file** → raises a clear `PipelineError` instead
  of an unhandled `FileNotFoundError`.
- **Missing required columns** → checked up front and reported by name.
- **Unparseable dates** → coerced to `NaT`, counted, logged, and dropped
  rather than silently corrupting downstream date-based features.
- **A category with no valid `Sales` values to compute a median from** →
  falls back to the global median instead of leaving `NaN`s behind.
- **More cluster requested than customers available** → `n_clusters` is
  automatically capped to the number of customers.
- **Plotting failures** → logged as errors without stopping the rest of
  the pipeline (e.g. a chart failing to render doesn't block the CSV
  outputs).
- `main()` returns exit code `0` on success and `1` on failure, so it
  can be used directly in a scheduled job or CI pipeline.

## Notes / ideas for extending it

- Swap the synthetic data generator for a real dataset (e.g. a Superstore-style
  export) — the cleaning → feature engineering → visualization → segmentation
  pipeline carries over largely unchanged.
- Try a different `k` for KMeans and use the elbow method or silhouette score
  to justify the choice.
- Add a regression model (e.g. `LinearRegression` or `RandomForestRegressor`)
  to predict `Net_Sales` or `Profit_Margin` from category/region/discount.
