# E-Commerce Customer Analysis (CA2 - MLE)

Basic data analysis and visualization of an online retailer's sales, followed by a simple RFM + K-Means customer segmentation.

## Dataset
[Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI Machine Learning Repository, CC BY 4.0), used unchanged in `data/online_retail.csv`.
* 541,909 rows, 8 columns, 01-12-2010 to 09-12-2011
* After cleaning: 397,884 rows, 4,338 customers, total revenue about 8.9 million

## What the code does (`customer_analysis.py`)
1. Load the data and check shape and missing values
2. Clean it (drop missing `CustomerID`, returns, zero prices) and add `TotalPrice`
3. Basic analysis with plots: monthly revenue, top countries, top products, orders by weekday
4. Build the RFM table (Recency, Frequency, Monetary) per customer
5. Simple K-Means (k = 4) on log-scaled RFM, with an elbow plot
6. Profile each segment and plot customers vs revenue share

## Results
| Cluster | Customers | Avg recency (days) | Avg orders | Avg spend | Meaning |
|---|---|---|---|---|---|
| 1 | 16.5% | 12 | 13.7 | 8,074 | Best customers (about 65% of revenue) |
| 2 | 27.0% | 71 | 4.1 | 1,803 | Regular customers |
| 0 | 19.3% | 18 | 2.1 | 552 | New customers |
| 3 | 37.2% | 183 | 1.3 | 344 | Lost customers |

## Plots
Saved in `images/`: `monthly_revenue`, `top_countries`, `top_products`, `orders_by_weekday`, `rfm_distributions`, `elbow`, `customers_vs_revenue`, `recency_vs_monetary`.

## How to run
```bash
pip install -r requirements.txt
python customer_analysis.py
```
Output: plots in `images/` and the segment table in `rfm_segments.csv`.

## Credit
Based on the idea of RFM customer segmentation on the UCI Online Retail dataset; code rewritten in a simpler form.
