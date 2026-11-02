# E-Commerce Customer Analysis (basic data analysis + visualization)
# Dataset: Online Retail (UCI) - data/online_retail.csv

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
df = pd.read_csv("data/online_retail.csv", encoding="ISO-8859-1")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------------------
# 2. Clean the data
# ---------------------------------------------------------------
df = df.dropna(subset=["CustomerID"])               # no customer id
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]  # returns / bad prices
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

print("\nRows after cleaning:", len(df))
print("Customers:", df["CustomerID"].nunique())
print("Total revenue:", round(df["TotalPrice"].sum(), 2))

# ---------------------------------------------------------------
# 3. Basic analysis + plots
# ---------------------------------------------------------------

# 3a. Monthly revenue
monthly = df.groupby("Month")["TotalPrice"].sum()
plt.figure(figsize=(9, 4))
monthly.plot(kind="line", marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.savefig("images/monthly_revenue.png", bbox_inches="tight")
plt.close()

# 3b. Top 10 countries by revenue (UK dominates, so show others too)
country = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4))
country.plot(kind="bar", color="steelblue")
plt.title("Top 10 Countries by Revenue")
plt.ylabel("Revenue")
plt.xticks(rotation=45, ha="right")
plt.savefig("images/top_countries.png", bbox_inches="tight")
plt.close()

# 3c. Top 10 products by quantity sold
products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4))
products.sort_values().plot(kind="barh", color="seagreen")
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Units sold")
plt.savefig("images/top_products.png", bbox_inches="tight")
plt.close()

# 3d. Orders by day of week
df["Weekday"] = df["InvoiceDate"].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = df.groupby("Weekday")["InvoiceNo"].nunique().reindex(order)
plt.figure(figsize=(7, 4))
weekday.plot(kind="bar", color="orange")
plt.title("Number of Orders by Day of Week")
plt.ylabel("Orders")
plt.xticks(rotation=45)
plt.savefig("images/orders_by_weekday.png", bbox_inches="tight")
plt.close()

print("\nTop 5 countries:\n", country.head(5).round(0))
print("\nTop 5 products:\n", products.head(5))

# ---------------------------------------------------------------
# 4. RFM table (Recency, Frequency, Monetary)
# ---------------------------------------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
)
print("\nRFM summary:\n", rfm.describe().round(1))

# RFM distributions
rfm.hist(bins=30, figsize=(10, 3), layout=(1, 3), color="gray")
plt.suptitle("RFM distributions")
plt.savefig("images/rfm_distributions.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# 5. Simple K-Means segmentation
# ---------------------------------------------------------------
# log to reduce skew, then scale so R, F, M count equally
scaled = StandardScaler().fit_transform(rfm.apply(lambda c: (c + 1).apply("log")))

# Elbow plot to pick k
inertia = []
for k in range(1, 11):
    inertia.append(KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled).inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.savefig("images/elbow.png", bbox_inches="tight")
plt.close()

rfm["Cluster"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(scaled)

# ---------------------------------------------------------------
# 6. Understand each segment
# ---------------------------------------------------------------
profile = rfm.groupby("Cluster").agg(
    Customers=("Recency", "size"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum"),
)
profile["Customers_%"] = profile["Customers"] / profile["Customers"].sum() * 100
profile["Revenue_%"] = profile["Total_Revenue"] / profile["Total_Revenue"].sum() * 100
print("\nSegment profile:\n", profile.round(1))

profile[["Customers_%", "Revenue_%"]].plot(kind="bar", figsize=(7, 4))
plt.title("Share of Customers vs Share of Revenue")
plt.xlabel("Cluster")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.savefig("images/customers_vs_revenue.png", bbox_inches="tight")
plt.close()

# Recency vs Monetary scatter coloured by cluster
plt.figure(figsize=(7, 5))
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], cmap="tab10", s=8)
plt.yscale("log")
plt.title("Customers: Recency vs Monetary")
plt.xlabel("Recency (days since last purchase)")
plt.ylabel("Monetary (log scale)")
plt.savefig("images/recency_vs_monetary.png", bbox_inches="tight")
plt.close()

# Save the result table
rfm.to_csv("rfm_segments.csv")
print("\nDone. Plots saved in images/, table saved as rfm_segments.csv")
