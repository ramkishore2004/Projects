# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- Step 2: Load CSV ---
df = pd.read_csv(r"D:\Python Files\Mall_Customers.csv")  # adjust path if needed

# --- Step 2a: Clean column names ---
# Remove leading/trailing spaces and convert to consistent format
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)

# Check exact column names
if 'Gender' not in df.columns:
    print("Columns available:", df.columns)
    raise ValueError("The column 'Gender' is missing. Please check the CSV file.")

# --- Step 3: Explore dataset ---
print("Shape of dataset:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isna().sum())

# --- Step 4: Rename columns for simplicity ---
df.rename(columns={'Annual Income (k$)': 'Annual_Income',
                   'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

# --- Step 5: Encode Gender ---
# Ensure there are only 'Male' or 'Female'
print("Unique values in Gender:", df['Gender'].unique())
df['Gender'] = np.where(df['Gender'] == 'Male', 1, 0)

# --- Step 6: Feature Selection ---
features = ['Gender', 'Age', 'Annual_Income', 'Spending_Score']
X = df[features].values  # convert to NumPy array

# --- Step 7: Feature Scaling using NumPy ---
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_scaled = (X - mu) / sigma

# --- Step 8: Elbow Method to find optimal k ---
sse = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    sse.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

# --- Step 9: Apply KMeans with k=5 ---
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Step 10: Evaluate Clustering ---
score = silhouette_score(X_scaled, df['Cluster'])
print("Silhouette Score:", score)

# --- Step 11: Visualize Clusters ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual_Income', y='Spending_Score',
                hue='Cluster', palette='tab10', data=df, s=100)
plt.title("Customer Segments (KMeans)")
plt.show()

# --- Step 12: Cluster Centers ---
centroids = kmeans.cluster_centers_
original_centroids = centroids * sigma + mu
centroid_df = pd.DataFrame(original_centroids, columns=features)
print("\nCluster centers (original scale):\n", centroid_df)

# --- Step 13: Business Recommendations ---
print("""
Business Recommendations:
1. Cluster customers with high Annual Income and high Spending Score:
   → Upsell premium products and loyalty programs.
2. Cluster with high income but low spending:
   → Personal offers to increase engagement.
3. Middle-income but high spending:
   → Reward loyalty with bundles or VIP perks.
4. Young high-spenders:
   → Target with trendier product lines.
5. Older or low spend clusters:
   → Use discount campaigns or seasonal promotions.
""")

# --- Step 14: Save clustered dataset ---
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("Clustered dataset saved as 'Mall_Customers_Clustered.csv'")