# ==================================================
# Customer Churn Prediction with Segmented Models
# Dataset: customer_churn.csv
# ==================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_csv(r"D:\Python Files\customer_churn.csv")

# Drop CustomerID (not useful for modeling)
df.drop(columns=["CustomerID"], inplace=True)

# --------------------------------------------------
# 2. Define Features
# --------------------------------------------------
numerical_features = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "SeniorCitizen"
]

target = "Churn"

# --------------------------------------------------
# 3. Preprocessing Pipeline
# --------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# --------------------------------------------------
# 4. Customer Segmentation using Clustering
# --------------------------------------------------
X_cluster = df[numerical_features]

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["segment_id"] = kmeans.fit_predict(X_cluster_scaled)

# --------------------------------------------------
# 5. Map Clusters to Business Segments
# --------------------------------------------------
segment_mapping = {
    0: "Premium Spenders",      # High tenure / high spend
    1: "Budget Conscious",      # Low spend
    2: "Young Professionals"   # Moderate spend / tenure
}

df["segment_name"] = df["segment_id"].map(segment_mapping)

# --------------------------------------------------
# 6. Train Segment-Specific Models
# --------------------------------------------------
segment_models = {}
segment_metrics = {}

for segment in df["segment_name"].unique():

    print(f"\nTraining model for segment: {segment}")

    segment_df = df[df["segment_name"] == segment]

    X = segment_df.drop(columns=["Churn", "segment_id", "segment_name"])
    y = segment_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    segment_models[segment] = model
    segment_metrics[segment] = {
        "Accuracy": round(accuracy, 2),
        "F1 Score": round(f1, 2)
    }

# --------------------------------------------------
# 7. Model Performance Summary
# --------------------------------------------------
print("\n================ MODEL PERFORMANCE ================")
for segment, metrics in segment_metrics.items():
    print(f"\nSegment: {segment}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

# --------------------------------------------------
# 8. Business Recommendations
# --------------------------------------------------
business_recommendations = {
    "Premium Spenders": [
        "Offer long-term contract incentives",
        "Provide priority customer support",
        "Bundle premium services to reduce churn"
    ],
    "Budget Conscious": [
        "Introduce low-cost retention plans",
        "Offer discounts for longer tenure",
        "Promote paperless billing incentives"
    ],
    "Young Professionals": [
        "Flexible month-to-month plans",
        "Digital-first engagement campaigns",
        "Loyalty rewards for early retention"
    ]
}

print("\n================ BUSINESS RECOMMENDATIONS ================")
for segment, recs in business_recommendations.items():
    print(f"\n{segment}:")
    for r in recs:
        print(f"- {r}")

# --------------------------------------------------
# 9. Predict Churn for a New Customer
# --------------------------------------------------
new_customer = pd.DataFrame({
    "Tenure": [12],
    "MonthlyCharges": [85],
    "TotalCharges": [1020],
    "Contract": ["Month-to-month"],
    "PaymentMethod": ["Electronic Check"],
    "PaperlessBilling": ["Yes"],
    "SeniorCitizen": [0]
})

# Assign segment
new_scaled = scaler.transform(new_customer[numerical_features])
new_segment_id = kmeans.predict(new_scaled)[0]
new_segment_name = segment_mapping[new_segment_id]

# Predict churn using segment-specific model
churn_prediction = segment_models[new_segment_name].predict(new_customer)[0]

print("\n================ NEW CUSTOMER PREDICTION ================")
print(f"Assigned Segment: {new_segment_name}")
print(f"Churn Prediction (1 = Yes, 0 = No): {churn_prediction}")