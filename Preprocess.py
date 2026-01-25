# ==================================================
# 1. IMPORT LIBRARIES
# ==================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ==================================================
# 2. LOAD DATA
# ==================================================
df = pd.read_csv("customer_churn.csv")
print("Initial shape:", df.shape)

# ==================================================
# 3. BASIC CLEANING
# ==================================================
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

df['Churn'] = (
    df['Churn']
    .astype(str)
    .str.strip()
    .str.lower()
    .map({'yes': 1, 'no': 0})
)

df = df.dropna(subset=['Churn'])
print("After cleaning Churn:", df.shape)

# ==================================================
# 4. HANDLE MISSING VALUES (SAFE)
# ==================================================
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        if df[col].notna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    else:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

# ==================================================
# 5. OUTLIER HANDLING (CAPPING)
# ==================================================
if 'MonthlyCharges' in df.columns:
    Q1 = df['MonthlyCharges'].quantile(0.25)
    Q3 = df['MonthlyCharges'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['MonthlyCharges'] = df['MonthlyCharges'].clip(
        lower=lower_bound,
        upper=upper_bound
    )

print("After preprocessing:", df.shape)

# ==================================================
# 6. FEATURE / TARGET SPLIT
# ==================================================
X = df.drop('Churn', axis=1)
y = df['Churn']

# ==================================================
# 7. COLUMN TYPES
# ==================================================
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# ==================================================
# 8. PREPROCESSING PIPELINE
# ==================================================
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ==================================================
# 9. MODEL PIPELINE
# ==================================================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    ))
])

# ==================================================
# 10. TRAIN-TEST SPLIT
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==================================================
# 11. TRAIN MODEL
# ==================================================
model.fit(X_train, y_train)

# ==================================================
# 12. EVALUATION
# ==================================================
y_pred = model.predict(X_test)

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\n✅ SUCCESS: Model trained without errors")