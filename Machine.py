import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("house_prices.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Check columns
print("Columns:", data.columns)

# Features and target
X = data[['area', 'bedrooms', 'location']]
y = data['price']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['area', 'bedrooms']),
        ('cat', OneHotEncoder(drop='first'), ['location'])
    ]
)

# Model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train on full dataset
model.fit(X, y)

# Predict
data['predicted_price'] = model.predict(X)

print(data.head())