import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# 1. Load the data
print("Loading data...")
df = pd.read_csv('crop_yield_dataset.csv')

# 2. Separate Features and Target
X = df.drop(columns=['Crop_Yield', 'Date'])
y = df['Crop_Yield']

# 3. Setup Preprocessing (Handling text data)
categorical_cols = ['Crop_Type', 'Soil_Type']
numerical_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

# 4. Create the Pipeline with Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train the model
print("Training the model (this might take a few seconds)...")
model.fit(X, y)

# 6. Save the model as a .pkl file
joblib.dump(model, 'crop_model.pkl')
print("Success! 'crop_model.pkl' has been created in your folder.")