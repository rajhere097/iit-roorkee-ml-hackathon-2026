from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# 1. Initialize the FastAPI app
app = FastAPI(title='Coupon Acceptance Prediction API', version='1.0')

# 2. Load models at startup
encoder = joblib.load('encoder.joblib')
model = joblib.load('xgb_model.joblib')

# 3. Define the Input Data Schema using Pydantic
# This ensures the API only accepts the correct data types
class CustomerData(BaseModel):
    destination: str
    passanger: str
    weather: str
    temperature: int
    time: str
    coupon: str
    expiration: str
    gender: str
    age: str
    maritalStatus: str
    has_children: int
    education: str
    occupation: str
    income: str
    Bar: str
    CoffeeHouse: str
    CarryAway: str
    RestaurantLessThan20: str
    Restaurant20To50: str
    toCoupon_GEQ5min: int
    toCoupon_GEQ15min: int
    toCoupon_GEQ25min: int
    direction_same: int
    direction_opp: int

# 4. Define the Prediction Endpoint
@app.post("/predict")
# Pydantic automatically filters out excluded columns like 'customer_id' and 'car', 
# and the following steps replicate the DataCleaner_and_Checker logic.
def predict_coupon(data: CustomerData):
    try:
        # Convert incoming JSON payload to a pandas DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Preprocessing
        input_data = input_data.replace('nan', np.nan)
        cat_cols = input_data.select_dtypes(include=['object', 'category']).columns
        input_data[cat_cols] = input_data[cat_cols].fillna('Unknown')

        # Encode and Predict
        X_encoded = encoder.transform(input_data).astype(float)
        prediction = model.predict(X_encoded)

        # Return the result as JSON
        return {'prediction': int(prediction[0]),
            'message': 'Accepted' if prediction[0] == 1 else 'Rejected'}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 5. Root endpoint for health check
@app.get('/')
def read_root():
    return {'status': 'API is running. Visit /docs for documentation.'}
