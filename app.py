from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load all transformers and model
x_imputer = joblib.load("Tuned_Model/x_imputer.joblib")
outlier_handler = joblib.load("Tuned_Model/outlier_handler.joblib")
encoder_scaler = joblib.load("Tuned_Model/encoder_scaler.joblib")
feature_selector = joblib.load("Tuned_Model/selector.joblib")
feature_extractor = joblib.load("Tuned_Model/extractor.joblib")
model = joblib.load("Tuned_Model/model.joblib")

# Initialize FastAPI app
app = FastAPI(title="Bank Churn Prediction API", version="1.0")

# Define input schema
class BankCustomer(BaseModel):
    # Add fields from your dataset
    Customer_Age: int
    Gender: str
    Dependent_count: int
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: float
    Total_Revolving_Bal: int
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: int
    Total_Trans_Ct: int
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float

@app.get("/")
def read_root():
    return {"message": "Bank Churn Prediction API is live."}

@app.post("/predict/")
def predict_churn(data: BankCustomer):
    try:
        # Step 1: Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Step 2: Preprocessing pipeline
        x = x_imputer.transform(input_df)
        x = outlier_handler.transform(x,"yeo")
        x = encoder_scaler.transform(x)
        x = feature_selector.transform(x)
        x = feature_extractor.transform(x)

        # Step 3: Predict
        prediction = model.predict(x)[0]
        prob = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(prob), 4) if prob is not None else "N/A"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
