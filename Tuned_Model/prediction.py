import requests

url = "http://127.0.0.1:8000/predict/"
payload = {
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 2,
    "Education_Level": "Graduate",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 36,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 2,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 8000.0,
    "Total_Revolving_Bal": 1200,
    "Avg_Open_To_Buy": 6800.0,
    "Total_Amt_Chng_Q4_Q1": 1.2,
    "Total_Trans_Amt": 5000,
    "Total_Trans_Ct": 80,
    "Total_Ct_Chng_Q4_Q1": 0.7,
    "Avg_Utilization_Ratio": 0.15
}

response = requests.post(url, json=payload)
print(response.json())
