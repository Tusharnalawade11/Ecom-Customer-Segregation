
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from utils.helper import preprocess, get_risk
from utils.config import load_api_key

app = FastAPI(title="Customer Segmentation API")

kmeans = joblib.load("../models/KmeansCluster.pkl")
gb_classifier = joblib.load("../models/GB_Classifier.pkl")
cluster_stats = pd.read_csv("../models/cluster_stats.csv", index_col=0)


groq_client = load_api_key()


class CustomerInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float


@app.post("/predict")
def predict_segment(data: CustomerInput):
    '''
        API : PREDICTION
        Input: Customer features (Recency, Frequency, Monetary, AvgOrderValue)
        Output: Customer Segment and Risk Profile
    '''
    X = [[data.Recency, data.Frequency, data.Monetary, data.AvgOrderValue]]
    
    segment = gb_classifier.predict(X)[0].split('---')[0].strip()
    risk = get_risk(data, cluster_stats)
    print(f"Segment: {segment} \nRisk: {risk}")

    return {
        "Segment": segment,
        "RiskProfile": risk
    }


@app.post("/recommend")
def recommend(data: CustomerInput):
    '''
        API : RECOMMENDATION
        Input: Customer features (Recency, Frequency, Monetary, AvgOrderValue)
        Output: Personalized marketing recommendations based on segment and risk profile
    '''
    X = [[data.Recency, data.Frequency, data.Monetary, data.AvgOrderValue]]

    segment = gb_classifier.predict(X)[0].split('---')[0].strip()
    risk = get_risk(data, cluster_stats)

    prompt = f"""
    Customer Segment: {segment}
    Risk Level: {risk}

    Suggest personalized marketing recommendations:
    - Upsell ideas
    - Cross-sell products
    - Retention strategy

    Keep it concise and actionable. Summarize reccomendations at start of the response in 3-5 bullet points and end with a seperator.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    recommendation = response.choices[0].message.content

    print(f"Using GROQ API: \n\nRecommendation: {recommendation}")

    recommendation_insights = recommendation.split("\n\n")[1].split("\n")[:]

    return {
        "Segment": segment,
        "RiskProfile": risk,
        "Recommendation": recommendation_insights,
        "Detailed Recommendation": recommendation
    }
