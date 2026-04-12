import pandas as pd
import joblib


gmm = joblib.load("../models/GMM_RiskProfile.pkl")
pt = joblib.load("../models/PowerTransformer.pkl")
scaler = joblib.load("../models/RobustScaler.pkl")
caps = joblib.load("../models/Caps.pkl")


def preprocess(data):
    df = pd.DataFrame([data.dict()])
    features = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]
    df = df[features]

    df["Recency"] = df["Recency"].clip(upper=caps["Recency"])
    df["Monetary"] = df["Monetary"].clip(upper=caps["Monetary"])
    df["Frequency"] = df["Frequency"].clip(upper=caps["Frequency"])
    df["AvgOrderValue"] = df["AvgOrderValue"].clip(upper=caps["AvgOrderValue"])

    X_pt = pt.transform(df)
    X_scaled = scaler.transform(X_pt)
    return X_scaled

def get_risk(data, cluster_stats):

    X_scaled = preprocess(data)

    # 1. Predict cluster
    cluster = gmm.predict(X_scaled)[0]

    # 2. Log-likelihood
    risk_ll = gmm.score_samples(X_scaled)[0]

    # 3. Fetch cluster stats
    mean = cluster_stats.loc[cluster, "mean"]
    std = cluster_stats.loc[cluster, "std"]

    std = std if std != 0 else 1

    # 5. Normalize (Z-score)
    ll_norm = (risk_ll - mean) / std

    if ll_norm < -1.5:
        risk = "High Risk"
    elif ll_norm < -0.5:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return risk
