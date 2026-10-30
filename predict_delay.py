import numpy as np
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features
new_raw = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")
model = joblib.load("models/delay_model.pkl")
threshold = pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0]
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = dict(zip(impute["column"], impute["median"]))
cleaned, _ = clean_shipments(new_raw, medians)
featured = build_features(cleaned, hubs)
X_new = featured[feature_order]
featured["delay_probability"] = np.round(model.predict_proba(X_new)[:, 1], 4)
print(len(featured))
print(featured[["shipment_id", "delay_probability"]].head())
featured["flagged"] = (featured["delay_probability"]>= threshold).astype(int)
featured = featured.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")
featured["action"] = np.where(
    (featured["flagged"] == 1) & (featured["segment"] == "Escalate"), "Escalate to hub manager",
    np.where(featured["flagged"] == 1, "Proactive customer call", "No action"))
actions = featured[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
actions.to_csv("outputs/new_shipment_actions.csv", index=False)
print(len(actions), actions["shipment_id"].is_unique, actions.isnull().sum().sum())
print(actions["flagged"].sum(), (actions["action"] == "Escalate to hub manager").sum())
print(actions.head(5).to_string())