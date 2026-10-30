import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.shipment_prep import FEATURES,clean_shipments
import joblib
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if config["max_depth"] == "unlimited" else int(config["max_depth"])
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(config["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42,
                                     n_estimators=int(config["n_estimators"]),
                                     max_depth=max_depth,
                                     min_samples_leaf=int(config["min_samples_leaf"])))
])
final_model.fit(X_train, y_train)
print(len(X_train), len(X_val), len(X_test))
print(round(f1_score(y_val, final_model.predict(X_val)), 4))
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [config["threshold"]]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature":FEATURES}).to_csv("models/feature_order.csv",index = False)
raw = pd.read_csv("data/raw_shipments.csv")
cleaned, medians = clean_shipments(raw)
impute_values = pd.DataFrame({"column": list(medians.keys()), "median": list(medians.values())})
impute_values.to_csv("models/impute_values.csv", index=False)
print(impute_values)
reloaded = joblib.load("models/delay_model.pkl")
proba_memory = final_model.predict_proba(X_test)[:, 1]
proba_reloaded = reloaded.predict_proba(X_test)[:, 1]
check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [len(X_test), np.max(np.abs(proba_memory - proba_reloaded)), config["threshold"]],
}, dtype=object)
check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(check)

