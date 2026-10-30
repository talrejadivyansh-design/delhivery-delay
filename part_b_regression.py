import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["delivery_hours"]
X_train , X_temp, y_train , y_temp = train_test_split(X , y , test_size = 0.3, random_state = 42)
X_val , X_test , y_val , y_test = train_test_split(X_temp,y_temp,test_size = 0.5,random_state = 42)
print(len(X_train), len(X_val), len(X_test))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
train_scaled_max_abs_mean = abs(X_train_scaled.mean(axis=0)).max()
test_scaled_max_abs_mean = abs(X_test_scaled.mean(axis=0)).max()
checks = {"train_rows": len(X_train), "val_rows": len(X_val), "test_rows": len(X_test),"train_scaled_max_abs_mean": round(train_scaled_max_abs_mean, 4), "test_scaled_max_abs_mean": round(test_scaled_max_abs_mean, 4)}
pd.DataFrame({"metric": list(checks.keys()), "value": list(checks.values())}).to_csv("outputs/part_b_checks.csv",index = False)
print(checks)
def score(name,split,y_true,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true,y_pred)
    return {"model": name, "split": split, "MAE": round(mae,4), "MSE": round(mse,4),"RMSE": round(rmse,4), "R2": round(r2,4) }
baseline_pred = np.full(len(y_val), y_train.mean())
baseline_row = score("mean_baseline","val",y_val,baseline_pred)
print(baseline_row)
depth_rows = []
for d in [1,2,3,4,6,8,12,None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train,y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, tree.predict(X_train)))
    val_rmse = np.sqrt(mean_squared_error(y_val, tree.predict(X_val)))
    depth_rows.append({"max_depth":"unlimited" if d is None else d, "train_rmse": round(train_rmse,4), "val_rmse": round(val_rmse,4)})
depth_table = pd.DataFrame(depth_rows)
depth_table.to_csv("outputs/part_b_tree_depth.csv",index = False)
print(depth_table)
linear_model = Pipeline([("scaler", StandardScaler()),("model", LinearRegression())])
linear_model.fit(X_train,y_train)
linear_pred = linear_model.predict(X_val)
linear_row = score("linear_regression", "val", y_val, linear_pred)
best_rmse = depth_table["val_rmse"].min()
best_row = depth_table[depth_table["val_rmse"] == best_rmse].iloc[0]
best_depth = None if best_row["max_depth"] == "unlimited" else int(best_row["max_depth"])
tree_model = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
tree_model.fit(X_train,y_train)
tree_row = score("decision_tree", "val", y_val, tree_model.predict(X_val))
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train,y_train)
rf_row = score("random_forest", "val", y_val, rf_model.predict(X_val))
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train,y_train)
gb_row = score("gradient_boosting", "val" , y_val, gb_model.predict(X_val))
metrics =  pd.DataFrame([baseline_row, linear_row, tree_row, rf_row, gb_row]).round(4)
metrics.to_csv("outputs/part_b_metrics.csv",index = False)
print(metrics)
lr_rows = []
for lr in [0.01,0.1,1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train,y_train)
    pred = gb.predict(X_val)
    row = score("gb","val",y_val,pred)
    lr_rows.append({"learning_rate": lr, "val_rmse": row["RMSE"],  "val_r2": row["R2"]})
lr_table = pd.DataFrame(lr_rows)
lr_table.to_csv("outputs/part_b_gb_learning_rate.csv",index = False)    
print(lr_table)
ranking = metrics[metrics["model"]!= "mean_baseline"].sort_values("RMSE")
print(ranking)
models = {
    "linear_regression": linear_model,
    "decision_tree": tree_model,
    "random_forest": rf_model,
    "gradient_boosting": gb_model,
}
top_two = ranking["model"].head(2).tolist()
best_name = ranking["model"].iloc[0]
print(top_two, best_name)
X_trval = pd.concat([X_train, X_val])
y_trval = pd.concat([y_train, y_val])
print(len(X_trval))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for model_name in top_two:
    scores = cross_val_score(models[model_name], X_trval, y_trval, cv=kf)
    cv_rows.append({"model": model_name, "mean_r2": round(scores.mean(),4), "std_r2": round(scores.std(),4)})
cv_table = pd.DataFrame(cv_rows)
cv_table.to_csv("outputs/part_b_cv.csv",index = False)
print(cv_table)
final_model = models[best_name]
final_pred = final_model.predict(X_test)
final_row = score(best_name, "test", y_test, final_pred)
metrics = pd.concat([metrics,pd.DataFrame([final_row]).round(4)], ignore_index = True)
metrics.to_csv("outputs/part_b_metrics.csv",index = False) 
print(metrics)   


                               




