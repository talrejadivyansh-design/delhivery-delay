import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from src.shipment_prep import FEATURES, NOISE_COLUMNS
os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(len(X_train), len(X_val), len(X_test))
print(y_train.sum(), y_val.sum(), y_test.sum())
def cls_row(name,y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}
blind_pred = np.zeros(len(y_val), dtype=int)
blind_row = cls_row("blind_all_on_time", y_val, blind_pred)
print(blind_row)
logistic_model = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
logistic_model.fit(X_train, y_train)
logistic_row = cls_row("logistic_regression", y_val, logistic_model.predict(X_val))
print(logistic_row)
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
forest_row = cls_row("random_forest_default", y_val, forest.predict(X_val))
print(forest_row)
best_d = None
best_f1 = -1
for d in [2, 3, 4, 5, 6, 8, 10]:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    row = cls_row("tmp", y_val, tree.predict(X_val))
    if row["f1"] > best_f1:
        best_f1 = row["f1"]
        best_d = d
print(best_d, best_f1)
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42)
tree_model.fit(X_train, y_train)
tree_row = cls_row(f"decision_tree_depth_{best_d}", y_val, tree_model.predict(X_val))
models_table = pd.DataFrame([blind_row, logistic_row, tree_row, forest_row]).round(4)
models_table.to_csv("outputs/part_c_models.csv", index=False)
print(models_table)
selector = SelectKBest(f_classif, k="all")
selector.fit(X_train, y_train)
scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector.scores_,
    "rf_importance": forest.feature_importances_,
})
scores["f_rank"] = scores["f_score"].rank(ascending=False).astype(int)
scores["is_noise"] = scores["feature"].isin(NOISE_COLUMNS)
scores = scores[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].round(4)
scores.to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores.sort_values("f_rank"))
print(scores[scores["is_noise"]]["rf_importance"].sum())
k_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    val_f1 = f1_score(y_val, pipe.predict(X_val))
    k_rows.append({"k": k, "val_f1": round(val_f1, 4)})
k_table = pd.DataFrame(k_rows)
k_table.to_csv("outputs/part_c_selectk.csv", index=False)
print(k_table)
selector7 = SelectKBest(f_classif,k=7)
X_train_sel = selector7.fit_transform(X_train,y_train)
X_val_sel = selector7.transform(X_val)
param_grid = {
    "n_estimators":[50,100,150],
    "max_depth": [4,6,8],
    "min_samples_leaf":[1,5,10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid, cv = 5)
grid_search.fit(X_train_sel,y_train)
random_search = RandomizedSearchCV(RandomForestClassifier(random_state = 42), param_grid, n_iter = 9, cv= 5, random_state =42)
random_search.fit(X_train_sel,y_train)
for name,search in [("grid", grid_search),("random", random_search)]:
    val_f1 = f1_score(y_val, search.predict(X_val_sel))
    print(name, search.best_params_,round(search.best_score_,4),round(val_f1,4))
default7 = RandomForestClassifier(n_estimators=100,random_state=42)
default7.fit(X_train_sel,y_train)
default_f1 = f1_score(y_val,default7.predict(X_val_sel))
grid_candidates = len(grid_search.cv_results_["params"])
random_candidates = len(random_search.cv_results_["params"])
tuning_rows = [
   {"search":"grid_search","candidates": grid_candidates,"cv_folds":5,
    "total_fits": grid_candidates*5,"best_params": grid_search.best_params_,
    "best_cv_accuracy": round(grid_search.best_score_,4),
    "val_f1": round(f1_score(y_val,grid_search.predict(X_val_sel)),4)},
    {"search": "randomized_search","candidates": random_candidates, "cv_folds": 5,
     "total_fits": random_candidates * 5, "best_params": random_search.best_params_,
     "best_cv_accuracy": round(random_search.best_score_, 4),
     "val_f1": round(f1_score(y_val, random_search.predict(X_val_sel)), 4)},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0,
     "total_fits": 1,
     "best_params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
     "best_cv_accuracy": None, "val_f1": round(default_f1, 4)},]
tuning_table = pd.DataFrame(tuning_rows)
tuning_table.to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_table) 
print(tuning_table.to_string())
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning_table["priority"] = tuning_table["search"].map(priority)
chosen_row = tuning_table.sort_values(["val_f1", "priority"], ascending=[False, True]).iloc[0]
chosen_name = chosen_row["search"]
chosen_params = chosen_row["best_params"]
print(chosen_name, chosen_params)
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params ))
])    
final_model.fit(X_train, y_train)
print(round(f1_score(y_val, final_model.predict(X_val)),4))
proba_val = final_model.predict_proba(X_val)[:,1]
threshold  = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
threshold_rows = []
for t in threshold:
    pred = (proba_val>= t).astype(int)
    row = cls_row("tmp",y_val,pred)
    row["threshold"] = t
    threshold_rows.append(row)
threshold_table = pd.DataFrame(threshold_rows)[["threshold","TN","FP","FN","TP","accuracy","precision","recall","f1"]]
threshold_table.to_csv("outputs/part_c_thresholds.csv",index = False)
print(threshold_table.to_string())
qualified = threshold_table[threshold_table["recall"]>=0.80]
if len(qualified) >0:
    choosen_threshold = qualified["threshold"].max()
else:
    choosen_threshold = 0.1
print(qualified[["threshold", "recall"]])
print("chosen threshold:", choosen_threshold) 
proba_test = final_model.predict_proba(X_test)[:,1]
test_pred = (proba_test>= choosen_threshold).astype(int)
test_row = cls_row("final",y_test,test_pred)
test_row["threshold"] = choosen_threshold
final_test = pd.DataFrame([test_row])[["threshold","TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"]] 
final_test.to_csv("outputs/part_c_final_test.csv",index = False) 
print(final_test.to_string())
max_depth = chosen_params["max_depth"]
final_config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if max_depth is None else max_depth,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": choosen_threshold,
    "chosen_from": chosen_name
}]) 
final_config.to_csv("outputs/part_c_final_config.csv", index=False)
print(final_config.to_string())




   


    
