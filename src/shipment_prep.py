import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
def clean_shipments(raw, medians=None):
    df = raw.drop_duplicates().copy()
    df["service_type"] = df["service_type"].str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    df["declared_value_inr"] = (
        df["declared_value"]
        .str.replace("₹", "",regex = False)
        .str.replace(",", "",regex = False)
        .astype(float)
    )
    df = df.drop(columns="declared_value")
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])
    df["parcel_weight_kg"] = np.where(df["parcel_weight_kg"] >100, np.nan,df["parcel_weight_kg"])
    if medians is None:
        medians = {
            "distance_km" : float(df["distance_km"].median()),
            "parcel_weight_kg" : float(df["parcel_weight_kg"].median()),
            "declared_value_inr" : float(df["declared_value_inr"].median()),
        }
    for col in ["distance_km", "parcel_weight_kg", "declared_value_inr"]:
        df[col] = df[col].fillna(medians[col])
    return df, medians 


def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    return df
       
 
