import os
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
os.makedirs("outputs",exist_ok = True)
df = pd.read_csv("data/modelling_table.csv")
profile = df.groupby("hub_id").agg({
    "shipment_id": "count",
    "delivery_hours":"mean",
    "is_delayed":"mean",
    "distance_km":"mean",
    "n_stops": "mean"
}).reset_index()
profile.columns = ["hub_id", "shipments", "mean_delivery_hours", "delay_rate","mean_distance_km", "mean_n_stops"]
profile = profile.sort_values("hub_id").round(4)
profile.to_csv("outputs/hub_profile.csv", index=False)
print(profile.head(3))
print(len(profile), profile["shipments"].unique())
cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[cluster_cols])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = kmeans.fit_predict(scaled)
print(profile["cluster"].value_counts().sort_values(ascending=False))
dist = pairwise_distances(scaled)
sep_rows = []
for K in [2,3,4,5]:
    labels = KMeans(n_clusters=K,random_state = 42,n_init = 10).fit_predict(scaled)
    across = []
    within = []
    for i in range(24):
        for j in range(i+1,24):
            if labels[i] == labels[j]:
                within.append(dist[i][j])
            else:
                across.append(dist[i][j])
    print(K, len(across) + len(within)) 
    ratio = np.mean(across) / np.mean(within)
    sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
    sep_rows.append({"K":K,"across_over_within": round(ratio, 4),"cluster_sizes": "-".join(str(s) for s in sizes)})
sep_table = pd.DataFrame(sep_rows)
sep_table.to_csv("outputs/part_d_separation.csv", index=False)
print(sep_table.to_string())
cluster_rates = profile.groupby("cluster")["delay_rate"].mean().reset_index()
cluster_rates = cluster_rates.sort_values("delay_rate").reset_index(drop=True)
cluster_rates["segment"] = ["Standard", "Watch", "Escalate"]
print(cluster_rates)
profile = profile.merge(cluster_rates[["cluster", "segment"]], on="cluster", how="left")
print(profile.groupby("segment")["hub_id"].count())
hubs = pd.read_csv("data/hubs.csv")
hub_segments = profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)
print(hub_segments.head(3).to_string())
segment_profile = profile.groupby("segment").agg({
    "hub_id": "count",
    "mean_delivery_hours": "mean",
    "delay_rate": "mean",
    "mean_distance_km": "mean",
    "mean_n_stops": "mean",
}).reset_index()
segment_profile.columns = ["segment", "hubs", "mean_delivery_hours", "delay_rate","mean_distance_km", "mean_n_stops"]
segment_profile = segment_profile.sort_values("delay_rate").round(4)
segment_profile.to_csv("outputs/segment_profile.csv", index=False)
print(segment_profile.to_string())
tier_to_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
implied = hub_segments["hub_tier"].map(tier_to_segment)
mismatch = hub_segments[hub_segments["segment"] != implied]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(mismatch.to_string())
                           
                    

                   
