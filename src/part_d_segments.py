import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

data = pd.read_csv(ROOT / "data" / "modelling_table.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")

# 1. Har hub ka performance profile
profile = data.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

profile = profile.merge(hubs, on="hub_id", how="left")

# 2. Clustering ke liye features
features = [
    "shipments",
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = profile[features].fillna(profile[features].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K = 2, 3, 4, 5 ka comparison
separation_results = []

for k in [2, 3, 4, 5]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    within = model.inertia_
    overall_mean = np.mean(X_scaled, axis=0)
    between = 0

    for cluster in range(k):
        cluster_points = X_scaled[labels == cluster]
        if len(cluster_points) > 0:
            between += len(cluster_points) * np.sum(
                (cluster_points.mean(axis=0) - overall_mean) ** 2
            )

    separation_results.append({
        "k": k,
        "within_cluster_distance": within,
        "between_cluster_distance": between,
        "separation_ratio": between / within if within != 0 else 0,
        "cluster_sizes": str(pd.Series(labels).value_counts().sort_index().to_dict())
    })

pd.DataFrame(separation_results).to_csv(
    ROOT / "outputs" / "part_d_separation.csv", index=False
)

# 4. Final KMeans model: 3 clusters
final_model = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = final_model.fit_predict(X_scaled)

# 5. Clusters ko delay rate ke hisaab se label dena
cluster_order = (
    profile.groupby("cluster")["delay_rate"]
    .mean()
    .sort_values()
    .index.tolist()
)

labels = {
    cluster_order[0]: "Standard",
    cluster_order[1]: "Watch",
    cluster_order[2]: "Escalate"
}

profile["segment"] = profile["cluster"].map(labels)

# 6. Hub profile save
profile.to_csv(ROOT / "outputs" / "hub_profile.csv", index=False)

# 7. Hub segments save
profile[["hub_id", "cluster", "segment"]].to_csv(
    ROOT / "outputs" / "hub_segments.csv", index=False
)

# 8. Segment profile
segment_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    shipments=("shipments", "sum"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

segment_profile.to_csv(
    ROOT / "outputs" / "segment_profile.csv", index=False
)

# 9. Hub tier aur segment ka comparison
tier_column = "hub_tier"

if tier_column not in profile.columns:
    raise ValueError("hub_tier column hubs.csv mein nahi mila.")

tier_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}

profile["expected_segment"] = profile[tier_column].map(tier_segment)
mismatch = profile[
    profile["expected_segment"].notna()
    & (profile["segment"] != profile["expected_segment"])
].copy()

mismatch[[
    "hub_id", tier_column, "expected_segment", "segment", "delay_rate"
]].to_csv(
    ROOT / "outputs" / "tier_vs_segment_mismatch.csv", index=False
)

print("Part D completed successfully!")
print("Total hubs:", len(profile))
print("Segments:", profile["segment"].value_counts().to_dict())
print("Mismatch hubs:", mismatch["hub_id"].tolist())
print("All Part D files saved.")
