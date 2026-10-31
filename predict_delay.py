
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 1. Files load karo
new_data = pd.read_csv(ROOT / "data" / "new_shipments.csv")
hubs = pd.read_csv(ROOT / "data" / "hubs.csv")
hub_segments = pd.read_csv(ROOT / "outputs" / "hub_segments.csv")

# 2. Saved model, features aur settings load karo
model = joblib.load(ROOT / "models" / "delay_model.pkl")

features = pd.read_csv(
    ROOT / "models" / "feature_order.csv"
)["feature"].tolist()

medians = pd.read_csv(
    ROOT / "models" / "impute_values.csv",
    index_col=0
)["median"]

with open(ROOT / "models" / "delay_threshold.txt") as file:
    threshold = float(file.read().strip())

# 3. Service type ko clean karo
new_data["service_type"] = (
    new_data["service_type"].astype(str).str.strip().str.title()
)

# 4. Promised hours banao
service_hours = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96
}

new_data["promised_hours"] = new_data["service_type"].map(
    service_hours
)

# 5. Hub information merge karo
new_data = new_data.merge(
    hubs[["hub_id", "hub_tier", "hub_daily_capacity"]],
    on="hub_id",
    how="left"
)

# 6. Hub segment merge karo
new_data = new_data.merge(
    hub_segments[["hub_id", "segment"]],
    on="hub_id",
    how="left"
)

# 7. Model ke selected features taiyar karo
X = new_data[features].copy()

for column in features:
    X[column] = pd.to_numeric(X[column], errors="coerce")
    X[column] = X[column].fillna(medians[column])

# 8. Delay probability calculate karo
probabilities = model.predict_proba(X)[:, 1]

# 9. Flagged shipment identify karo
flagged = (probabilities >= threshold).astype(int)

# 10. Action decide karo
actions = []

for is_flagged, segment in zip(flagged, new_data["segment"]):

    if is_flagged == 0:
        action = "No action"

    elif segment == "Escalate":
        action = "Escalate to hub manager"

    else:
        action = "Proactive customer call"

    actions.append(action)

# 11. Final output banao
result = pd.DataFrame({
    "shipment_id": new_data["shipment_id"],
    "hub_id": new_data["hub_id"],
    "segment": new_data["segment"],
    "delay_probability": probabilities,
    "flagged": flagged,
    "action": actions
})

# 12. Output save karo
output_path = ROOT / "outputs" / "new_shipment_actions.csv"

result.to_csv(output_path, index=False)

print("Prediction completed successfully!")
print("Total shipments:", len(result))
print("Flagged shipments:", int(result["flagged"].sum()))
print("Not flagged:", int((result["flagged"] == 0).sum()))
print("Output file:", output_path)
print("All predictions saved!")

