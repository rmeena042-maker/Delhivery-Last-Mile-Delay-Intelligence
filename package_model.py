
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

data = pd.read_csv(ROOT / "data" / "modelling_table.csv")

models_dir = ROOT / "models"
outputs_dir = ROOT / "outputs"

# 1. Saved model load karo
model = joblib.load(models_dir / "delay_model.pkl")

# 2. Saved selected features load karo
feature_order = pd.read_csv(
    models_dir / "feature_order.csv"
)["feature"].tolist()

# 3. Median values load karo
medians = pd.read_csv(
    models_dir / "impute_values.csv",
    index_col=0
)["median"]

# 4. Alert threshold load karo
with open(models_dir / "delay_threshold.txt") as file:
    threshold = float(file.read().strip())

# 5. Data prepare karo
X = data[feature_order].copy()
y = data["is_delayed"].astype(int)

for column in feature_order:
    X[column] = pd.to_numeric(X[column], errors="coerce")
    X[column] = X[column].fillna(medians[column])

# 6. Same 840 / 180 / 180 split recreate karo
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 7. Reloaded model se 180 predictions
probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)

# 8. Test probabilities save karo
result = pd.DataFrame({
    "actual": y_test.values,
    "delay_probability": probabilities,
    "predicted_delayed": predictions
})

result.to_csv(
    outputs_dir / "package_test_probabilities.csv",
    index=False
)

# 9. Threshold ki CSV save karo
pd.DataFrame({
    "alert_threshold": [threshold]
}).to_csv(
    models_dir / "delay_threshold.csv",
    index=False
)

# 10. Final checks
assert len(probabilities) == 180
assert len(feature_order) == 7

print("Model packaging completed successfully!")
print("Train / Validation / Test:", len(X_train), len(X_val), len(X_test))
print("Selected features:", feature_order)
print("Alert threshold:", threshold)
print("Test probabilities:", len(probabilities))
print("All package files saved!")

