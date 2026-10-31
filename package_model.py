import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

data = pd.read_csv(ROOT / "data" / "modelling_table.csv")
models_dir = ROOT / "models"
outputs_dir = ROOT / "outputs"

models_dir.mkdir(exist_ok=True)
outputs_dir.mkdir(exist_ok=True)

# 1. Existing trained model load karo
model = joblib.load(models_dir / "delay_model.pkl")

# 2. Feature order load karo
feature_df = pd.read_csv(models_dir / "feature_order.csv")
feature_order = feature_df["feature"].tolist()

# 3. Median values load karo
medians = pd.read_csv(
    models_dir / "impute_values.csv",
    index_col=0
)["median"]

# 4. Threshold load karo
with open(models_dir / "delay_threshold.txt") as file:
    threshold = float(file.read().strip())

# 5. Required package files save karo
joblib.dump(model, models_dir / "delay_model.pkl")

feature_df.to_csv(
    models_dir / "feature_order.csv",
    index=False
)

medians.to_frame(name="median").to_csv(
    models_dir / "impute_values.csv"
)

pd.DataFrame({
    "alert_threshold": [threshold]
}).to_csv(
    models_dir / "delay_threshold.csv",
    index=False
)

# 6. Saved package files dobara load karke check karo
model = joblib.load(models_dir / "delay_model.pkl")

feature_order = pd.read_csv(
    models_dir / "feature_order.csv"
)["feature"].tolist()

medians = pd.read_csv(
    models_dir / "impute_values.csv",
    index_col=0
)["median"]

threshold = pd.read_csv(
    models_dir / "delay_threshold.csv"
)["alert_threshold"].iloc[0]

# 7. Data prepare karo
X = data[feature_order].copy()
y = data["is_delayed"].astype(int)

for column in feature_order:
    X[column] = pd.to_numeric(X[column], errors="coerce")
    X[column] = X[column].fillna(medians[column])

# 8. Same 840 / 180 / 180 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 9. Reloaded model se test predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# 10. Test probabilities save karo
result = pd.DataFrame({
    "actual": y_test.values,
    "delay_probability": probabilities,
    "predicted_delayed": predictions
})

result.to_csv(
    outputs_dir / "package_test_probabilities.csv",
    index=False
)

# 11. Final checks
assert len(probabilities) == 180
assert len(feature_order) == 7
assert len(medians) >= 7

print("Model packaging completed successfully!")
print("Train / Validation / Test:", len(X_train), len(X_val), len(X_test))
print("Selected features:", feature_order)
print("Alert threshold:", threshold)
print("Test probabilities:", len(probabilities))
print("All package files saved and reloaded successfully!")
