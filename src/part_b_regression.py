import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

TARGET = "delivery_hours"


def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def metrics_row(name, split, actual, predicted):
    return {
        "model": name,
        "split": split,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2": r2_score(actual, predicted)
    }


def main():
    df = pd.read_csv(ROOT / "data" / "modelling_table.csv")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Split: 840 train, 180 validation, 180 test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Save feature order and training medians
    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    pd.DataFrame({"feature": FEATURES}).to_csv(
        models_dir / "feature_order.csv", index=False
    )

    medians = X_train.median()
    medians.rename("median").to_csv(
        models_dir / "impute_values.csv"
    )

    # Mean baseline
    baseline_prediction = np.repeat(y_train.mean(), len(y_val))
    rows = [
        metrics_row("Mean Baseline", "validation",
                    y_val, baseline_prediction)
    ]

    candidates = {}

    # Linear Regression with StandardScaler
    linear = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    linear.fit(X_train, y_train)
    candidates["Linear Regression"] = linear

    # Decision Tree: select depth using validation RMSE
    depth_rows = []
    depth_models = []

    for depth in [1, 2, 3, 4, 6, 8, 12, None]:
        tree = DecisionTreeRegressor(
            max_depth=depth, random_state=42
        )
        tree.fit(X_train, y_train)

        train_pred = tree.predict(X_train)
        val_pred = tree.predict(X_val)

        depth_rows.append({
            "max_depth": "unlimited" if depth is None else depth,
            "train_RMSE": rmse(y_train, train_pred),
            "validation_RMSE": rmse(y_val, val_pred)
        })

        depth_models.append((depth, tree, rmse(y_val, val_pred)))

    pd.DataFrame(depth_rows).to_csv(
        outputs_dir / "part_b_tree_depth.csv", index=False
    )

    best_depth, best_tree, _ = min(
        depth_models, key=lambda item: (item[2], float("inf") if item[0] is None else item[0])
    )
    candidates["Decision Tree"] = best_tree

    # Random Forest
    forest = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    forest.fit(X_train, y_train)
    candidates["Random Forest"] = forest

    # Gradient Boosting: compare learning rates
    gb_rows = []
    gb_models = []

    for rate in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=rate,
            random_state=42
        )
        gb.fit(X_train, y_train)

        train_pred = gb.predict(X_train)
        val_pred = gb.predict(X_val)

        gb_rows.append({
            "learning_rate": rate,
            "train_RMSE": rmse(y_train, train_pred),
            "validation_RMSE": rmse(y_val, val_pred)
        })

        gb_models.append((rate, gb, rmse(y_val, val_pred)))

    pd.DataFrame(gb_rows).to_csv(
        outputs_dir / "part_b_gb_learning_rate.csv", index=False
    )

    best_rate, best_gb, _ = min(gb_models, key=lambda item: item[2])
    candidates["Gradient Boosting"] = best_gb

    # Validation metrics for all four models
    for name, model in candidates.items():
        prediction = model.predict(X_val)
        rows.append(metrics_row(name, "validation", y_val, prediction))

    # Select winner using validation RMSE
    validation_scores = {
        name: rmse(y_val, model.predict(X_val))
        for name, model in candidates.items()
    }

    winner_name = min(validation_scores, key=validation_scores.get)
    winner = candidates[winner_name]

    # Cross-validation of top two on train + validation data
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    top_two = sorted(
        validation_scores, key=validation_scores.get
    )[:2]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    for name in top_two:
        scores = cross_val_score(
            candidates[name], X_trainval, y_trainval,
            cv=cv, scoring="r2"
        )
        cv_rows.append({
            "model": name,
            "mean_CV_R2": scores.mean(),
            "std_CV_R2": scores.std()
        })

    pd.DataFrame(cv_rows).to_csv(
        outputs_dir / "part_b_cv.csv", index=False
    )

    # Final test evaluation: winner trained on training rows only
    winner.fit(X_train, y_train)
    test_prediction = winner.predict(X_test)

    rows.append(
        metrics_row(winner_name, "test", y_test, test_prediction)
    )

    pd.DataFrame(rows).to_csv(
        outputs_dir / "part_b_metrics.csv", index=False
    )

    # Save final model
    joblib.dump(winner, models_dir / "delay_model.pkl")

    # Checks
    scaled_train = linear.named_steps["scaler"].transform(X_train)

    checks = [
        ["Total rows", len(df)],
        ["Feature count", len(FEATURES)],
        ["Training rows", len(X_train)],
        ["Validation rows", len(X_val)],
        ["Test rows", len(X_test)],
        ["Scaled training mean absolute value",
         float(np.abs(scaled_train.mean(axis=0)).mean())],
        ["Selected model", winner_name],
        ["Selected tree depth",
         "not selected" if best_depth is None else best_depth],
        ["Selected GB learning rate", best_rate]
    ]

    pd.DataFrame(checks, columns=["metric", "value"]).to_csv(
        outputs_dir / "part_b_checks.csv", index=False
    )

    print("Part B completed successfully!")
    print("Rows:", len(df))
    print("Train / Validation / Test:",
          len(X_train), len(X_val), len(X_test))
    print("Selected model:", winner_name)
    print("Test RMSE:", round(rmse(y_test, test_prediction), 3))
    print("All Part B files saved.")


if __name__ == "__main__":
    main()
