import pandas as pd
import numpy as np
import joblib
import shutil

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

NOISE_FEATURES = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

TARGET = "is_delayed"


def get_metrics(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "model": name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }


def main():

    # 1. Read modelling table
    df = pd.read_csv(ROOT / "data" / "modelling_table.csv")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # 2. Stratified train-validation-test split
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

    # 3. Create folders
    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"

    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    # 4. Save feature scores using training data only
    score_selector = SelectKBest(
        score_func=f_classif,
        k=17
    )
    score_selector.fit(X_train, y_train)

    feature_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": score_selector.scores_,
        "p_value": score_selector.pvalues_
    })

    feature_scores["rank"] = (
        feature_scores["f_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    feature_scores["is_noise_feature"] = (
        feature_scores["feature"].isin(NOISE_FEATURES)
    )

    feature_scores = feature_scores.sort_values("rank")

    feature_scores.to_csv(
        outputs_dir / "part_c_feature_scores.csv",
        index=False
    )

    # 5. Compare SelectKBest values
    selectk_rows = []

    for k in [5, 7, 10, 12, 17]:

        selector = SelectKBest(f_classif, k=k)
        selector.fit(X_train, y_train)

        selected = X_train.columns[
            selector.get_support()
        ].tolist()

        temp_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        temp_model.fit(
            selector.transform(X_train),
            y_train
        )

        val_pred = temp_model.predict(
            selector.transform(X_val)
        )

        selectk_rows.append({
            "k": k,
            "validation_F1": f1_score(
                y_val, val_pred, zero_division=0
            ),
            "selected_features": ", ".join(selected)
        })

    pd.DataFrame(selectk_rows).to_csv(
        outputs_dir / "part_c_selectk.csv",
        index=False
    )

    # 6. Final feature selector: k = 7
    selector = SelectKBest(f_classif, k=7)
    selector.fit(X_train, y_train)

    X_train_7 = selector.transform(X_train)
    X_val_7 = selector.transform(X_val)
    X_test_7 = selector.transform(X_test)

    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    # 7. Blind baseline: predict all shipments on time
    baseline_pred = np.zeros(len(y_val), dtype=int)

    model_rows = [
        get_metrics(
            "blind_all_on_time",
            y_val,
            baseline_pred
        )
    ]

    # 8. Logistic Regression
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ])

    logistic.fit(X_train_7, y_train)

    logistic_pred = logistic.predict(X_val_7)

    model_rows.append(
        get_metrics(
            "Logistic Regression",
            y_val,
            logistic_pred
        )
    )

    # 9. Decision Tree: compare different depths
    tree_rows = []
    tree_models = []

    for depth in [2, 3, 4, 5, 6, 8, 10]:

        tree = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            random_state=42
        )

        tree.fit(X_train_7, y_train)

        train_pred = tree.predict(X_train_7)
        val_pred = tree.predict(X_val_7)

        val_f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        tree_rows.append({
            "max_depth": depth,
            "train_F1": f1_score(
                y_train, train_pred, zero_division=0
            ),
            "validation_F1": val_f1
        })

        tree_models.append((depth, tree, val_f1))

    pd.DataFrame(tree_rows).to_csv(
        outputs_dir / "part_c_tree_depth.csv",
        index=False
    )

    # Highest F1; smaller depth wins ties
    best_depth, best_tree, _ = sorted(
        tree_models,
        key=lambda item: (-item[2], item[0])
    )[0]

    tree_pred = best_tree.predict(X_val_7)

    model_rows.append(
        get_metrics(
            "Decision Tree",
            y_val,
            tree_pred
        )
    )

    # 10. Default Random Forest
    default_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    default_forest.fit(X_train_7, y_train)

    default_pred = default_forest.predict(X_val_7)

    model_rows.append(
        get_metrics(
            "Random Forest Default",
            y_val,
            default_pred
        )
    )

    pd.DataFrame(model_rows).to_csv(
        outputs_dir / "part_c_models.csv",
        index=False
    )

    # 11. Random Forest Grid Search
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(X_train_7, y_train)

    # 12. Random Forest Randomized Search
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=9,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_7, y_train)

    # 13. Compare all three Random Forest versions
    tuning_candidates = [
        ("default_forest_same_7_features", default_forest),
        ("grid_search", grid_search.best_estimator_),
        ("randomized_search", random_search.best_estimator_)
    ]

    tuning_rows = []
    tuning_scores = []

    for name, model in tuning_candidates:

        val_pred = model.predict(X_val_7)

        val_f1 = f1_score(
            y_val, val_pred, zero_division=0
        )

        tuning_scores.append((name, model, val_f1))

        tuning_rows.append({
            "model": name,
            "validation_F1": val_f1,
            "best_parameters": str(
                model.get_params()
            )
        })

    pd.DataFrame(tuning_rows).to_csv(
        outputs_dir / "part_c_tuning.csv",
        index=False
    )

    # Highest F1; listed order breaks ties
    winner_name, winner, winner_f1 = max(
        tuning_scores,
        key=lambda item: item[2]
    )

    # 14. Select alert threshold using validation recall
    val_probabilities = winner.predict_proba(X_val_7)[:, 1]

    threshold_rows = []

    for threshold in np.arange(0.1, 1.0, 0.1):

        threshold = round(float(threshold), 1)

        val_pred = (
            val_probabilities >= threshold
        ).astype(int)

        threshold_rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_val, val_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, val_pred, zero_division=0
            ),
            "F1": f1_score(
                y_val, val_pred, zero_division=0
            )
        })

    threshold_df = pd.DataFrame(threshold_rows)

    eligible = threshold_df[
        threshold_df["recall"] >= 0.80
    ]

    if len(eligible) > 0:
        best_threshold = float(
            eligible["threshold"].max()
        )
    else:
        best_threshold = 0.1

    threshold_df["selected_threshold"] = (
        threshold_df["threshold"] == best_threshold
    )

    threshold_df.to_csv(
        outputs_dir / "part_c_thresholds.csv",
        index=False
    )

    # 15. Final test evaluation only once
    test_probabilities = winner.predict_proba(X_test_7)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    final_metrics = get_metrics(
        winner_name,
        y_test,
        test_predictions
    )

    final_metrics["threshold"] = best_threshold

    pd.DataFrame([final_metrics]).to_csv(
        outputs_dir / "part_c_final_test.csv",
        index=False
    )

    # 16. Preserve Part B regression model
    old_model = models_dir / "delay_model.pkl"
    regression_copy = models_dir / "delivery_hours_model.pkl"

    if old_model.exists() and not regression_copy.exists():
        shutil.copy2(old_model, regression_copy)

    # 17. Save final classification model and threshold
    joblib.dump(winner, models_dir / "delay_model.pkl")

    with open(models_dir / "delay_threshold.txt", "w") as file:
        file.write(str(best_threshold))

    pd.DataFrame({
        "feature": selected_features
    }).to_csv(
        models_dir / "feature_order.csv",
        index=False
    )

    # 18. Save final configuration
    config = [
        ["total_rows", len(df)],
        ["feature_count", len(FEATURES)],
        ["selected_feature_count", 7],
        ["selected_features", ", ".join(selected_features)],
        ["train_rows", len(X_train)],
        ["validation_rows", len(X_val)],
        ["test_rows", len(X_test)],
        ["train_delayed", int(y_train.sum())],
        ["validation_delayed", int(y_val.sum())],
        ["test_delayed", int(y_test.sum())],
        ["selected_tree_depth", best_depth],
        ["selected_model", winner_name],
        ["validation_F1", winner_f1],
        ["alert_threshold", best_threshold]
    ]

    pd.DataFrame(
        config,
        columns=["setting", "value"]
    ).to_csv(
        outputs_dir / "part_c_final_config.csv",
        index=False
    )

    print("Part C completed successfully!")
    print("Rows:", len(df))
    print("Train / Validation / Test:",
          len(X_train), len(X_val), len(X_test))
    print("Delayed counts:",
          int(y_train.sum()),
          int(y_val.sum()),
          int(y_test.sum()))
    print("Selected features:", len(selected_features))
    print("Selected model:", winner_name)
    print("Alert threshold:", best_threshold)
    print("All Part C files saved.")


if __name__ == "__main__":
    main()
