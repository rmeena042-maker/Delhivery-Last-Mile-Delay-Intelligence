import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def prepare_data():

    # 1. Read CSV files
    df = pd.read_csv(ROOT / "data" / "raw_shipments.csv")
    hubs = pd.read_csv(ROOT / "data" / "hubs.csv")

    # 2. Remove duplicates
    original_rows = len(df)
    df = df.drop_duplicates()

    # 3. Clean service type
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 4. Handle missing payment mode
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 5. Convert declared value
    df["declared_value_inr"] = pd.to_numeric(
        df["declared_value"].astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

    df = df.drop(columns=["declared_value"])

    # 6. Convert dispatch time
    df["dispatch_time"] = pd.to_datetime(
        df["dispatch_time"], errors="coerce"
    )

    # 7. Convert numeric columns
    numeric_columns = [
        "parcel_weight_kg", "distance_km",
        "declared_value_inr", "n_stops",
        "rain_flag", "sale_event_flag",
        "delivery_hours", "label_print_batch",
        "scanner_build_no", "tape_roll_no",
        "sticker_serial_mod", "shelf_slot_no"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 8. Invalid weight handling
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 9. Merge hub information
    df = df.merge(hubs, on="hub_id", how="left")

    # 10. Promised delivery hours
    promised_hours = {
        "Express": 36,
        "Standard": 60,
        "Economy": 96
    }

    df["promised_hours"] = df["service_type"].map(promised_hours)

    # 11. Additional modelling features
    df["log_declared_value"] = np.log1p(
        df["declared_value_inr"].clip(lower=0)
    )

    df["is_cod"] = (
        df["payment_mode"].str.upper() == "COD"
    ).astype(int)

    df["is_weekend"] = (
        df["dispatch_time"].dt.dayofweek >= 5
    ).astype(int)

    df["dispatch_hour"] = df["dispatch_time"].dt.hour

    # 12. Delayed shipment target
    df["is_delayed"] = (
        df["delivery_hours"] > df["promised_hours"]
    ).astype(int)

    # 13. Fill missing numeric feature values with medians
    feature_columns = [
        "distance_km", "parcel_weight_kg",
        "n_stops", "rain_flag", "sale_event_flag",
        "label_print_batch", "scanner_build_no",
        "tape_roll_no", "sticker_serial_mod",
        "shelf_slot_no", "hub_tier",
        "hub_daily_capacity", "log_declared_value",
        "dispatch_hour"
    ]

    for col in feature_columns:
        df[col] = df[col].fillna(df[col].median())

    # Remove rows with missing target
    df = df.dropna(subset=["delivery_hours", "promised_hours"])

    # 14. Save modelling table
    data_path = ROOT / "data" / "modelling_table.csv"
    df.to_csv(data_path, index=False)

    # 15. Create data quality checks
    checks = pd.DataFrame([
        ["Original rows", original_rows],
        ["Final rows", len(df)],
        ["Duplicate rows removed", original_rows - len(df)],
        ["Missing hub information", int(df["hub_city"].isna().sum())],
        ["Missing delivery hours", int(df["delivery_hours"].isna().sum())],
        ["Missing values in modelling features",
         int(df[feature_columns].isna().sum().sum())]
    ], columns=["check", "value"])

    checks.to_csv(ROOT / "outputs" / "checks.csv", index=False)

    # 16. Delay by group
    delay_by_group = (
        df.groupby(["service_type", "region"], dropna=False)
        .agg(
            shipment_count=("shipment_id", "count"),
            delayed_count=("is_delayed", "sum"),
            delay_rate=("is_delayed", "mean")
        )
        .reset_index()
    )

    delay_by_group.to_csv(
        ROOT / "outputs" / "delay_by_group.csv",
        index=False
    )

    # 17. Correlation analysis
    correlations = df.select_dtypes(include="number").corr()

    correlations.to_csv(
        ROOT / "outputs" / "correlations.csv"
    )

    print("Data preparation completed successfully!")
    print("Final shape:", df.shape)
    print("Modelling table and EDA files saved.")


if __name__ == "__main__":
    prepare_data()
