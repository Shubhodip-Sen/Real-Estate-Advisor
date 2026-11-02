"""Step 1 - Data preprocessing & feature engineering.

Run:  python src/preprocess.py
Reads data/raw/india_housing_prices.csv -> writes data/processed/housing_clean.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *  # noqa: E402,F403


def add_growth_and_label(df: pd.DataFrame) -> pd.DataFrame:
    g = GROWTH
    df["Annual_Growth_Rate"] = (
        g["base"]
        + g["infra_weight"] * df["Infrastructure_Score"]
        + g["metro_bonus"] * df["Is_Metro"]
        + g["under_construction"] * (df["Availability_Status"] == "Under_Construction")
        - g["age_penalty"] * df["Age_of_Property"]
    ).clip(g["min"], g["max"])
    df["Future_Price_5Y_Lakhs"] = (
        df["Price_in_Lakhs"] * (1 + df["Annual_Growth_Rate"]) ** HORIZON_YEARS
    ).round(2)
    city_med = df.groupby("City")["Price_per_SqFt_INR"].transform("median")
    df["Below_City_Median_PPSF"] = (df["Price_per_SqFt_INR"] <= city_med).astype(int)
    df["Good_Investment"] = (
        (df["Below_City_Median_PPSF"] == 1)
        & (df["Annual_Growth_Rate"] >= GOOD_INVESTMENT_GROWTH_THRESHOLD)
    ).astype(int)
    return df


def main():
    df = pd.read_csv(RAW_CSV)
    log = {"rows_raw": len(df)}

    # --- quality checks ------------------------------------------------------
    log["missing_values"] = int(df.isna().sum().sum())
    cols_wo_id = [c for c in df.columns if c != "ID"]
    log["duplicate_rows"] = int(df.duplicated(subset=cols_wo_id).sum())
    df = df.drop_duplicates(subset=cols_wo_id).copy()
    text_cols = df.select_dtypes(exclude="number").columns
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip()
    num = [c for c in df.select_dtypes("number").columns if c != "ID"]
    df[num] = df[num].fillna(df[num].median())          # defensive imputation (none needed here)

    # --- logical consistency -------------------------------------------------
    log["floor_gt_total_floors"] = int((df["Floor_No"] > df["Total_Floors"]).sum())
    df["Floor_Data_Flag"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)   # keep the evidence
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])             # cap to a valid floor
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

    # --- outlier flags (IQR) - flagged, not dropped (distributions are bounded) ---
    for c in ["Price_in_Lakhs", "Size_in_SqFt"]:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        df[f"{c}_Outlier"] = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).astype(int)

    # --- feature engineering -------------------------------------------------
    # Raw Price_per_SqFt is in lakhs and rounded to 2 dp (0.05, 0.10 ...) -> recompute precisely in INR.
    df["Price_per_SqFt_INR"] = (df["Price_in_Lakhs"] * LAKH / df["Size_in_SqFt"]).round(2)
    q1, q3 = df["Price_per_SqFt_INR"].quantile([.25, .75])
    df["PPSF_Outlier"] = (df["Price_per_SqFt_INR"] > q3 + 1.5 * (q3 - q1)).astype(int)
    log["ppsf_outliers_iqr"] = int(df["PPSF_Outlier"].sum())
    for c in ["Price_in_Lakhs", "Size_in_SqFt"]:
        log[f"{c}_outliers_iqr"] = int(df[f"{c}_Outlier"].sum())

    df["Locality_Key"] = df["City"] + " | " + df["Locality"]    # locality labels repeat across cities
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP)
    df["School_Density_Score"] = df["Nearby_Schools"] / 10
    df["Hospital_Density_Score"] = df["Nearby_Hospitals"] / 10
    df["Infrastructure_Score"] = (
        (df["School_Density_Score"] + df["Hospital_Density_Score"] + df["Transport_Score"]) / 3
    ).round(4)
    df["Is_Metro"] = df["City"].isin(METRO_CITIES).astype(int)
    df["Amenity_Count"] = df["Amenities"].str.split(", ").str.len()
    for a in AMENITY_LIST:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a).astype(int)
    df["Parking_Flag"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Security_Flag"] = (df["Security"] == "Yes").astype(int)
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"]).round(3)
    df["Size_per_BHK"] = (df["Size_in_SqFt"] / df["BHK"]).round(1)
    df["Age_Bucket"] = pd.cut(df["Age_of_Property"], [-1, 5, 15, 25, 100],
                              labels=["0-5y", "6-15y", "16-25y", "26y+"]).astype(str)

    df = add_growth_and_label(df)
    log["rows_clean"] = len(df)
    log["good_investment_rate"] = round(float(df["Good_Investment"].mean()), 4)

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "preprocess_log.json").write_text(json.dumps(log, indent=2))
    print("Preprocessing complete ->", CLEAN_CSV)
    for k, v in log.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
