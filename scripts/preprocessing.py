import os
import argparse
from pprint import pprint
import pandas as pd
import warnings
import datetime
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set up warnings and pandas options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

def describe_numeric_col(x):
    """Describe a numeric Pandas Series."""
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """Impute missing values in a Pandas Series."""
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def main(args):
    # Ensure artifacts directory exists
    os.makedirs(args.artifacts_dir, exist_ok=True)
    print(f"Created artifacts directory: {args.artifacts_dir}")

    # Read the data
    print("Loading training data")
    data = pd.read_csv(os.path.join(args.raw_data_dir, "raw_data.csv"))

    max_date = pd.to_datetime(args.max_date).date() if args.max_date else pd.to_datetime(datetime.datetime.now().date()).date()
    min_date = pd.to_datetime(args.min_date).date()

    # Filter data by date
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open(os.path.join(args.artifacts_dir, "date_limits.json"), "w") as f:
        json.dump(date_limits, f)

    # Feature selection
    data = data.drop(
        ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
        axis=1
    )
    data = data.drop(["domain", "country", "visited_learn_more_before_booking", "visited_faq"], axis=1)

    # Data cleaning
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    data = data[data.source == "signup"]
    result = data.lead_indicator.value_counts(normalize=True)

    vars = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")

    # Separate categorical and numerical columns
    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    # Handle outliers
    cont_vars = cont_vars.apply(lambda x: x.clip(lower=(x.mean() - 2 * x.std()), upper=(x.mean() + 2 * x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(os.path.join(args.artifacts_dir, "outlier_summary.csv"))

    # Impute missing values
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(os.path.join(args.artifacts_dir, "cat_missing_impute.csv"))

    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars.loc[cat_vars['customer_code'].isna(), 'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values)

    # Data standardization
    scaler_path = os.path.join(args.artifacts_dir, "scaler.pkl")
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    # Combine categorical and continuous variables
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    print(f"Data cleansed and combined.\nRows: {len(data)}")

    # Save column names for drift detection
    data_columns = list(data.columns)
    with open(os.path.join(args.artifacts_dir, "columns_drift.json"), "w+") as f:
        json.dump(data_columns, f)

    # Save training data
    data.to_csv(os.path.join(args.artifacts_dir, "training_data.csv"), index=False)

    # Bin source column
    data['bin_source'] = data['source']
    values_list = ['li', 'organic', 'signup', 'fb']
    data.loc[~data['source'].isin(values_list), 'bin_source'] = 'Others'
    mapping = {
        'li': 'socials',
        'fb': 'socials',
        'organic': 'group1',
        'signup': 'group1'
    }
    data['bin_source'] = data['source'].map(mapping)

    # Save final dataset
    data.to_csv(os.path.join(args.processed_data_dir, "train_data_gold.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Path to save artifacts")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--min_date", type=str, required=True, help="Minimum date for filtering (YYYY-MM-DD)")
    parser.add_argument("--max_date", type=str, required=False, help="Maximum date for filtering (YYYY-MM-DD)")

    args = parser.parse_args()
    main(args)
