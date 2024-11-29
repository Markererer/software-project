import os
import pandas as pd
import warnings
import argparse
import pickle #🥒

# Set up warnings and pandas options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def main(args):
    # Read the data
    data = pd.read_csv(os.path.join(args.processed_data_dir, "train_data_gold.csv"))

    # Process the data
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")

    print("Features extracted.")        

    # Instantiate feature space
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    # Save as a 🥒 file to pass between scripts
    with open(args.internim_data_dir + "/X.pkl", "wb") as f:
        pickle.dump(X, f)
    with open(args.internim_data_dir + "/y.pkl", "wb") as f:
        pickle.dump(y, f)

    print("Training and test data saved into the internim directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")
    parser.add_argument("--internim_data_dir", type=str, required=True, help="Path to save internim data")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to save processed data")

    args = parser.parse_args()
    main(args)