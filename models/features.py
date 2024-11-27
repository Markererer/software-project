# This is the code to create the features for the ML models to work with. 
from pathlib import Path
import os
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import typer
from loguru import logger
from tqdm import tqdm

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ARTIFACT_DIR

# Constants used:

data_gold_path = PROCESSED_DATA_DIR / "train_data_gold.csv"

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("mlruns/.trash", exist_ok=True)



def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


data = pd.read_csv(data_gold_path)

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
    

y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

# save X_train and y_train and X_test and y_test into a readable file
X_train.to_csv(ARTIFACT_DIR / "X_train.csv", index=False)
X_test.to_csv(ARTIFACT_DIR / "X_test.csv", index=False)
y_train.to_csv(ARTIFACT_DIR / "y_train.csv", index=False)
y_test.to_csv(ARTIFACT_DIR / "y_test.csv", index=False)

logger.info("Data split and saved successfully")