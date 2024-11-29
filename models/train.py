from pathlib import Path
import os
import datetime
import json
import shutil
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier
import mlflow
import mlflow.pyfunc
import typer

from config import MODELS_DIR, PROCESSED_DATA_DIR, ARTIFACT_DIR


# somehow the code got broken. Could 









# Constants used:

data_gold_path = "itu-sdse-project-main/data/processed/train_data_gold.csv"

os.makedirs("itu-sdse-project-main/data/artifacts", exist_ok=True)
os.makedirs("itu-sdse-project-main/mlruns", exist_ok=True)
os.makedirs("itu-sdse-project-main/mlruns/.trash", exist_ok=True)



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


# Model training

model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)


# TODO change approach

# we get issues by using the parquet bs. Maybe passing the variables directly would solve these errors? They appear not in the ipynb file, so should be an easy fix if we just pass them as vars directly from the memory, without saving these into parquets.

model_grid.fit(X_train, y_train)

best_model_xgboost_params = model_grid.best_params_


y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)


xgboost_model = model_grid.best_estimator_
xgboost_model_path = "itu-sdse-project-main/models/lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}



class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


current_date = datetime.datetime.now().strftime("%Y_%B_%d")
experiment_name = current_date
mlflow.set_experiment(experiment_name)


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()
    lr_model_path = "itu-sdse-project-main/models/lead_model_lr.pkl"

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)


    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))


model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

model_results[lr_model_path] = model_classification_report



column_list_path = 'itu-sdse-project-main/data/artifacts/columns_list.json'
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    json.dump(columns, columns_file)



model_results_path = "itu-sdse-project-main/data/artifacts/model_results.json"
with open(model_results_path, 'w+') as results_file:
    
    json.dump(model_results, results_file)