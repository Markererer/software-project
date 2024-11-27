from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config import MODELS_DIR, PROCESSED_DATA_DIR, ARTIFACT_DIR


# Ignoring this code for now, assembling the main.ipynb ML environment setup and training code here


import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import joblib
import datetime

import shutil
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json




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

# how de we get X_train and y_train?

X_train_path = ARTIFACT_DIR / "X_train.csv"
y_train_path = ARTIFACT_DIR / "y_train.csv"
X_test_path = ARTIFACT_DIR / "X_test.csv"
y_test_path = ARTIFACT_DIR / "y_test.csv"

#read the data
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)




model_grid.fit(X_train, y_train)



best_model_xgboost_params = model_grid.best_params_


y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)





xgboost_model = model_grid.best_estimator_
xgboost_model_path = MODELS_DIR / "lead_model_xgboost.json"
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
    lr_model_path = ARTIFACT_DIR / "lead_model_lr.pkl"

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



column_list_path = ARTIFACT_DIR / 'columns_list.json'
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    json.dump(columns, columns_file)



model_results_path = ARTIFACT_DIR / "model_results.json"
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)