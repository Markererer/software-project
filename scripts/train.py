import os
import datetime
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRFClassifier
import mlflow
import mlflow.pyfunc
import warnings
import argparse
import pickle #🥒

# Set up warnings and pandas options
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

def main(args):
    # Ensure the MLRuns folder exists
    os.makedirs(args.mlruns_dir, exist_ok=True)
    os.makedirs(os.path.join(args.mlruns_dir, ".trash"), exist_ok=True)

    # Load X
    with open(args.interim_data_dir + "/X.pkl", "rb") as f:
        X = pickle.load(f)

    # Load y
    with open(args.interim_data_dir + "/y.pkl", "rb") as f:
        y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    experiment_name = current_date
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id) as run:
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
        model_grid.fit(X_train, y_train)

        best_model_xgboost_params = model_grid.best_params_
        print("The best-performing XGBoost model parameters were: " + str(best_model_xgboost_params))

        y_pred_test = model_grid.predict(X_test)
        xgboost_model = model_grid.best_estimator_

        # Log XGBoost metrics and model
        #mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        #for param_name, param_value in best_model_xgboost_params.items():
        #    mlflow.log_param(f"xgb_{param_name}", param_value)
        #mlflow.xgboost.log_model(xgboost_model, artifact_path="model")

        # Store model for interpretability
        lr_model_path = (os.path.join(args.artifacts_dir, "lead_model_xgboost.pkl"))
        joblib.dump(value=model, filename=lr_model_path)

        #xgb_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()

        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty': ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid = RandomizedSearchCV(model, param_distributions=params, verbose=3, n_iter=10, cv=3)
        model_grid.fit(X_train, y_train)

        y_pred_test = model_grid.predict(X_test)

        # Log LR metrics and artifacts
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")

        # Store model for interpretability
        lr_model_path = (os.path.join(args.artifacts_dir, "lead_model_lr.pkl"))
        joblib.dump(value=model, filename=lr_model_path)

        # Custom python model for predicting probability
        mlflow.pyfunc.log_model('model', python_model=model)

        lr_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

    model_results = {
        #"XGBoost": xgb_classification_report,
        "LogisticRegression": lr_classification_report
    }

    column_list_path = (os.path.join(args.artifacts_dir, "columns_list.json"))
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        json.dump(columns, columns_file)

    model_results_path = (os.path.join(args.artifacts_dir, "model_results.json"))
    with open(model_results_path, 'w+') as results_file:        
        json.dump(model_results, results_file)

    print("Train script finished without errors.")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")
    parser.add_argument("--interim_data_dir", type=str, required=True, help="Path to save interim data")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to save processed data")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Path to save artifacts")
    parser.add_argument("--mlruns_dir", type=str, required=True, help="Path to save MLFlow runs")

    args = parser.parse_args()
    main(args)