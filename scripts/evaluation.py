import time
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import datetime
import mlflow
import os
import datetime
import json
import pandas as pd
import mlflow.pyfunc
import argparse
from mlflow_client import get_mlflow_client
import json

def wait_until_ready(model_name, model_version):
    client = get_mlflow_client()
    for _ in range(10):
        model_version_details = client.get_model_version(
        name=model_name,
        version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

def main(args):
    client = get_mlflow_client()

    # Constants used:
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    artifact_path = "model"
    model_name = "lead_model"
    experiment_name = current_date

    # Getting experiment model results
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]

    with open((os.path.join(args.artifacts_dir, "model_results.json")), "r") as f:
        model_results = json.load(f)
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"Best model: {best_model}")

    client = get_mlflow_client()
    prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']
    prod_model_exists = len(prod_model)>0

    # Get production model details
    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']
        
        print('Production model name: ', model_name)
        print('Production model version:', prod_model_version)
        print('Production model run id:', prod_model_run_id)
        
    else:
        print('No model in production')

    # Compare prod and best trained model
    train_model_score = experiment_best["metrics.f1_score"]
    model_details = {}
    model_status = {}
    run_id = None

    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score>prod_model_score:
            print("Registering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]

    print(f"Registered model: {run_id}")

    # Register best model
    if run_id is not None:
        print(f'Best model found: {run_id}')

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--artifacts_dir", type=str, required=True, help="Path to save artifacts")

    args = parser.parse_args()
    main(args)

